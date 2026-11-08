"""Scaled FP8 linear using PyTorch, preserving offload and PEFT wrappers."""
import ctypes as c
import os
import warnings
import torch


_quantizer=None
_quantizer_device=None
_quantizer_unavailable=False
_SOURCE=r'''
extern "C" __global__ void maximum(const unsigned short* input,int* output,long long count) {
    __shared__ int maxima[256];
    int value=0,t=threadIdx.x;
    for(long long i=(long long)blockIdx.x*256+t;i<count;i+=(long long)gridDim.x*256)
        value=max(value,(int)(input[i]&0x7fff));
    maxima[t]=value;__syncthreads();
    for(int offset=128;offset;offset/=2){if(t<offset)maxima[t]=max(maxima[t],maxima[t+offset]);__syncthreads();}
    if(t==0)atomicMax(output,maxima[0]);
}
extern "C" __global__ void scale_maximum(const int* maximum,float* output,int bf16) {
    unsigned short bits=(unsigned short)maximum[0];
    float value;
    if(bf16)value=__uint_as_float(((unsigned int)bits)<<16);
    else asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    value=value*(1.0f/448.0f);
    output[0]=value<1.0e-12f ? 1.0e-12f : value;
}
extern "C" __global__ void decode(const unsigned short* input,const float* scale,
    unsigned char* output,long long count,int bf16) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;
    if(i>=count)return;
    float value;
    if(bf16)value=__uint_as_float(((unsigned int)input[i])<<16);
    else asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(input[i]));
    value=value/scale[0];
    if(value==value)value=fminf(448.0f,fmaxf(-448.0f,value));
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(value));
    output[i]=(unsigned char)pair;
}
'''


def fused_kernel(values):
    global _quantizer,_quantizer_device,_quantizer_unavailable
    if (os.name=='nt' and values.is_cuda and values.dtype in (torch.bfloat16,torch.float16) and values.numel()>0
            and not _quantizer_unavailable and values.device.index==torch.cuda.current_device()):
        if _quantizer is None:
            from backend.app.inference.ops.cuda_decode import Kernel
            try:
                _quantizer=Kernel(source=_SOURCE)
                for name in ('maximum','scale_maximum'):
                    function=c.c_void_p()
                    _quantizer.check(_quantizer.driver.cuModuleGetFunction(c.byref(function),_quantizer.module,name.encode()))
                    setattr(_quantizer,name,function)
                _quantizer_device=values.device
            except (OSError,RuntimeError) as error:
                _quantizer_unavailable=True
                warnings.warn('Fused FP8 input conversion unavailable; using Torch: '+str(error),RuntimeWarning)
        if _quantizer is not None and _quantizer_device==values.device:return _quantizer
    return None


def input_scale(values):
    kernel=fused_kernel(values)
    if kernel is None:return (values.abs().amax().float()/448).clamp_min(1e-12)
    values=values.contiguous()
    maximum=torch.zeros((),device=values.device,dtype=torch.int32)
    scale=maximum.view(torch.float32)
    for function,blocks,args in (
        (kernel.maximum,min(1024,(values.numel()+255)//256),[c.c_void_p(values.data_ptr()),c.c_void_p(maximum.data_ptr()),c.c_longlong(values.numel())]),
        (kernel.scale_maximum,1,[c.c_void_p(maximum.data_ptr()),c.c_void_p(scale.data_ptr()),c.c_int(values.dtype==torch.bfloat16)])):
        pointers=(c.c_void_p*len(args))(*(c.addressof(x) for x in args))
        kernel.check(kernel.driver.cuLaunchKernel(function,blocks,1,1,256 if function is kernel.maximum else 1,1,1,0,
            c.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None))
    return scale


def quantize_input(values,scale):
    kernel=fused_kernel(values)
    if kernel is not None:
        values=values.contiguous()
        output=torch.empty_like(values,dtype=torch.float8_e4m3fn)
        args=[c.c_void_p(x.data_ptr()) for x in (values,scale,output)]+[c.c_longlong(values.numel()),c.c_int(values.dtype==torch.bfloat16)]
        pointers=(c.c_void_p*len(args))(*(c.addressof(x) for x in args))
        kernel.check(kernel.driver.cuLaunchKernel(kernel.function,(values.numel()+255)//256,1,1,256,1,1,0,
            c.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None))
        return output
    return (values.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)


def supported_input(x):
    return (torch.version.cuda is not None and x.is_cuda and x.dtype in (torch.bfloat16,torch.float16) and x.ndim>=2
            and x.shape[-1]>0 and x.shape[-1]%16==0 and x.numel()>0
            and torch.cuda.get_device_capability(x.device)>=(8,9))


def native_linear(x,weight,scale):
    if not supported_input(x):return None
    if not (weight.dtype==torch.float8_e4m3fn and weight.ndim==2 and weight.is_contiguous()
            and weight.device==x.device and weight.shape[1]==x.shape[-1] and weight.shape[0]%16==0
            and scale.dtype==torch.float32 and scale.numel()==1 and scale.device==x.device):return None
    values=x.reshape(-1,x.shape[-1])
    rows=values.shape[0]
    # Dynamic scaling avoids clipping large activations; zero rows retain a finite scale.
    scale_a=input_scale(values)
    quantized=quantize_input(values,scale_a)
    if rows%16:quantized=torch.nn.functional.pad(quantized,(0,0,0,16-rows%16))
    output=torch._scaled_mm(quantized,weight.t(),scale_a=scale_a,scale_b=scale.reshape(()),
                            out_dtype=x.dtype,use_fast_accum=False)
    return output[:rows].reshape(*x.shape[:-1],weight.shape[0])


def attach_native_linear(model,mapping):
    for name,module in model.named_modules():
        key=(name+'.' if name else '')+'weight'
        if (not isinstance(module,torch.nn.Linear) or key not in mapping.packed_sources
                or mapping.packed_sources[key][2] is not None or key in mapping.full_precision_sources
                or mapping.sources[key][1] not in (torch.bfloat16,torch.float16)
                or module.in_features%16 or module.out_features%16 or not hasattr(module,'_old_forward')):continue
        mapping.native_fp8_sources.add(key)
        bias_key=(name+'.' if name else '')+'bias'
        def forward(input,*args,_module=module,_key=key,_bias=bias_key,_old=module.forward,**kwargs):
            if not args and not kwargs and supported_input(input):
                weight,scale,_=mapping.linear_packed(_key,input.device)
                result=native_linear(input,weight,scale)
                if result is not None:
                    mapping.fp8_hits+=1
                    mapping.fp8_layers[_key]=mapping.fp8_layers.get(_key,0)+1
                    return result if _module.bias is None else result+mapping[_bias].to(input.device)
            mapping.fp8_fallbacks+=1
            return _old(input,*args,**kwargs)
        module.forward=forward
