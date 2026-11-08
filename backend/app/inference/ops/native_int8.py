"""Krea2 ConvRot INT8 linear, preserving existing offload and LoRA wrappers."""
import ctypes as c
import os
import warnings
import torch
from backend.app.inference.ops.cuda_decode import Kernel

SOURCE=r'''
__device__ __forceinline__ float rotate16(float value,int lane) {
 int base=lane&~3;
 float sum=((__shfl_sync(0xffffffff,value,base)+__shfl_sync(0xffffffff,value,base+1))
            +__shfl_sync(0xffffffff,value,base+2))+__shfl_sync(0xffffffff,value,base+3);
 value=sum-2*__shfl_sync(0xffffffff,value,base+3-(lane&3));
 base=lane&~12;
 sum=((__shfl_sync(0xffffffff,value,base)+__shfl_sync(0xffffffff,value,base+4))
       +__shfl_sync(0xffffffff,value,base+8))+__shfl_sync(0xffffffff,value,base+12);
 return sum-2*__shfl_sync(0xffffffff,value,base+(3-((lane/4)%4))*4);
}
template<int WIDTH>
__device__ void rotate256(const unsigned short* x,float* scale,signed char* q) {
 __shared__ float maxima[8];
 int t=threadIdx.x,lane=t%32,warp=t/32,row=blockIdx.x;
 float values[WIDTH/256],maximum=0.0f;
 #pragma unroll
 for(int j=0;j<WIDTH/2048;j++) {
  int base=warp*256+j*2048+lane;
  float v[8];
  #pragma unroll
  for(int k=0;k<8;k++)v[k]=rotate16(__uint_as_float(((unsigned int)x[(long long)row*WIDTH+base+k*32])<<16),lane);
  #pragma unroll
  for(int k=0;k<8;k+=2) {
   float a=v[k],b=v[k+1];
   float sum=((__shfl_sync(0xffffffff,a,lane&15)+__shfl_sync(0xffffffff,a,(lane&15)+16))
              +__shfl_sync(0xffffffff,b,lane&15))+__shfl_sync(0xffffffff,b,(lane&15)+16);
   v[k]=sum-2*__shfl_sync(0xffffffff,b,lane^16);
   v[k+1]=sum-2*__shfl_sync(0xffffffff,a,lane^16);
  }
  #pragma unroll
  for(int p=0;p<2;p++) {
   float a=v[p],b=v[p+2],c=v[p+4],d=v[p+6];
   float sum=((a+b)+c)+d;
   v[p]=sum-2*d;v[p+2]=sum-2*c;v[p+4]=sum-2*b;v[p+6]=sum-2*a;
  }
  #pragma unroll
  for(int k=0;k<8;k++) {
   values[j*8+k]=v[k]*0.0625f;
   maximum=fmaxf(maximum,fabsf(values[j*8+k]));
  }
 }
 for(int offset=16;offset;offset/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
 if(lane==0)maxima[warp]=maximum;
 __syncthreads();
 if(warp==0) {
  maximum=lane<8?maxima[lane]:0.0f;
  for(int offset=16;offset;offset/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
  if(lane==0){maxima[0]=fmaxf(maximum/127.0f,1.0e-12f);scale[row]=maxima[0];}
 }
 __syncthreads();
 float s=maxima[0];
 #pragma unroll
 for(int j=0;j<WIDTH/2048;j++) {
  int base=warp*256+j*2048+lane;
  #pragma unroll
  for(int k=0;k<8;k++)q[(long long)row*WIDTH+base+k*32]=(signed char)__float2int_rn(fminf(127.0f,fmaxf(-127.0f,values[j*8+k]/s)));
 }
}
extern "C" __global__ void rotate256_6144(const unsigned short* x,float* s,signed char* q,int width,int group){rotate256<6144>(x,s,q);}
extern "C" __global__ void rotate256_16384(const unsigned short* x,float* s,signed char* q,int width,int group){rotate256<16384>(x,s,q);}

extern "C" __global__ void decode(const unsigned short* x,float* scale,signed char* q,int width,int group) {
 extern __shared__ float values[];
 __shared__ float maxima[256];
 int t=threadIdx.x,row=blockIdx.x;
 float next[64];
 for(int i=t;i<width;i+=256)values[i]=__uint_as_float(((unsigned int)x[(long long)row*width+i])<<16);
 __syncthreads();
 for(int stride=1;stride<group;stride*=4) {
  for(int i=t;i<width;i+=256) {
   int digit=(i/stride)%4,base=i-digit*stride;
   next[i/256]=values[base]+values[base+stride]+values[base+2*stride]+values[base+3*stride]-2*values[base+(3-digit)*stride];
  }
  __syncthreads();
  for(int i=t;i<width;i+=256)values[i]=next[i/256];
  __syncthreads();
 }
 float norm=group==16?0.25f:group==64?0.125f:0.0625f,maximum=0;
 for(int i=t;i<width;i+=256)maximum=fmaxf(maximum,fabsf(values[i]*norm));
 maxima[t]=maximum;__syncthreads();
 for(int offset=128;offset;offset/=2){if(t<offset)maxima[t]=fmaxf(maxima[t],maxima[t+offset]);__syncthreads();}
 float s=fmaxf(maxima[0]/127.0f,1.0e-12f);
 if(t==0)scale[row]=s;
 for(int i=t;i<width;i+=256)q[(long long)row*width+i]=(signed char)__float2int_rn(fminf(127.0f,fmaxf(-127.0f,values[i]*norm/s)));
}
extern "C" __global__ void finish(const int* sums,const float* xs,const float* ws,unsigned short* out,long long count,int width) {
 long long i=(long long)blockIdx.x*256+threadIdx.x;
 if(i<count){float v=((float)sums[i]*xs[i/width])*ws[i%width];unsigned int bits=__float_as_uint(v);out[i]=(unsigned short)((bits+0x7fffu+((bits>>16)&1u))>>16);}
}
'''

class NativeLinear:
    def __init__(self):
        self.kernel=Kernel(source=SOURCE)
        self.kernel.driver.cuFuncSetAttribute.argtypes=[c.c_void_p,c.c_int,c.c_int]
        # Opt in to >48 KiB shared memory for the 16384-wide MLP input.
        self.kernel.check(self.kernel.driver.cuFuncSetAttribute(self.kernel.function,8,65536))
        self.fast={};self.fast_calls=0
        for width in (6144,16384):
            function=c.c_void_p()
            self.kernel.check(self.kernel.driver.cuModuleGetFunction(c.byref(function),self.kernel.module,('rotate256_'+str(width)).encode()))
            self.fast[width]=function
        self.finish=c.c_void_p()
        self.kernel.check(self.kernel.driver.cuModuleGetFunction(c.byref(self.finish),self.kernel.module,b'finish'))

    def launch(self,function,blocks,args,shared=0):
        pointers=(c.c_void_p*len(args))(*(c.addressof(value) for value in args))
        self.kernel.check(self.kernel.driver.cuLaunchKernel(function,blocks,1,1,256,1,1,shared,
            c.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None))

    def __call__(self,x,weight,scale,group):
        rows=x.numel()//x.shape[-1];width=x.shape[-1]
        values=x.reshape(rows,width).contiguous()
        quantized=torch.empty_like(values,dtype=torch.int8)
        input_scale=torch.empty((rows,1),device=x.device,dtype=torch.float32)
        fast=group==256 and width in self.fast and rows>=1024
        if fast:self.fast_calls+=1
        self.launch(self.fast[width] if fast else self.kernel.function,rows,[c.c_void_p(v.data_ptr()) for v in (values,input_scale,quantized)]+[c.c_int(width),c.c_int(group)],shared=0 if fast else width*4)
        sums=torch._int_mm(quantized,weight.t())
        output=torch.empty_like(sums,dtype=torch.bfloat16)
        self.launch(self.finish,(output.numel()+255)//256,[c.c_void_p(v.data_ptr()) for v in (sums,input_scale,scale,output)]+[c.c_longlong(output.numel()),c.c_int(weight.shape[0])])
        return output.reshape(*x.shape[:-1],weight.shape[0])


_kernel=None
_unavailable=False


def native_linear(x,weight,scale,group):
    global _kernel,_unavailable
    if os.name!='nt' or _unavailable:return None
    # shortcut: inputs wider than the validated MLP shape retain the BF16 path.
    if not (x.is_cuda and x.dtype==torch.bfloat16 and x.ndim>=2 and 0<x.shape[-1]<=16384):return None
    if not (weight.ndim==2 and weight.dtype==torch.int8 and weight.device==x.device and weight.is_contiguous()):return None
    if not (type(group) is int and group in (16,64,256) and weight.shape[1]==x.shape[-1] and x.shape[-1]%group==0 and weight.shape[0]%8==0):return None
    if not (scale.dtype==torch.float32 and scale.device==x.device and scale.shape==(weight.shape[0],1) and scale.is_contiguous()):return None
    if x.device.index!=torch.cuda.current_device() or x.numel()//x.shape[-1]<16:return None
    if _kernel is None:
        try:_kernel=NativeLinear()
        except (OSError,RuntimeError) as error:
            _unavailable=True
            warnings.warn('Native INT8 unavailable; using BF16: '+str(error),RuntimeWarning)
            return None
    return _kernel(x,weight,scale,group)


def attach_native_linear(model,mapping):
    for name,module in model.named_modules():
        key=(name+'.' if name else '')+'weight'
        if not isinstance(module,torch.nn.Linear) or key not in mapping.packed_sources or not mapping.packed_sources[key][2] or module.in_features>16384 or not hasattr(module,'_old_forward'):continue
        bias_key=(name+'.' if name else '')+'bias'
        def forward(input,*args,_module=module,_key=key,_bias=bias_key,_old=module.forward,**kwargs):
            x=input
            if os.name=='nt' and not _unavailable and not args and not kwargs and x.is_cuda and x.dtype==torch.bfloat16 and x.ndim>=2 and x.numel()//x.shape[-1]>=16:
                packed=mapping.linear_packed(_key,x.device)
                result=native_linear(x,*packed)
                if result is not None:
                    mapping.native_hits+=1
                    mapping.native_layers[_key]=mapping.native_layers.get(_key,0)+1
                    return result if _module.bias is None else result+mapping[_bias].to(x.device)
            return _old(x,*args,**kwargs)
        module.forward=forward
