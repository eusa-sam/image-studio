"""Fuse the existing BF16 rotary formula without changing its rounding."""
import ctypes as c
import os
import warnings
from contextlib import contextmanager
import torch

_kernel=None
_device=None
_unavailable=False
_SOURCE=r'''
extern "C" __global__ void decode(const unsigned short* input,const float* cosine,
    const float* sine,unsigned short* output,long long count,int width,int heads,int sequence) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;
    if(i>=count)return;
    int position=(i/(width*heads))%sequence;
    int frequency=position*width+i%width;
    float value=__uint_as_float(((unsigned int)input[i])<<16);
    float rotated=__uint_as_float(((unsigned int)input[i^1])<<16);
    if(!(i&1))rotated=-rotated;
    float result=value*cosine[frequency]+rotated*sine[frequency];
    unsigned int bits=__float_as_uint(result);
    output[i]=(unsigned short)((bits+0x7fffu+((bits>>16)&1u))>>16);
}
'''


def native_rotary(x,freqs):
    global _kernel,_device,_unavailable
    if (os.name!='nt' or _unavailable or torch.is_grad_enabled() or not x.is_cuda or x.dtype!=torch.bfloat16
            or x.ndim!=4 or x.numel()==0 or x.shape[-1]%2 or not x.is_contiguous()
            or x.device.index!=torch.cuda.current_device()):return None
    if not isinstance(freqs,(tuple,list)) or len(freqs)!=2:return None
    if any(v.device!=x.device or v.dtype!=torch.float32 or not v.is_contiguous()
           or tuple(v.shape)!=(x.shape[1],x.shape[-1]) for v in freqs):return None
    if _kernel is None:
        from backend.app.inference.ops.cuda_decode import Kernel
        try:
            _kernel=Kernel(source=_SOURCE);_device=x.device
        except (OSError,RuntimeError) as error:
            _unavailable=True
            warnings.warn('Fused Krea2 rotary unavailable; using Torch: '+str(error),RuntimeWarning)
            return None
    if _device!=x.device:return None
    output=torch.empty_like(x)
    args=[c.c_void_p(v.data_ptr()) for v in (x,*freqs,output)]+[c.c_longlong(x.numel()),c.c_int(x.shape[3]),c.c_int(x.shape[2]),c.c_int(x.shape[1])]
    pointers=(c.c_void_p*len(args))(*(c.addressof(v) for v in args))
    _kernel.check(_kernel.driver.cuLaunchKernel(_kernel.function,(x.numel()+255)//256,1,1,256,1,1,0,
        c.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None))
    return output


@contextmanager
def fused_rotary():
    from diffusers.models.transformers import transformer_krea2 as module
    original=module.apply_rotary_emb
    def apply(x,freqs_cis,use_real=True,use_real_unbind_dim=-1,sequence_dim=2):
        if use_real and use_real_unbind_dim==-1 and sequence_dim==1:
            result=native_rotary(x,freqs_cis)
            if result is not None:return result
        return original(x,freqs_cis,use_real,use_real_unbind_dim,sequence_dim)
    module.apply_rotary_emb=apply
    try:yield
    finally:module.apply_rotary_emb=original
