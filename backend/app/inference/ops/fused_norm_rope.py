"""BF16 RMSNorm and RoPE with the original intermediate BF16 rounding."""
import ctypes as c
import os
import warnings
import torch

_kernel=None
_device=None
_unavailable=False
_SOURCE=r'''
__device__ float bf(unsigned short v) {return __uint_as_float(((unsigned int)v)<<16);}
__device__ unsigned short round_bf(float v) {
 unsigned int b=__float_as_uint(v);
 if((b&0x7fffffffu)>0x7f800000u)return 0x7fc0u;
 return (unsigned short)((b+0x7fffu+((b>>16)&1u))>>16);
}
extern "C" __global__ void decode(const unsigned short* input,const unsigned short* scale,
 const float* cosine,const float* sine,unsigned short* output,int width,int heads,int sequence,
 float epsilon,int interleaved) {
 __shared__ float values[256];__shared__ float squares[256];
 int t=threadIdx.x;long long i=(long long)blockIdx.x*width+t;
 float x=t<width ? bf(input[i]) : 0.0f;squares[t]=x*x;__syncthreads();
 for(int stride=128;stride>0;stride/=2) {
  if(t<stride)squares[t]+=squares[t+stride];__syncthreads();
 }
 if(t<width)values[t]=bf(round_bf(bf(round_bf(x*rsqrtf(squares[0]/width+epsilon)))*bf(scale[t])));
 __syncthreads();
 if(t<width) {
  if(interleaved==2) {output[i]=round_bf(values[t]);return;}
  int partner=interleaved ? (t^1) : (t+width/2)%width;
  float rotated=values[partner];if(interleaved ? !(t&1) : t<width/2)rotated=-rotated;
  int pos=blockIdx.x/heads;int freq=interleaved ? pos*width+(t/2)*2 : (pos%sequence)*width+t;
  float co=cosine[freq];float si=interleaved ? cosine[freq+1] : sine[freq];
  output[i]=round_bf(values[t]*co+rotated*si);
 }
}
'''


def native_norm_rope(x,norm,frequencies,interleaved):
    global _kernel,_device,_unavailable
    from diffusers.models.normalization import RMSNorm
    if (os.name!='nt' or _unavailable or torch.is_grad_enabled() or not x.is_cuda or x.dtype!=torch.bfloat16
        or x.ndim!=4 or not x.numel() or not x.is_contiguous() or x.shape[-1] not in (32,64,128,256)
        or type(norm) is not RMSNorm or norm.bias is not None or norm.weight is None
        or norm.weight.dtype!=x.dtype or norm.weight.device!=x.device or not norm.weight.is_contiguous()
        or tuple(norm.weight.shape)!=(x.shape[-1],) or x.device.index!=torch.cuda.current_device()):return None
    if frequencies is None:
        cosine=sine=x
    elif interleaved:
        if not isinstance(frequencies,torch.Tensor) or frequencies.dtype!=torch.complex64 or tuple(frequencies.shape)!=(x.shape[0],x.shape[1],x.shape[3]//2):return None
        cosine=sine=frequencies
    else:
        if not isinstance(frequencies,(tuple,list)) or len(frequencies)!=2:return None
        cosine,sine=frequencies
        if any(v.dtype!=torch.float32 or tuple(v.shape)!=(x.shape[1],x.shape[3]) for v in (cosine,sine)):return None
    if any(v.device!=x.device or not v.is_contiguous() for v in (cosine,sine)):return None
    if _kernel is None:
        from backend.app.inference.ops.cuda_decode import Kernel
        try:_kernel=Kernel(source=_SOURCE);_device=x.device
        except (OSError,RuntimeError) as error:
            _unavailable=True;warnings.warn('Fused RMSNorm/RoPE unavailable; using Torch: '+str(error),RuntimeWarning);return None
    if x.device!=_device:return None
    output=torch.empty_like(x)
    args=[c.c_void_p(v.data_ptr()) for v in (x,norm.weight,cosine,sine,output)]+[c.c_int(x.shape[3]),c.c_int(x.shape[2]),c.c_int(x.shape[1]),c.c_float(norm.eps),c.c_int(2 if frequencies is None else interleaved)]
    pointers=(c.c_void_p*len(args))(*(c.addressof(v) for v in args))
    _kernel.check(_kernel.driver.cuLaunchKernel(_kernel.function,x.numel()//x.shape[-1],1,1,256,1,1,0,c.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None))
    return output


class FusedProcessor:
    def __init__(self,original,interleaved):self.original,self.interleaved=original,interleaved

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,image_rotary_emb=None,freqs_cis=None):
        frequencies=freqs_cis if self.interleaved else image_rotary_emb
        if torch.is_grad_enabled() or (self.interleaved and encoder_hidden_states is not None):
            return self.original(attn,hidden_states,encoder_hidden_states,attention_mask,**({'freqs_cis':freqs_cis} if self.interleaved else {'image_rotary_emb':image_rotary_emb}))
        context=hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query,key,value=[layer(tensor).unflatten(-1,(attn.heads,-1)) for layer,tensor in ((attn.to_q,hidden_states),(attn.to_k,context),(attn.to_v,context))]
        results=[]
        for tensor,norm in ((query,attn.norm_q),(key,attn.norm_k)):
            result=native_norm_rope(tensor,norm,frequencies,self.interleaved)
            if result is None:
                tensor=norm(tensor) if norm is not None else tensor
                if frequencies is None:result=tensor
                elif self.interleaved:
                    result=torch.view_as_real(torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1],-1,2))*frequencies.unsqueeze(2)).flatten(3).to(tensor.dtype)
                else:
                    from diffusers.models.embeddings import apply_rotary_emb
                    result=apply_rotary_emb(tensor,frequencies,use_real_unbind_dim=-2,sequence_dim=1)
            results.append(result)
        query,key=results
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        options={}
        if self.interleaved:
            if attention_mask is not None and attention_mask.ndim==2:attention_mask=attention_mask[:,None,None,:]
            options=dict(backend=self.original._attention_backend,parallel_config=self.original._parallel_config)
        else:
            key=key.repeat_interleave(query.shape[-1]//key.shape[-1],dim=-1)
            value=value.repeat_interleave(query.shape[-1]//value.shape[-1],dim=-1)
        result=dispatch_attention_fn(query,key,value,attn_mask=attention_mask,dropout_p=0.0,is_causal=False,**options).flatten(2,3).to(query.dtype)
        for layer in attn.to_out:result=layer(result)
        return result


def install(pipe,family):
    from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
    from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor
    expected=ZSingleStreamAttnProcessor if family=='zimage' else CosmosAttnProcessor2_0
    for module in pipe.transformer.modules():
        if type(getattr(module,'processor',None)) is expected:module.set_processor(FusedProcessor(module.processor,family=='zimage'))
