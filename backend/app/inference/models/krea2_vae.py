"""Batch the VAE's independent blend rows while preserving each rounded operation."""
from contextlib import contextmanager
import torch


def blend(a,b,extent,dim,fallback):
    extent=min(a.shape[dim],b.shape[dim],extent)
    # These dyadic coefficients are exact even in BF16; other overlaps keep Diffusers' scalar path.
    if not (0<extent<=256 and extent&(extent-1)==0 and a.dtype==b.dtype
            and b.dtype in (torch.bfloat16,torch.float16,torch.float32)):
        return fallback(a,b,extent)
    shape=[1]*b.ndim;shape[dim]=extent
    weight=(torch.arange(extent,device=b.device,dtype=b.dtype)/extent).reshape(shape)
    source=[slice(None)]*a.ndim;source[dim]=slice(-extent,None)
    target=[slice(None)]*b.ndim;target[dim]=slice(0,extent)
    b[tuple(target)]=a[tuple(source)]*(1-weight)+b[tuple(target)]*weight
    return b


@contextmanager
def vectorized_blending(vae):
    saved={name:vars(vae).get(name) for name in ('blend_v','blend_h')}
    for name,dim in (('blend_v',-2),('blend_h',-1)):
        original=getattr(vae,name)
        setattr(vae,name,lambda a,b,extent,_dim=dim,_old=original:blend(a,b,extent,_dim,_old))
    try:yield
    finally:
        for name,previous in saved.items():
            if previous is None:delattr(vae,name)
            else:setattr(vae,name,previous)
