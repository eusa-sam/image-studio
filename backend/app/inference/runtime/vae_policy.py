"""Select full decode only inside the calibrated single-image memory envelope."""
from contextlib import contextmanager


def full_decode_fits(vae,latents):
    import torch
    from backend.app.inference.runtime.memory_policy import available_vram
    # ANIMA's full decode cleared the speed gate at 1024 square only.
    if latents.ndim!=5 or latents.shape[0]!=1 or latents.shape[2:]!=(1,128,128):return False
    try:free,_,_=available_vram()
    except RuntimeError:return False
    free+=torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
    missing=sum(p.numel()*p.element_size() for p in vae.parameters() if not p.is_cuda)
    required=5*1024**3+512*1024**2
    return free>=missing+required


@contextmanager
def adaptive_decode(vae):
    original=vae.decode
    had_override='decode' in vars(vae)
    def decode(latents,*args,**kwargs):
        tiled=vae.use_tiling
        try:
            vae.use_tiling=not full_decode_fits(vae,latents)
            return original(latents,*args,**kwargs)
        finally:vae.use_tiling=tiled
    vae.decode=decode
    try:yield
    finally:
        if had_override:vae.decode=original
        else:del vae.decode
