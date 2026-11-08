"""Remove masked text lanes after encoding, retaining the valid assistant suffix."""
from contextlib import contextmanager
import torch


def compact(embeddings,mask):
    if (embeddings.ndim!=4 or embeddings.shape[0]!=1 or mask.dtype!=torch.bool
            or mask.shape!=embeddings.shape[:2] or embeddings.device!=mask.device):return embeddings,mask,False
    indices=mask[0].nonzero().flatten()
    if not indices.numel():return embeddings,mask,False
    if indices.numel()==mask.shape[1]:return embeddings,mask,True
    return embeddings.index_select(1,indices),torch.ones((1,indices.numel()),device=mask.device,dtype=torch.bool),True


@contextmanager
def compact_tokens(pipe):
    original=pipe.encode_prompt;had_override='encode_prompt' in vars(pipe);masks=[]
    pipe._compacted_text_tokens=[]
    def encode(*args,**kwargs):
        embeddings,mask=original(*args,**kwargs)
        if torch.is_grad_enabled() or pipe.do_classifier_free_guidance:return embeddings,mask
        previous=embeddings.shape[1]
        embeddings,mask,eligible=compact(embeddings,mask)
        if eligible:
            masks.append(mask)
            pipe._compacted_text_tokens.append(dict(before=previous,after=embeddings.shape[1]))
        return embeddings,mask
    def before(module,args,kwargs):
        # Only this encode call's proven all-valid mask can be omitted.
        if any(kwargs.get('encoder_attention_mask') is mask for mask in masks):
            kwargs=dict(kwargs);kwargs['encoder_attention_mask']=None
        return args,kwargs
    pipe.encode_prompt=encode
    handle=pipe.transformer.register_forward_pre_hook(before,with_kwargs=True)
    try:yield
    finally:
        handle.remove();masks.clear()
        if had_override:pipe.encode_prompt=original
        else:del pipe.encode_prompt
