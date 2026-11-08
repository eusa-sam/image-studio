"""Phase and resource lifetime shared by serialized image pipelines."""
from contextlib import contextmanager


def enter_phase(pipe,name):
    state=getattr(pipe,'__dict__',{})
    if state.get('_inference_phase')==name:return
    handler=state.get('_phase_handler')
    if handler is not None:handler(name)
    pipe._inference_phase=name


def finish_transfers(pipe):
    transfer=getattr(pipe,'__dict__',{}).get('_weight_transfer')
    if transfer is not None:transfer.close()


def release(pipe):
    try:finish_transfers(pipe)
    finally:
        from .cpu_weights import forget
        forget(pipe)
        state=getattr(pipe,'__dict__',{})
        state.pop('_phase_handler',None)
        for mapping in state.get('weight_maps',()):
            vars(mapping).pop('shared_budget',None)
            mapping.clear_cache()


@contextmanager
def phase(pipe,name):
    previous=getattr(pipe,'__dict__',{}).get('_inference_phase','idle')
    enter_phase(pipe,name)
    try:yield
    finally:enter_phase(pipe,previous)


@contextmanager
def pipeline_phases(pipe):
    import torch
    from .memory_policy import cache_budget_fits
    handles=[];methods=[];budgets=[]
    def before(name):
        def begin(module,args):enter_phase(pipe,name)
        return begin
    try:
        for mapping in getattr(pipe,'weight_maps',()):
            budgets.append((mapping,vars(mapping).get('shared_budget')))
            # FileWeights already checks free RAM/VRAM before allocating each cache entry.
            mapping.shared_budget=lambda **allocation:cache_budget_fits(pipe,**allocation)
        for name in ('text_encoder','text_encoder_2','text_conditioner','transformer','unet'):
            module=getattr(pipe,name,None)
            if isinstance(module,torch.nn.Module):
                handles.append(module.register_forward_pre_hook(before('denoise' if name in ('transformer','unet') else 'condition')))
        vae=getattr(pipe,'vae',None)
        if vae is not None:
            for name in ('encode','decode'):
                original=getattr(vae,name,None)
                if not callable(original):continue
                had_override=name in vars(vae)
                def call(*args,_method=original,_name=name,**kwargs):
                    with phase(pipe,_name):return _method(*args,**kwargs)
                setattr(vae,name,call);methods.append((vae,name,original,had_override))
        with phase(pipe,'condition'):yield
    finally:
        for handle in handles:handle.remove()
        for module,name,original,had_override in reversed(methods):
            if had_override:setattr(module,name,original)
            else:delattr(module,name)
        try:finish_transfers(pipe)
        finally:
            for mapping,previous in budgets:
                if previous is None:vars(mapping).pop('shared_budget',None)
                else:mapping.shared_budget=previous
