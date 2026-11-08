"""Reuse CPU storage for immutable inference weights when evicting GPU copies."""
def remember(pipe):
    import torch
    from backend.app.inference.runtime.memory_policy import available_ram
    if getattr(pipe,'_cpu_weights',None):return
    entries=[];storages={}
    budget=min(16*1024**3,max(0,available_ram()-8*1024**3))
    size=0
    for component in (pipe.transformer,pipe.vae,getattr(pipe,'text_encoder',None)):
        if component is None:continue
        for module in component.modules():
            for kind in ('_parameters','_buffers'):
                for name,value in getattr(module,kind).items():
                    if value is None:continue
                    # Quantized tensor subclasses carry placement metadata of their own.
                    if type(value) not in (torch.Tensor,torch.nn.Parameter) or value.device.type!='cpu':return
                    storage=value.untyped_storage();pointer=storage.data_ptr()
                    extra=0 if pointer in storages else storage.nbytes()
                    if size+extra>budget:continue
                    entries.append((module,kind,name,value.detach()))
                    storages[pointer]=storage.nbytes();size+=extra
    pipe._cpu_weights=entries
    pipe._cpu_weight_bytes=size


def restore(pipe):
    import torch
    entries=getattr(pipe,'_cpu_weights',())
    for module,kind,name,backup in entries:
        value=getattr(module,kind).get(name)
        if value is None or value.shape!=backup.shape or value.dtype!=backup.dtype:
            forget(pipe)
            return False
    with torch.no_grad():
        for module,kind,name,backup in entries:
            if kind=='_parameters':module._parameters[name].data=backup
            else:module._buffers[name]=backup
    return bool(entries)


def forget(pipe):
    if hasattr(pipe,'_cpu_weights'):del pipe._cpu_weights
    if hasattr(pipe,'_cpu_weight_bytes'):del pipe._cpu_weight_bytes


def backed_tensors(pipe):
    backed=set()
    for module,kind,name,backup in getattr(pipe,'_cpu_weights',()):
        value=getattr(module,kind).get(name)
        if value is None or value.shape!=backup.shape or value.dtype!=backup.dtype:
            forget(pipe)
            return set()
        backed.add(id(value))
    return backed
