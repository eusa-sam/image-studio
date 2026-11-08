"""Memory-only, layer-at-a-time quantized LoRA; original files stay immutable."""
import torch

from .quantized_weights import open_weights, undo_convrot


def merge_packed(raw,scale,group,a,b,strength):
    base=undo_convrot(raw.float(),group)*scale if group else raw.float()*scale.reshape(())
    merged=base.addmm(b.float(),a.float(),alpha=float(strength))
    if not torch.isfinite(merged).all():raise ValueError('Nonfinite merged LoRA weight')
    if group:
        rotated=undo_convrot(merged,group)
        new_scale=(rotated.abs().amax(dim=1,keepdim=True)/127).clamp_min(1e-12)
        weight=(rotated/new_scale).round().clamp(-127,127).to(torch.int8)
    else:
        new_scale=(merged.abs().amax()/448).clamp_min(1e-12)
        weight=(merged/new_scale).clamp(-448,448).to(torch.float8_e4m3fn)
    return weight,new_scale,group


def invalidate(mapping):
    mapping.derived_cpu_cache={};mapping.derived_gpu_cache={};mapping.derived_identity=None
    mapping.derived_cpu_bytes=mapping.derived_gpu_bytes=0
    mapping.derived_failed=set()
    getattr(mapping,'transfer_following',{}).clear()


def configure(pipe,model_hash,selections):
    """Install only for one ordinary adapter; every unvalidated combination falls back."""
    from peft.tuners.lora.layer import Linear
    from backend.app.inference.runtime.memory_policy import cache_allocation_fits
    mapping=pipe.weight_maps[0]
    identity=(model_hash,tuple((row['profile']['sha256'],row['weight']) for row,_ in selections),1)
    if getattr(mapping,'derived_identity',None)!=identity:
        invalidate(mapping);mapping.derived_identity=identity
    mapping.derived_failed=set()
    mapping.derived_request_calls=mapping.derived_request_fallback_calls=0
    mapping.lora_layers=mapping.derived_layers=0
    for item in pipe.weight_maps:item.shared_budget=lambda **allocation:cache_allocation_fits(pipe,**allocation)
    eligible=len(selections)==1 and selections[0][0]['weight']!=0
    for name,module in pipe.transformer.named_modules():
        if not isinstance(module,Linear):continue
        mapping.lora_layers+=1
        attribute='_old_forward' if hasattr(module,'_old_forward') else 'forward'
        if hasattr(module,'_derived_original'):
            setattr(module,attribute,module._derived_original)
            del module._derived_original
        key=name+'.base_layer.weight'
        if (not eligible or key not in mapping.packed_sources or key in mapping.full_precision_sources
                or mapping.sources[key][1]!=torch.bfloat16):continue
        mapping.derived_layers+=1
        module._derived_original=getattr(module,attribute)
        def forward(x,*args,_module=module,_name=name,_key=key,**kwargs):
            old=_module._derived_original;active=_module.active_adapters
            if (args or kwargs or torch.is_grad_enabled() or _module.disable_adapters or _module.merged or
                    len(active)!=1 or active[0] not in _module.lora_A or active[0] in _module.lora_variant or
                    not x.is_cuda or x.dtype!=torch.bfloat16):return old(x,*args,**kwargs)
            if _module.training and getattr(_module.lora_dropout[active[0]],'p',0):return old(x,*args,**kwargs)
            mapping.derived_request_calls+=1
            packed=get_weight(pipe,mapping,_key,_name,active[0],_module.scaling[active[0]],x.device)
            if packed is None:
                mapping.derived_request_fallback_calls+=1
                return old(x,*args,**kwargs)
            if packed[2]:
                from backend.app.inference.ops.native_int8 import native_linear
                output=native_linear(x,*packed)
            else:
                from backend.app.inference.ops.native_fp8 import native_linear
                output=native_linear(x,*packed[:2])
            if output is None:
                mapping.derived_request_fallback_calls+=1
                return old(x,*args,**kwargs)
            mapping.derived_hits=getattr(mapping,'derived_hits',0)+1
            bias=_name+'.base_layer.bias'
            return output if _module.base_layer.bias is None else output+mapping[bias].to(x.device)
        setattr(module,attribute,forward)


def get_weight(pipe,mapping,key,module,adapter,strength,device):
    from backend.app.inference.runtime.memory_policy import cache_allocation_fits,reclaim_gpu
    if mapping._stamp()!=mapping.stamp:raise ValueError('COMPONENT_CHANGED during LoRA merge')
    cache_key=(mapping.derived_identity,key,mapping.packed_sources[key][2])
    transfer=getattr(pipe,'__dict__',{}).get('_weight_transfer')
    if transfer is not None and cache_key in mapping.derived_cpu_cache:
        return transfer.linear(cache_key,device,lambda name,target:upload(pipe,mapping,name,target))
    if cache_key in mapping.derived_gpu_cache:return mapping.derived_gpu_cache[cache_key]
    if cache_key in mapping.derived_failed:return None
    elements,scale_name,group,scale_elements=mapping.packed_sources[key]
    size=elements+4*scale_elements
    if cache_key not in mapping.derived_cpu_cache:
        # Retain only one representation of the base layer in the bounded CPU cache.
        previous=mapping.native_cpu_cache.pop(key,None)
        if previous is not None:mapping.cpu_bytes-=sum(v.numel()*v.element_size() for v in previous)
        previous=mapping.cache.get(key)
        if previous is not None and previous.device.type=='cpu':
            del mapping.cache[key];mapping.cpu_bytes-=previous.numel()*previous.element_size()
        del previous
        if not cache_allocation_fits(pipe,cpu=size) or not reclaim_gpu(pipe,temporary=elements*20):
            mapping.derived_failed.add(cache_key)
            mapping.derived_fallbacks=getattr(mapping,'derived_fallbacks',0)+1
            return None
        a=mapping.extra.get(module+'.lora_A.'+adapter+'.weight')
        b=mapping.extra.get(module+'.lora_B.'+adapter+'.weight')
        if a is None or b is None:return None
        try:
            with open_weights(mapping.path) as reader:
                raw=reader.get_tensor(mapping.sources[key][0]).to(device)
                scale=reader.get_tensor(scale_name).to(device)
            mapping._checked_raw(key,raw)
            try:
                packed=merge_packed(raw,scale,group,a.to(device),b.to(device),strength)
            except (RuntimeError,ValueError):
                mapping.derived_failed.add(cache_key)
                mapping.derived_fallbacks=getattr(mapping,'derived_fallbacks',0)+1
                return None
            if mapping._stamp()!=mapping.stamp:raise ValueError('COMPONENT_CHANGED during LoRA merge')
            mapping.derived_cpu_cache[cache_key]=(packed[0].cpu(),packed[1].cpu(),group)
            mapping.derived_cpu_bytes+=size
            mapping.derived_preparations=getattr(mapping,'derived_preparations',0)+1
        except torch.cuda.OutOfMemoryError:
            mapping.derived_failed.add(cache_key)
            mapping.derived_fallbacks=getattr(mapping,'derived_fallbacks',0)+1
            return None
        finally:
            # Temporary restored weights never survive a single layer's preparation.
            raw=scale=None
    else:
        packed=mapping.derived_cpu_cache[cache_key]
        packed=(packed[0].to(device),packed[1].to(device),group)
    previous=mapping.packed_cache.pop(key,None)
    if previous is not None:mapping.cache_bytes-=sum(v.numel()*v.element_size() for v in previous[:2])
    previous=mapping.cache.pop(key,None)
    if previous is not None:
        if previous.is_cuda:mapping.cache_bytes-=previous.numel()*previous.element_size()
        else:mapping.cpu_bytes-=previous.numel()*previous.element_size()
    if mapping.cache_bytes+mapping.derived_gpu_bytes+size<=mapping.budget and cache_allocation_fits(pipe,gpu=size):
        mapping.derived_gpu_cache[cache_key]=packed;mapping.derived_gpu_bytes+=size
    return packed


def upload(pipe,mapping,cache_key,device):
    from backend.app.inference.runtime.memory_policy import cache_allocation_fits
    if cache_key in mapping.derived_gpu_cache:return mapping.derived_gpu_cache[cache_key]
    raw,scale,group=mapping.derived_cpu_cache[cache_key]
    size=raw.numel()*raw.element_size()+scale.numel()*scale.element_size()
    packed=(raw.to(device),scale.to(device),group)
    if mapping.cache_bytes+mapping.derived_gpu_bytes+size<=mapping.budget and cache_allocation_fits(pipe,gpu=size):
        mapping.derived_gpu_cache[cache_key]=packed;mapping.derived_gpu_bytes+=size
    return packed
