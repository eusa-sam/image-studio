"""One compressed weight CPU read ahead, scoped to one serial inference."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from safetensors import safe_open
from backend.app.inference.runtime.memory_policy import available_ram
import torch

@contextmanager
def prefetch_weights(pipe):
    mapping=pipe.weight_maps[0]
    sequence=[name+'.weight' for name,module in pipe.transformer.named_modules()
        if isinstance(module,torch.nn.Linear) and name+'.weight' in mapping.packed_sources
        and ((mapping.packed_sources[name+'.weight'][2] and module.in_features<=16384)
             or name+'.weight' in getattr(mapping,'native_fp8_sources',())) and hasattr(module,'_old_forward')]
    following=dict(zip(sequence,sequence[1:]))
    original=mapping.linear_packed;previous=vars(mapping).get('linear_packed');pending={}
    if not any(name not in mapping.packed_cache and name not in mapping.native_cpu_cache for name in sequence):
        yield
        return
    def read(name):
        if mapping._stamp()!=mapping.stamp:raise ValueError('COMPONENT_CHANGED before prefetch')
        key,_=mapping.sources[name];scale_name=mapping.packed_sources[name][1]
        with safe_open(str(mapping.path),framework='pt',device='cpu') as reader:
            raw,scale=reader.get_tensor(key).clone(),reader.get_tensor(scale_name).clone()
        if mapping._stamp()!=mapping.stamp:raise ValueError('COMPONENT_CHANGED during prefetch')
        return raw,scale
    with ThreadPoolExecutor(max_workers=1) as executor:
        def schedule(name):
            if name is None or pending or name in mapping.packed_cache or name in mapping.native_cpu_cache:return
            elements,_,_,scale_elements=mapping.packed_sources[name];size=elements+4*scale_elements
            if available_ram()<8*1024**3+2*size or mapping.cpu_bytes+size>mapping.cpu_budget or not mapping._allocation_fits(cpu=size):return
            mapping.pending_cpu_bytes=size
            pending[name]=executor.submit(read,name)
        def linear(name,device):
            if pending and name not in pending:
                key,future=pending.popitem();future.result();mapping.pending_cpu_bytes=0
            if name in pending:
                raw,scale=pending.pop(name).result()
                mapping.pending_cpu_bytes=0
                if hasattr(mapping,'_checked_raw'):mapping._checked_raw(name,raw)
                size=raw.numel()*raw.element_size()+scale.numel()*scale.element_size()
                if name not in mapping.native_cpu_cache and name not in mapping.packed_cache and mapping.cpu_bytes+size<=mapping.cpu_budget and available_ram()>=8*1024**3 and mapping._allocation_fits(cpu=size):
                    mapping.native_cpu_cache[name]=(raw,scale);mapping.cpu_bytes+=size
                del raw,scale
            schedule(following.get(name))
            return original(name,device)
        mapping.linear_packed=linear
        try:
            schedule(sequence[0] if sequence else None)
            yield
        finally:
            if previous is None:del mapping.linear_packed
            else:mapping.linear_packed=previous
            for future in pending.values():future.cancel()
            pending.clear()
            mapping.pending_cpu_bytes=0

