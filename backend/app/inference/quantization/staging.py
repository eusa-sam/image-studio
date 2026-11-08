"""Two bounded transfer slots with copy/consumer event dependencies."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from .quantized_weights import open_weights
from ..runtime.memory_policy import cache_allocation_fits,available_ram


class Staging:
    def __init__(self,pipe,mapping,capacity):
        if not 0<capacity<=128*1024**2:raise ValueError('Invalid staging slot capacity')
        if not cache_allocation_fits(pipe,cpu=capacity*2,gpu=capacity*2):raise MemoryError('Staging budget')
        self.mapping=mapping;self.capacity=capacity;self.closed=False
        self.cpu_tensors=[torch.empty(capacity,dtype=torch.uint8,pin_memory=True) for _ in range(2)]
        self.cuda_tensors=[torch.empty(capacity,dtype=torch.uint8,device='cuda') for _ in range(2)]
        self.stream=torch.cuda.Stream();self.executor=ThreadPoolExecutor(max_workers=1)
        self.slots=[SimpleNamespace(cpu=cpu,gpu=gpu,ready=None,consumed=None,future=None,name=None)
                    for cpu,gpu in zip(self.cpu_tensors,self.cuda_tensors)]
        self.pending={};self.last=None;self.previous=None;self.hits=0;self.fallbacks=0
        self.resident_hits=0;self.synchronous_transfers=0
        self.following=getattr(mapping,'transfer_following',{})
        mapping.transfer_following=self.following

    @torch.inference_mode()
    def prepare(self,slot,name):
        mapping=self.mapping
        if mapping._stamp()!=mapping.stamp:raise ValueError('COMPONENT_CHANGED before staging')
        if slot.ready is not None:slot.ready.synchronize()
        derived=isinstance(name,tuple)
        source=name[1] if derived else name
        group=mapping.packed_sources[source][2]
        slot.validated=derived or name in mapping.native_cpu_cache
        if derived:raw,scale,_=mapping.derived_cpu_cache[name]
        elif slot.validated:raw,scale=mapping.native_cpu_cache[name]
        else:
            with open_weights(mapping.path) as reader:
                raw=reader.get_tensor(mapping.sources[name][0]);scale=reader.get_tensor(mapping.packed_sources[name][1])
        count=raw.numel()*raw.element_size();offset=(count+3)//4*4
        end=offset+scale.numel()*scale.element_size()
        if end>self.capacity:raise ValueError('Staging shape exceeds slot')
        slot.cpu[:count].copy_(raw.contiguous().view(torch.uint8).reshape(-1))
        slot.cpu[offset:end].copy_(scale.reshape(-1).view(torch.uint8))
        with torch.cuda.stream(self.stream):
            if slot.consumed is not None:self.stream.wait_event(slot.consumed)
            slot.gpu[:end].copy_(slot.cpu[:end],non_blocking=True)
            slot.ready=self.stream.record_event()
        if mapping._stamp()!=mapping.stamp:raise ValueError('COMPONENT_CHANGED during staging')
        return slot.gpu[:count].view(raw.dtype).reshape(raw.shape),slot.gpu[offset:end].view(scale.dtype).reshape(scale.shape),group

    def schedule(self,name):
        mapping=self.mapping
        if name is None or name in self.pending or name in mapping.packed_cache:return
        derived=isinstance(name,tuple)
        if derived and (name in mapping.derived_gpu_cache or name not in mapping.derived_cpu_cache):return
        info=mapping.packed_sources.get(name[1] if derived else name)
        if info is None or (info[0]+3)//4*4+4*info[3]>self.capacity:return
        # GPU slots are already allocated; their reuse needs no extra GPU headroom.
        if available_ram()<8*1024**3:return
        slot=next((slot for slot in self.slots if slot.name is None and slot is not self.last),None)
        if slot is None:return
        slot.name=name;slot.future=self.executor.submit(self.prepare,slot,name);self.pending[name]=slot

    def linear(self,name,device,original):
        if self.closed:return original(name,device)
        if self.last is not None:
            self.last.consumed=torch.cuda.current_stream().record_event()
            self.last.name=None;self.last=None
        if self.previous is not None:self.following[self.previous]=name
        self.previous=name
        slot=self.pending.pop(name,None)
        if slot is None:
            if name in self.mapping.packed_cache or name in getattr(self.mapping,'derived_gpu_cache',{}):self.resident_hits+=1
            else:self.synchronous_transfers+=1
            self.fallbacks+=1;result=original(name,device)
        else:
            result=slot.future.result()
            torch.cuda.current_stream().wait_event(slot.ready)
            if not slot.validated:self.mapping._checked_raw(name,result[0])
            self.last=slot;self.hits+=1
        following=self.following.get(name)
        self.schedule(following);self.schedule(self.following.get(following))
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            for slot in self.slots:
                if slot.future is not None:slot.future.result()
        finally:
            self.executor.shutdown(wait=True,cancel_futures=True)
            self.stream.synchronize();torch.cuda.current_stream().synchronize()
            self.pending.clear();self.last=None;self.slots.clear()
            self.cpu_tensors.clear();self.cuda_tensors.clear()


@contextmanager
def staging_weights(pipe):
    mapping=pipe.weight_maps[0]
    pipe.memory_policy.update(weight_prefetch='synchronous',tensor_transfer='blocking')
    pipe._staging_statistics=None
    sizes=[(info[0]+3)//4*4+info[3]*4 for info in mapping.packed_sources.values()]
    capacity=max((size for size in sizes if size<=128*1024**2),default=0)
    if not capacity:yield;return
    # The pinned allocator owns power-of-two blocks; charge their actual capacity.
    capacity=1<<(capacity-1).bit_length()
    try:transfer=Staging(pipe,mapping,capacity)
    except (MemoryError,RuntimeError):yield;return
    transfer.pipe=pipe;pipe._weight_transfer=transfer
    pipe.memory_policy.update(weight_prefetch='bounded_two_slot_staging',tensor_transfer='event_ordered_async_with_sync_fallback')
    gpu_budget,cpu_budget=mapping.budget,mapping.cpu_budget
    mapping.budget=max(0,gpu_budget-2*capacity)
    mapping.limit_cpu_cache(max(0,cpu_budget-2*capacity))
    previous=vars(mapping).get('linear_packed');original=mapping.linear_packed
    budgets=[vars(item).get('shared_budget') for item in pipe.weight_maps]
    for item in pipe.weight_maps:item.shared_budget=lambda **allocation:cache_allocation_fits(pipe,**allocation)
    mapping.linear_packed=lambda name,device:transfer.linear(name,device,original)
    try:yield
    finally:
        try:transfer.close()
        finally:
            pipe._staging_statistics=dict(hits=transfer.hits,fallbacks=transfer.fallbacks,slot_capacity=capacity,
                                          resident_hits=transfer.resident_hits,synchronous_transfers=transfer.synchronous_transfers)
            if previous is None:del mapping.linear_packed
            else:mapping.linear_packed=previous
            for item,budget in zip(pipe.weight_maps,budgets):
                if budget is None:del item.shared_budget
                else:item.shared_budget=budget
            mapping.budget=gpu_budget;mapping.cpu_budget=cpu_budget
            del pipe._weight_transfer
