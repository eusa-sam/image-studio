"""Choose residency from available VRAM, keeping space for activations and other apps."""
def int8_preload_budget(path, free_ram):
    from safetensors import safe_open
    with safe_open(str(path),framework='pt',device='cpu') as reader:
        has_int8=any(key.endswith('.weight') and reader.get_slice(key).get_dtype()=='I8' for key in reader.keys())
    return min(8*1024**3,max(0,free_ram-16*1024**3)) if has_int8 else 0


def available_ram():
    import os
    if os.name=='nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):return 0
        return status.available
    try:return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    except (ValueError,OSError):return 0


def driver_memory():
    import subprocess
    result=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.free,memory.total',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,
                          timeout=5,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    free,total=(int(value.strip())*1024**2 for value in result.stdout.strip().split(','))
    if total<=0 or not 0<=free<=total:raise ValueError('Invalid driver memory')
    return free,total


def available_vram():
    import os
    import subprocess
    import torch
    free,total=torch.cuda.mem_get_info()
    snapshot=dict(cuda_free_bytes=free,cuda_total_bytes=total)
    try:
        driver_free,driver_total=driver_memory()
        snapshot.update(driver_free_bytes=driver_free,driver_total_bytes=driver_total)
        free=min(free,driver_free)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        if os.name=='nt':
            raise RuntimeError('NVIDIA GPU 메모리 조회에 실패했습니다. 다른 앱의 점유량을 확인할 수 없어 자동 적재를 중단합니다.') from error
        snapshot['driver_query']='unavailable'
    return free,total,snapshot


def residency_plan(free_bytes, total_bytes, weight_bytes, width, height, free_ram=0, activation_reserve_bytes=None):
    gib = 1024**3
    reserve = max(4*gib, int(4*gib*width*height/(1024*1024)))
    if activation_reserve_bytes is not None:
        if type(activation_reserve_bytes) is not int or activation_reserve_bytes<=0:raise ValueError('Invalid activation reserve')
        reserve=activation_reserve_bytes
    budget = max(0, free_bytes-reserve)
    return dict(mode='full' if weight_bytes <= budget else 'partial',
                free_vram_bytes=free_bytes, total_vram_bytes=total_bytes,
                weight_bytes=weight_bytes, activation_reserve_bytes=reserve,
                resident_budget_bytes=budget,free_ram_bytes=free_ram,
                cpu_cache_budget_bytes=min(16*gib,max(0,free_ram-8*gib)))


def cached_storage_bytes(pipe,device):
    """Count storage once across original, derived, and transfer caches."""
    import torch
    state=getattr(pipe,'__dict__',{});storages={}
    def collect(value):
        if isinstance(value,torch.Tensor) and value.device.type==device:
            storage=value.untyped_storage()
            storages[(value.device.index,storage.data_ptr())]=storage.nbytes()
        elif isinstance(value,dict):
            for item in value.values():collect(item)
        elif isinstance(value,(list,tuple)):
            for item in value:collect(item)
    for entry in state.get('_cpu_weights',()):collect(entry[-1])
    for mapping in state.get('weight_maps',()):
        for name in ('cache','native_cpu_cache','packed_cache','extra','extra_gpu','derived_cpu_cache','derived_gpu_cache'):
            collect(getattr(mapping,name,{}))
    transfer=state.get('_weight_transfer')
    if transfer is not None:collect(getattr(transfer,device+'_tensors',()))
    pending=sum(getattr(mapping,'pending_cpu_bytes',0) for mapping in state.get('weight_maps',())) if device=='cpu' else 0
    return sum(storages.values())+pending


def cache_budget_fits(pipe,*,cpu=0,gpu=0,temporary=0):
    if min(cpu,gpu,temporary)<0:raise ValueError('Negative cache allocation')
    policy=pipe.memory_policy
    if cpu and retained_bytes(pipe,'cpu')+cpu>policy['cpu_cache_budget_bytes']:return False
    if (gpu or temporary) and retained_bytes(pipe,'cuda')+gpu+temporary>policy['resident_budget_bytes']:return False
    return True


def cache_allocation_fits(pipe,*,cpu=0,gpu=0,temporary=0):
    import torch
    if not cache_budget_fits(pipe,cpu=cpu,gpu=gpu,temporary=temporary):return False
    if cpu and available_ram()<8*1024**3+cpu:return False
    if gpu or temporary:
        free=torch.cuda.mem_get_info()[0]+max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
        if free<pipe.memory_policy['activation_reserve_bytes']+gpu+temporary:return False
    return True


def retained_bytes(pipe,device):
    maps=getattr(pipe,'weight_maps',())
    if not maps or any(not hasattr(mapping,'extra_cpu_bytes') for mapping in maps):return cached_storage_bytes(pipe,device)
    if device=='cpu':
        size=sum(mapping.cpu_bytes+mapping.extra_cpu_bytes+getattr(mapping,'derived_cpu_bytes',0)+getattr(mapping,'pending_cpu_bytes',0) for mapping in maps)
        size+=getattr(pipe,'_cpu_weight_bytes',0)
    else:size=sum(mapping.cache_bytes+getattr(mapping,'derived_gpu_bytes',0) for mapping in maps)
    transfer=getattr(pipe,'__dict__',{}).get('_weight_transfer')
    if transfer is not None:size+=sum(v.numel()*v.element_size() for v in getattr(transfer,device+'_tensors',()))
    return size


def reclaim_gpu(pipe,*,temporary):
    """Release file-backed/CPU-backed caches only when temporary work needs their space."""
    if cache_allocation_fits(pipe,temporary=temporary):return True
    for mapping in reversed(pipe.weight_maps):
        for field in ('derived_gpu_cache','packed_cache','extra_gpu','cache'):
            values=getattr(mapping,field,{})
            for key in reversed(list(values)):
                value=values[key]
                if field=='cache' and not value.is_cuda:continue
                tensors=value[:2] if isinstance(value,tuple) else (value,)
                size=sum(tensor.numel()*tensor.element_size() for tensor in tensors)
                del values[key]
                if field!='derived_gpu_cache':mapping.cache_bytes-=size
                else:mapping.derived_gpu_bytes-=size
                mapping.reclaimed_gpu_bytes=getattr(mapping,'reclaimed_gpu_bytes',0)+size
                del value,tensors
                if cache_allocation_fits(pipe,temporary=temporary):return True
    return False


def needs_vram_rebalance(policy, free_bytes):
    previous=policy.setdefault('free_vram_with_allocator_bytes',policy['free_vram_bytes'])
    change=free_bytes-previous
    # Initialization can consume spare capacity without invalidating the planned reserve.
    required=policy.get('resident_budget_bytes',previous)+policy.get('activation_reserve_bytes',0)
    if policy.get('mode') in ('full','denoiser_resident') and 'weight_bytes' in policy:
        required=policy['weight_bytes']+policy['activation_reserve_bytes']
    return (change < -256*1024**2 and free_bytes < required) or (policy.get('mode') not in ('full','denoiser_resident') and change > 1024**3)


def fresh_scheduler(scheduler):
    # Ignore constructor-default metadata; preserve the effective sampling settings.
    return type(scheduler).from_config({key:value for key,value in scheduler.config.items() if not key.startswith('_')})


def convert_pipeline(pipe, pipeline_class):
    # Diffusers from_pipe defaults to FP32, mutating the shared GPU components.
    scheduler=fresh_scheduler(pipe.scheduler)
    return pipeline_class.from_pipe(pipe, dtype=pipe.unet.dtype, scheduler=scheduler)


def configure_pipeline(pipe,width,height,*,activation_reserve_bytes=None):
    import torch
    modules=[value for value in pipe.components.values() if isinstance(value,torch.nn.Module)]
    sizes=[sum(value.numel()*value.element_size() for value in list(model.parameters())+list(model.buffers())) for model in modules]
    # Previous postprocessing may leave free allocator blocks that hide usable VRAM.
    torch.cuda.empty_cache()
    free,total,measurement=available_vram()
    policy=residency_plan(free,total,sum(sizes),width,height,available_ram(),activation_reserve_bytes)
    policy['measurement']=measurement
    if policy['mode']=='full':pipe.to('cuda')
    elif max(sizes,default=0)<=policy['resident_budget_bytes']:
        policy['mode']='model_offload'
        pipe.enable_model_cpu_offload()
    else:
        policy['mode']='sequential_offload'
        pipe.enable_sequential_cpu_offload()
    return policy


def weight_validation_device(free_vram, largest_tensor_elements, has_int8, *, allow_int8=False):
    # Allow input, FP32 intermediates, finite masks and output without filling VRAM.
    required=largest_tensor_elements*20+2*1024**3
    return 'cuda:0' if (not has_int8 or allow_int8) and free_vram>=required else 'cpu'
