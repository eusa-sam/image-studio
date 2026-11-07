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


def needs_vram_rebalance(policy, free_bytes):
    previous=policy.setdefault('free_vram_with_allocator_bytes',policy['free_vram_bytes'])
    change=free_bytes-previous
    # Reclaim a useful amount at once; respond to pressure at the existing threshold.
    return change < -256*1024**2 or (policy.get('mode')!='full' and change > 1024**3)


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
