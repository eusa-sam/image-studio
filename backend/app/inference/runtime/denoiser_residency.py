"""Keep the denoiser resident after cached conditioning, retaining the CPU encoder."""
def cpu_transfer_fits(pipe):
    from backend.app.inference.runtime.memory_policy import available_ram
    from backend.app.inference.runtime.cpu_weights import backed_tensors
    backed=backed_tensors(pipe)
    moving=sum(p.numel()*p.element_size() for module in (pipe.transformer,pipe.vae,pipe.text_encoder) if module is not None
               for p in list(module.parameters())+list(module.buffers()) if p.is_cuda and id(p) not in backed)
    return available_ram()>=8*1024**3+moving


def retain_denoiser(pipe,width,height):
    import torch
    from backend.app.inference.runtime.memory_policy import configure_pipeline, available_vram
    policy=pipe.memory_policy
    if policy['mode']!='model_offload' or pipe.text_encoder is None:return
    modules=[pipe.transformer,pipe.vae]
    size=sum(p.numel()*p.element_size() for module in modules for p in list(module.parameters())+list(module.buffers()))
    if size>policy['resident_budget_bytes'] or not cpu_transfer_fits(pipe):return
    from backend.app.inference.runtime.cpu_weights import restore
    pipe.remove_all_hooks();restore(pipe);pipe.to('cpu')
    encoder=pipe.text_encoder
    pipe.text_encoder=None
    try:
        placement=configure_pipeline(pipe,width,height,activation_reserve_bytes=policy['activation_reserve_bytes'])
    finally:
        pipe.text_encoder=encoder
    if placement['mode']=='full':
        placement['mode']='denoiser_resident'
        placement['idle_encoder_bytes']=sum(p.numel()*p.element_size() for p in list(encoder.parameters())+list(encoder.buffers()))
    else:
        # Capacity can change during placement; restore the complete offload chain.
        pipe.remove_all_hooks();restore(pipe);pipe.to('cpu')
        placement=configure_pipeline(pipe,width,height,activation_reserve_bytes=policy['activation_reserve_bytes'])
    policy.update(placement)
    policy['free_vram_with_allocator_bytes']=available_vram()[0]+torch.cuda.memory_reserved()


def prepare_encoder(pipe,width,height,resume):
    if pipe.memory_policy['mode']!='denoiser_resident':return
    if not cpu_transfer_fits(pipe):raise RuntimeError('문장을 다시 인코딩할 여유 RAM이 부족합니다. 모델 메모리를 해제한 뒤 다시 시도하세요.')
    from backend.app.inference.runtime.cpu_weights import restore, forget
    from backend.app.inference.runtime.memory_policy import available_ram
    pipe.remove_all_hooks();restore(pipe);pipe.to('cpu')
    if available_ram()<8*1024**3:forget(pipe)
    policy=pipe.memory_policy
    reused=policy.get('pipeline_reused')
    resume(pipe,width,height)
    policy.clear();policy.update(pipe.memory_policy);pipe.memory_policy=policy
    if reused is not None:policy['pipeline_reused']=reused
