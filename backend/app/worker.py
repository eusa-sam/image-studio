"""Offline inference in an isolated worker, with optional validated pipeline reuse."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import hashlib
import json
import sys
import time
import threading
import faulthandler
import gc
from pathlib import Path

from .models import MODEL_ROOT, fingerprint
from .resource_usage import record_resources


def generation_resolution(settings, resolved):
    detail=settings['detail_resolution'] if any(key in resolved for key in ('face_detector','hand_detector')) else 0
    return max(settings['width'],detail),max(settings['height'],detail)


def component_path(component,cache=None):
    path = (MODEL_ROOT / component['path']).resolve()
    if not path.is_relative_to(MODEL_ROOT.resolve()) or fingerprint(path) != component['fingerprint']:
        raise ValueError('COMPONENT_CHANGED')
    if cache is not None:checksum=cache.file_hash(path)
    else:
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checksum != component['profile']['sha256'] or fingerprint(path) != component['fingerprint']:
        raise ValueError('COMPONENT_HASH_MISMATCH')
    return path


def verify_components(plan,cache=None):
    resolved={kind:component_path(row,cache) for kind,row in plan.get('components',{}).items()}
    loras=[(row,component_path(row,cache)) for row in plan.get('loras',[])]
    return resolved,loras


def verify_inputs(plan,cache=None,verified_components=None):
    from concurrent.futures import ThreadPoolExecutor
    path=(MODEL_ROOT/plan['model_path']).resolve()
    if not path.is_relative_to(MODEL_ROOT.resolve()) or fingerprint(path)!=plan['fingerprint']:
        raise ValueError('MODEL_CHANGED')
    # Each component is used only after its own validation completes.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending=verified_components if verified_components is not None else executor.submit(verify_components,plan,cache)
        print('Hashing checkpoint',flush=True)
        if path.is_dir():
            if plan['runner'].startswith(('flux2_','zimage_')):
                if plan['runner'].startswith('zimage_'):from backend.app.inference.models.z_image import verify_bundle
                else:from backend.app.inference.models.flux2 import verify_bundle
                verify_bundle(path,cache=cache)
            else:
                from backend.app.inference.models.anima import verify_bundle
                verify_bundle(path,cache=cache)
            model_hash=hashlib.sha256((path/'download_manifest.json').read_bytes()).hexdigest()
        elif cache is not None:model_hash=cache.file_hash(path)
        else:
            with path.open('rb') as stream:model_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        resolved,loras=pending.result()
    return path,model_hash,resolved,loras


def upscale(image, path, scale, progress=None):
    import torch
    from spandrel import ModelLoader, ImageModelDescriptor
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(str(path), device='cpu')
    else:
        weights = torch.load(path, map_location='cpu', weights_only=True)
    model = ModelLoader().load_from_state_dict(weights)
    if not isinstance(model, ImageModelDescriptor) or model.input_channels != 3 or model.output_channels != 3 or model.scale != scale:
        raise ValueError('업스케일 모델의 채널/배율이 프로필과 맞지 않습니다.')
    model = model.eval().cuda()
    return tiled_upscale(image, model, scale, progress=progress)


def tiled_upscale(image, model, scale, tile_size=256, halo=32, progress=None):
    """Keep context around each tile; assemble only its center on the CPU."""
    import torch
    import numpy as np
    from PIL import Image
    from spandrel import ModelTiling
    if tile_size < 1 or halo < 0 or scale not in (1,2,3,4):
        raise ValueError('잘못된 타일/배율 설정')
    width, height = image.size
    if max(width,height) > tile_size and model.tiling != ModelTiling.SUPPORTED:
        raise ValueError('이 업스케일 모델은 타일 실행을 권장하지 않습니다.')
    source = np.asarray(image.convert('RGB'))
    output = np.empty((height*scale,width*scale,3),dtype=np.uint8)
    total = ((width+tile_size-1)//tile_size)*((height+tile_size-1)//tile_size)
    index = 0
    with torch.inference_mode():
        for y in range(0,height,tile_size):
            for x in range(0,width,tile_size):
                right, bottom = min(x+tile_size,width), min(y+tile_size,height)
                left0, top0 = max(0,x-halo), max(0,y-halo)
                right0, bottom0 = min(width,right+halo), min(height,bottom+halo)
                tensor = torch.from_numpy(source[top0:bottom0,left0:right0].copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
                result = model(tensor)
                if not torch.isfinite(result).all() or result.shape != (1,3,(bottom0-top0)*scale,(right0-left0)*scale):
                    raise ValueError('업스케일 결과 크기 또는 픽셀 오류')
                pixels = result[0,:, (y-top0)*scale:(bottom-top0)*scale, (x-left0)*scale:(right-left0)*scale]
                output[y*scale:bottom*scale,x*scale:right*scale] = (pixels.permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype('uint8')
                del tensor, result, pixels
                index += 1
                print(json.dumps(dict(stage='upscale',tile=index,total=total)),flush=True)
                if progress:progress.update('upscale',step=index,total=total)
    image_out = Image.fromarray(output)
    if 'A' in image.getbands() or 'transparency' in image.info:
        image_out.putalpha(image.convert('RGBA').getchannel('A').resize(image_out.size,Image.Resampling.LANCZOS))
    return image_out


def parent_watch():
    if os.name == 'nt':
        # Blocking pipe reads before NumPy import hang DLL initialization on this host.
        import ctypes
        import msvcrt
        from ctypes import wintypes
        peek = ctypes.WinDLL('kernel32', use_last_error=True).PeekNamedPipe
        peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                         wintypes.LPVOID, wintypes.LPVOID, wintypes.LPVOID]
        peek.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        while peek(handle, None, 0, None, None, None):
            time.sleep(0.25)
    else:
        while os.read(sys.stdin.fileno(), 1):
            pass
    os._exit(20)


def pipeline_output(pipe,*,on_step=None,**arguments):
    from contextlib import nullcontext
    from backend.app.inference.quantization.quantized_weights import weight_readers
    from backend.app.inference.runtime.vae_policy import adaptive_decode
    from backend.app.inference.runtime.execution import pipeline_phases
    if getattr(pipe,'memory_policy',{}).get('family')=='anima' and arguments.get('image') is not None:
        arguments.setdefault('width',arguments['image'].width)
        arguments.setdefault('height',arguments['image'].height)
    decode=adaptive_decode(pipe.vae) if getattr(pipe,'memory_policy',{}).get('family')=='anima' else nullcontext()
    with weight_readers(),decode,pipeline_phases(pipe):
        if getattr(pipe,'memory_policy',{}).get('int8_compute')=='native_convrot_with_bf16_fallback':
            from backend.app.inference.quantization.weight_prefetch import prefetch_weights
            from backend.app.inference.models.krea2_vae import vectorized_blending
            from backend.app.inference.conditioning.krea2_tokens import compact_tokens
            packed=pipe.weight_maps[0].packed_sources
            transfer=prefetch_weights
            pipe.memory_policy['weight_prefetch']='bounded_cpu_read_ahead'
            if any(info[2] for info in packed.values()):
                from backend.app.inference.quantization.staging import staging_weights
                transfer=staging_weights
            tokens=compact_tokens(pipe) if packed else nullcontext()
            with tokens,transfer(pipe),vectorized_blending(pipe.vae):return pipe(**arguments)
        if on_step is not None and getattr(pipe,'memory_policy',{}).get('family')=='anima':
            from backend.app.inference.models.anima import progress_reporting
            with progress_reporting(pipe,on_step):return pipe(**arguments)
        return pipe(**arguments)


@record_resources
def run(request_path,cache=None):
    global MODEL_ROOT
    if cache is not None:cache.hash_checks.clear()
    request=json.loads(Path(request_path).read_text(encoding='utf-8'))
    if 'model_root' in request:
        root=Path(request['model_root'])
        if not root.is_absolute() or not root.is_dir():raise ValueError('Invalid configured model folder')
        if root.resolve()!=MODEL_ROOT.resolve() and cache is not None:cache.clear()
        MODEL_ROOT=root.resolve()
        from backend.app import models; from backend.app.inference.conditioning import prompt_tokens
        models.MODEL_ROOT=prompt_tokens.MODEL_ROOT=MODEL_ROOT
        prompt_tokens.tokenizers.cache_clear()
    if request['plan']['runner'].startswith('krea2_'):
        from backend.app.inference.runtime.offload import reuse_allocator, prefer_cudnn
        from backend.app.inference.models.krea2_rotary import fused_rotary
        with reuse_allocator(),prefer_cudnn(),fused_rotary():
            try:
                return _run(request_path,cache) if cache is not None else _run(request_path)
            except BaseException:
                if cache is not None:cache.clear()
                raise
    if request['plan']['runner'].startswith(('flux2_','zimage_','sdxl_','sd1_','sd2_','anima_')):
        from contextlib import nullcontext
        from backend.app.inference.runtime.offload import reuse_allocator, prefer_cudnn
        attention=nullcontext() if request['plan']['runner'].startswith('sdxl_') else prefer_cudnn()
        with reuse_allocator(),attention:
            try:return _run(request_path,cache)
            except BaseException:
                if cache is not None:cache.clear()
                raise
    if cache is not None:cache.clear()
    return _run(request_path)


def _run(request_path,cache=None):
    request_path = Path(request_path)
    from .progress import Progress
    status=Progress(request_path.parent)
    status.update('verify')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    plan = request['plan']
    settings = plan['settings']
    components = plan.get('components', {})
    started = time.monotonic()
    from concurrent.futures import ThreadPoolExecutor
    prepared_encoder=None
    early_encoder=plan.get('runner','').startswith('krea2_') and (cache is None or cache.pipe is None)
    with ThreadPoolExecutor(max_workers=2) as startup:
        verified_components=startup.submit(verify_components,plan,cache) if early_encoder else None
        pending=(startup.submit(verify_inputs,plan,cache,verified_components) if early_encoder
                 else startup.submit(verify_inputs,plan,cache))
        print('Loading inference libraries while verifying files',flush=True)
        faulthandler.dump_traceback_later(30)
        try:
            import torch
            torch.set_num_threads(4)
            import diffusers
            import numpy as np
            from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL
        finally:faulthandler.cancel_dump_traceback_later()
        if early_encoder:
            resolved,_=verified_components.result()
            if not torch.cuda.is_available():raise RuntimeError('CUDA GPU를 사용할 수 없습니다.')
            from backend.app.inference.models.krea2 import prepare_encoder
            config=MODEL_ROOT/'configs/krea2'
            if hashlib.sha256((config/'manifest.json').read_bytes()).hexdigest()!=plan['config_sha256']:
                raise ValueError('KREA2_CONFIG_CHANGED')
            torch.cuda.reset_peak_memory_stats()
            status.update('condition')
            prepared_encoder=prepare_encoder(resolved['clip'],config,request['positive'])
            status.update('verify')
        path,model_hash,resolved,loras=pending.result()
    status.update('load')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU를 사용할 수 없습니다.')
    if prepared_encoder is None:torch.cuda.reset_peak_memory_stats()
    family = plan['runner'].split('_')[0]
    manager=None
    memory_policy=None
    if family=='krea2':
        from backend.app.inference.models.krea2 import load_pipeline
        config=MODEL_ROOT/'configs/krea2'
        if model_hash!=plan['krea2']['sha256']:raise ValueError('KREA2_HASH_MISMATCH')
        if hashlib.sha256((config/'manifest.json').read_bytes()).hexdigest()!=plan['config_sha256']:
            raise ValueError('KREA2_CONFIG_CHANGED')
        resolution=generation_resolution(settings,resolved)
        pipe=None
        if cache is not None:
            key=cache.identity(plan,resolution)
            pipe=cache.get(key,request['positive'],replace_loras=loras)
            if pipe is not None and not cache.suspended:
                from backend.app.inference.runtime.memory_policy import available_vram, available_ram, needs_vram_rebalance
                free=available_vram()[0]+torch.cuda.memory_reserved()
                host_free=available_ram()
                if host_free<8*1024**3:
                    cache.clear();pipe=None
                elif needs_vram_rebalance(pipe.memory_policy,free):
                    retained=cache.suspend()
                    print(json.dumps(dict(event='pipeline_rebalance',free_vram_with_allocator=free,
                                          free_ram_bytes=host_free,retained_on_cpu=retained)),flush=True)
                    if not retained:pipe=None
            if pipe is not None and cache.suspended:
                from backend.app.inference.models.krea2 import resume_pipeline
                resume_pipeline(pipe,resolution)
                cache.suspended=False
            if pipe is None:
                gc.collect();torch.cuda.empty_cache()
        reused=pipe is not None
        if pipe is None:
            prepared_options={'prepared_encoder':prepared_encoder} if prepared_encoder is not None else {}
            pipe=load_pipeline(path,resolved['clip'],resolved['vae'],config,adaptive=True,loras=loras,resolution=resolution,experimental_int8_gpu=True,**prepared_options)
            if cache is not None:
                cache.key,cache.pipe,cache.positive=key,pipe,request['positive']
                if prepared_encoder is not None:cache.conditioning[request['positive']]=prepared_encoder[3]
        if reused and loras:
            from backend.app.inference.conditioning.lora import set_lora_weights
            set_lora_weights(pipe,loras)
        for mapping in pipe.weight_maps:
            mapping.native_hits=0
            mapping.native_layers.clear()
            if reused:
                mapping.fp8_hits=0;mapping.fp8_layers.clear();mapping.fp8_fallbacks=0
        memory_policy=pipe.memory_policy
        from backend.app.inference.runtime.memory_policy import available_vram
        memory_policy.setdefault('free_vram_with_allocator_bytes',available_vram()[0]+torch.cuda.memory_reserved())
        memory_policy.update(allocator_cache='reuse_within_job',tensor_transfer='blocking',pipeline_reused=reused,
                             attention_policy='cudnn_first_with_native_fallback')
        print(json.dumps(dict(memory_policy=pipe.memory_policy)),flush=True)
        pipe.vae.enable_tiling()
    elif family in ('flux2','zimage'):
        if family=='zimage':from backend.app.inference.models.z_image import load_pipeline, resume_pipeline
        else:from backend.app.inference.models.flux2 import load_pipeline, resume_pipeline
        from backend.app.inference.runtime.memory_policy import available_vram, available_ram, needs_vram_rebalance
        resolution=settings['width'],settings['height']
        pipe=None
        if cache is not None:
            key=cache.identity(plan,resolution)
            pipe=cache.get(key,request['positive'],replace_loras=loras)
            if pipe is not None:
                if not cache.suspended and (needs_vram_rebalance(pipe.memory_policy,available_vram()[0]+torch.cuda.memory_reserved()) or available_ram()<8*1024**3):
                    if not cache.suspend():pipe=None
                if pipe is not None and cache.suspended:
                    resume_pipeline(pipe,*resolution)
                    cache.suspended=False
        reused=pipe is not None
        if pipe is None:
            gc.collect();torch.cuda.empty_cache()
            pipe=load_pipeline(path,*resolution,loras=loras)
            if cache is not None:cache.key,cache.pipe,cache.positive=key,pipe,request['positive']
        if reused and loras:
            from backend.app.inference.conditioning.lora import set_lora_weights
            set_lora_weights(pipe,loras)
        memory_policy=pipe.memory_policy
        memory_policy.setdefault('free_vram_with_allocator_bytes',available_vram()[0]+torch.cuda.memory_reserved())
        memory_policy['pipeline_reused']=reused
        print(json.dumps(dict(memory_policy=memory_policy)),flush=True)
    elif family=='anima':
        from backend.app.inference.models.anima import load_pipeline, configure_pipeline
        bundle=path
        if plan.get('anima'):
            bundle=(MODEL_ROOT/plan['anima']['bundle']).resolve()
            if not bundle.is_relative_to(MODEL_ROOT.resolve()) or fingerprint(bundle)!=plan['bundle_fingerprint']:
                raise ValueError('ANIMA_BUNDLE_CHANGED')
            if model_hash!=plan['anima']['sha256']:raise ValueError('ANIMA_HASH_MISMATCH')
            from backend.app.inference.models.anima import verify_bundle
            verify_bundle(bundle,cache=cache)
        resolution=generation_resolution(settings,resolved)
        pipe=None
        if cache is not None:
            from backend.app.inference.runtime.memory_policy import available_ram
            key=cache.identity(plan,resolution)
            pipe=cache.get(key,request['positive'],replace_loras=loras)
            if pipe is not None and available_ram()<8*1024**3:
                cache.clear();pipe=None
            if pipe is not None and cache.suspended:
                configure_pipeline(pipe,resolution)
                cache.suspended=False
        reused=pipe is not None
        if pipe is None:
            gc.collect();torch.cuda.empty_cache()
            pipe,manager=load_pipeline(bundle,settings['guidance'],checkpoint=path if path.is_file() else None,loras=loras,resolution=resolution)
            if cache is not None:cache.key,cache.pipe,cache.positive=key,pipe,request['positive']
        else:manager=pipe.anima_manager
        if reused and loras:
            from backend.app.inference.conditioning.lora import set_lora_weights
            set_lora_weights(pipe,loras)
        from diffusers import ClassifierFreeGuidance
        # Re-registering the scheduler resets the manager's calibrated offload reserve.
        if pipe.guider.config.guidance_scale!=settings['guidance']:
            pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=settings['guidance']))
            configure_pipeline(pipe,resolution)
        memory_policy=pipe.memory_policy
        memory_policy['pipeline_reused']=reused
        print(json.dumps(dict(memory_policy=memory_policy)),flush=True)
    else:
        if family not in ('sdxl','sd1','sd2'):
            raise ValueError('Unsupported runner')
        pipeline_class = StableDiffusionXLPipeline if family == 'sdxl' else StableDiffusionPipeline
        image_pipeline = StableDiffusionXLImg2ImgPipeline if family == 'sdxl' else StableDiffusionImg2ImgPipeline
        if plan.get('inpaint_channels') == 9:
            if family != 'sd1' or settings['flow'] != 'inpaint':raise ValueError('Invalid dedicated inpaint plan')
            from diffusers import StableDiffusionInpaintPipeline
            pipeline_class = StableDiffusionInpaintPipeline
        config = MODEL_ROOT / 'configs' / (family+'-inpaint' if plan.get('inpaint_channels') == 9 else family)
        from backend.app.inference.runtime.memory_policy import configure_pipeline, available_vram, available_ram, fresh_scheduler, needs_vram_rebalance
        resolution=generation_resolution(settings,resolved)
        if 'ip_adapter' in resolved:
            vision_config=MODEL_ROOT/'configs/ip_adapter/clip_vision_h/config.json'
            if hashlib.sha256(vision_config.read_bytes()).hexdigest()!=plan['vision_config_sha256']:raise ValueError('VISION_CONFIG_CHANGED')
        pipe=None
        if cache is not None:
            cache_key=cache.identity(plan,resolution)
            pipe=cache.get(cache_key,request['positive'],replace_loras=loras)
            if pipe is not None and not cache.suspended:
                if needs_vram_rebalance(pipe.memory_policy,available_vram()[0]+torch.cuda.memory_reserved()) or available_ram()<8*1024**3:
                    if not cache.suspend():pipe=None
            if pipe is not None and cache.suspended:
                pipe.memory_policy=configure_pipeline(pipe,*resolution)
                cache.suspended=False
        reused=pipe is not None
        if pipe is None:
            gc.collect();torch.cuda.empty_cache()
            options = {} if family == 'sdxl' else {'safety_checker':None, 'feature_extractor':None, 'requires_safety_checker':False}
            pipe = pipeline_class.from_single_file(str(path), config=str(config),
                                                  local_files_only=True, torch_dtype=torch.float16, **options)
            if 'vae' in resolved:
                pipe.vae = AutoencoderKL.from_single_file(str(resolved['vae']), config=str(config / 'vae'),
                                                        local_files_only=True, torch_dtype=torch.float16)
            pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, prediction_type=settings['prediction_type'])
            from backend.app.inference.conditioning.text_encoder import replace_encoder
            for key,attribute in (('clip','text_encoder'),('clip2','text_encoder_2')):
                if key in resolved:replace_encoder(getattr(pipe,attribute),resolved[key])
            from backend.app.inference.conditioning.lora import apply_loras
            apply_loras(pipe, loras)
            if settings.get('embeddings'):
                from backend.app.inference.conditioning.textual_inversion import apply_embeddings
                apply_embeddings(pipe,[(components[key],resolved[key]) for key in components if key.startswith('embedding')],family)
            if 'ip_adapter' in resolved:
                from backend.app.inference.conditioning.ip_adapter import attach_adapter
                attach_adapter(pipe,resolved['ip_adapter'],resolved['clip_vision'],vision_config,settings['ip_adapter_scale'])
            pipe.memory_policy=configure_pipeline(pipe,*resolution)
            if cache is not None:cache.key,cache.pipe,cache.positive=cache_key,pipe,request['positive']
        if 'ip_adapter' in resolved:pipe.set_ip_adapter_scale(settings['ip_adapter_scale'])
        if reused and loras:
            from backend.app.inference.conditioning.lora import set_lora_weights
            set_lora_weights(pipe,loras)
        pipe.scheduler=fresh_scheduler(pipe.scheduler)
        memory_policy=pipe.memory_policy
        memory_policy.update(family=family,pipeline_reused=reused)
        print(json.dumps(dict(memory_policy=memory_policy)),flush=True)
        pipe.vae.enable_tiling()
    from backend.app.inference.runtime.execution import enter_phase
    enter_phase(pipe,'idle')
    status.policy=memory_policy
    status.update('condition')
    image_args = {'width':settings['width'], 'height':settings['height']}
    if settings['flow'] in ('img2img','inpaint'):
        from PIL import Image, ImageOps
        source_path = request_path.parent / 'source.png'
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != plan['source']['sha256']:
            raise ValueError('SOURCE_CHANGED')
        with Image.open(source_path) as source:
            source_image = ImageOps.fit(source.convert('RGB'), (settings['width'],settings['height']), method=Image.Resampling.LANCZOS)
        if settings['flow']=='inpaint' and family!='krea2':
            from diffusers import StableDiffusionInpaintPipeline,StableDiffusionXLInpaintPipeline
            cls=StableDiffusionXLInpaintPipeline if family=='sdxl' else StableDiffusionInpaintPipeline
            from backend.app.inference.runtime.memory_policy import convert_pipeline
            pipe=convert_pipeline(pipe,cls)
        elif family not in ('anima','krea2'):
            from backend.app.inference.runtime.memory_policy import convert_pipeline
            pipe=convert_pipeline(pipe,image_pipeline)
        image_args = {'image':source_image, 'strength':settings['strength']}
        if settings['flow']=='inpaint':
            mask_path=request_path.parent/'mask.png'
            if hashlib.sha256(mask_path.read_bytes()).hexdigest()!=plan['mask']['sha256']:raise ValueError('MASK_CHANGED')
            with Image.open(mask_path) as source_mask:
                mask_image=ImageOps.fit(source_mask.convert('L'),source_image.size,method=Image.Resampling.NEAREST)
            image_args.update(mask_image=mask_image,width=settings['width'],height=settings['height'])

    def progress(pipeline, step, timestep, kwargs):
        total=len(pipeline.scheduler.timesteps)
        print(json.dumps(dict(step=step + 1, total=total)), flush=True)
        status.update('decode' if step+1>=total else 'generate',step=step+1,total=total)
        return kwargs

    from backend.app.inference.conditioning.prompt_tokens import count_tokens
    if family=='krea2':
        from backend.app.inference.models.krea2 import count_prompt_tokens
        token_report=count_prompt_tokens(request['positive'],request['negative'],pipe.tokenizer)
    elif family in ('flux2','zimage'):
        if family=='zimage':from backend.app.inference.models.z_image import count_prompt_tokens
        else:from backend.app.inference.models.flux2 import count_prompt_tokens
        token_report=count_prompt_tokens(path,request['positive'],request['negative'],pipe.tokenizer)
    elif family=='anima':
        from backend.app.inference.models.anima import count_prompt_tokens
        token_report=count_prompt_tokens(path,request['positive'],request['negative'],[pipe.tokenizer,pipe.t5_tokenizer])
    else:
        token_report = count_tokens(request['positive'], request['negative'], [pipe.tokenizer] + ([pipe.tokenizer_2] if family == 'sdxl' else []),family=family,mode=settings.get('prompt_mode','native'))
    call_options={} if family=='anima' else dict(guidance_scale=settings['guidance'],callback_on_step_end=progress)
    reference_args={}
    if 'reference' in plan:
        from PIL import Image,ImageOps
        reference_path=request_path.parent/'reference.png'
        if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=plan['reference']['sha256']:raise ValueError('REFERENCE_CHANGED')
        with Image.open(reference_path) as reference:reference_args['image' if family=='flux2' else 'ip_adapter_image']=ImageOps.exif_transpose(reference).convert('RGB')
    krea_conditioning=cache.conditioning if cache is not None and family=='krea2' else {}
    if family=='krea2' and prepared_encoder is not None:krea_conditioning[request['positive']]=prepared_encoder[3]
    def prompt_arguments(positive,negative):
        if family in ('flux2','zimage'):
            if family=='zimage':from backend.app.inference.models.z_image import prompt_arguments as encode_bundle_prompt
            else:from backend.app.inference.models.flux2 import prompt_arguments as encode_bundle_prompt
            from backend.app.inference.runtime.denoiser_residency import prepare_encoder, retain_denoiser
            conditioning=cache.conditioning if cache is not None else {}
            if positive not in conditioning:prepare_encoder(pipe,*resolution,resume_pipeline)
            arguments=encode_bundle_prompt(pipe,positive,conditioning)
            retain_denoiser(pipe,*resolution)
            return arguments
        if family=='krea2':
            from backend.app.inference.models.krea2 import prompt_arguments as encode_krea_prompt
            return encode_krea_prompt(pipe,positive,krea_conditioning)
        if settings.get('prompt_mode')=='weighted':
            from backend.app.inference.conditioning.prompt_encoding import encode_weighted
            return encode_weighted(pipe,positive,negative,family)
        return dict(prompt=positive,negative_prompt=negative)

    generator=torch.Generator('cpu').manual_seed(settings['seed'])
    image_conditioning=None
    if family=='krea2' and settings['flow'] in ('img2img','inpaint'):
        from backend.app.inference.models.krea2 import image_arguments
        image_args,image_conditioning=image_arguments(pipe,source_image,settings['width'],settings['height'],settings['steps'],settings['strength'],generator,
            mask=mask_image if settings['flow']=='inpaint' else None,progress=progress)
        if 'callback_on_step_end' in image_args:call_options['callback_on_step_end']=image_args.pop('callback_on_step_end')
    from backend.app.inference.runtime.execution import phase
    with phase(pipe,'condition'):arguments=prompt_arguments(request['positive'],request['negative'])
    if family=='krea2' and pipe.memory_policy['mode']=='partial' and pipe.weight_maps[0].packed_sources:
        from backend.app.inference.quantization.lora_weights import configure
        configure(pipe,model_hash,loras)
    status.update('generate')
    image = pipeline_output(pipe,on_step=lambda step,total:status.update('decode' if step>=total else 'generate',step=step,total=total),**arguments,
                 **image_args, **reference_args, num_inference_steps=settings['steps'],
                 generator=generator, **call_options).images[0]
    if settings['flow']=='inpaint':
        from PIL import Image
        image=Image.composite(image,source_image,mask_image)
    if plan.get('anima') and fingerprint(bundle)!=plan['bundle_fingerprint']:
        raise ValueError('ANIMA_BUNDLE_CHANGED during inference')
    if fingerprint(path) != plan['fingerprint']:
        raise ValueError('MODEL_CHANGED during inference')
    if family=='krea2':
        from backend.app.inference.models.krea2 import verify_config
        verify_config(config)
        if hashlib.sha256((config/'manifest.json').read_bytes()).hexdigest()!=plan['config_sha256']:
            raise ValueError('KREA2_CONFIG_CHANGED during inference')
    if not np.isfinite(np.asarray(image)).all() or np.asarray(image).std() < 1:
        raise ValueError('Image is flat or invalid')
    for kind, component in components.items():
        if fingerprint(resolved[kind]) != component['fingerprint']:
            raise ValueError('COMPONENT_CHANGED during inference')
    for selection, lora_path in loras:
        if fingerprint(lora_path) != selection['fingerprint']:
            raise ValueError('LORA_CHANGED during inference')
    image.save(request_path.parent / 'original.png')
    report = dict(schema_version=1, model_sha256=model_hash, plan=plan, torch=torch.__version__,
                  diffusers=diffusers.__version__, prompt_tokens=token_report, seconds=time.monotonic() - started,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(), scheduler=dict(pipe.scheduler.config),
                  stages=[{'name':settings.get('flow','txt2img'), 'status':'SUCCEEDED', 'size':list(image.size)}])
    if image_conditioning:report['image_conditioning']=image_conditioning
    if memory_policy:report['memory_policy']=memory_policy
    if cache is not None:report['file_hash_checks']=list(cache.hash_checks)
    if family=='krea2':report['weight_cache']=[mapping.statistics() for mapping in pipe.weight_maps]
    report['phase_seconds']=status.timings()
    report_path = request_path.parent / 'report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if any(key in resolved for key in ('face_detector','hand_detector')):
        from backend.app.inference.postprocess.detailer import detect, refine_regions, select_regions, detail_prompt
        try:
            status.update('detect')
            regions=[]
            for target in ('face','hand'):
                key=target+'_detector'
                if key in resolved:
                    regions.extend(dict(region,target=target) for region in detect(image,resolved[key],settings['detail_confidence']))
            regions=select_regions(regions,settings['detail_max_regions'],settings['seed'])
            detail_stage=dict(name='detail',status='RUNNING',regions=regions,resolution=settings['detail_resolution'],
                              order='confidence_descending_sequential',strength=settings['detail_strength'])
            report['stages'].append(detail_stage)
            report['phase_seconds']=status.timings()
            report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            if regions:
                if family not in ('anima','krea2') and plan.get('inpaint_channels') != 9:
                    from backend.app.inference.runtime.memory_policy import convert_pipeline
                    pipe=convert_pipeline(pipe,image_pipeline)

            def regenerate(crop,index,region):
                status.update('detail',region=index+1,regions=len(regions),target=region['target'])
                print(json.dumps(dict(stage='detail',region=index+1,total=len(regions),target=region['target'])),flush=True)
                crop.save(request_path.parent / f'detail-{index}-before.png')
                detail_generator=torch.Generator('cpu').manual_seed(region['seed'])
                detail_args=dict(image=crop,strength=settings['detail_strength'])
                if family=='krea2':
                    from backend.app.inference.models.krea2 import image_arguments
                    detail_args,region['image_conditioning']=image_arguments(pipe,crop,crop.width,crop.height,
                        settings['detail_steps'],settings['detail_strength'],detail_generator)
                if plan.get('inpaint_channels') == 9:
                    from PIL import Image
                    detail_args.update(mask_image=Image.new('L',crop.size,255),width=crop.width,height=crop.height)
                def detail_progress(pipeline,step,timestep,kwargs):
                    total=len(pipeline.scheduler.timesteps)
                    status.update('detail',region=index+1,regions=len(regions),target=region['target'],
                                  step=step+1,total=total,decoding=step+1>=total)
                    return kwargs
                region['prompt']=detail_prompt(region['target'],request['positive'],family)
                result=pipeline_output(pipe,on_step=lambda step,total:status.update('detail',region=index+1,regions=len(regions),target=region['target'],step=step,total=total,decoding=step>=total),**detail_args, **reference_args, **prompt_arguments(region['prompt'],request['negative']),
                    num_inference_steps=settings['detail_steps'], **({} if family=='anima' else {'guidance_scale':settings['guidance'],'callback_on_step_end':detail_progress}),
                    generator=detail_generator).images[0]
                if np.asarray(result).std() < 1:
                    raise ValueError('Invalid detail result')
                result.save(request_path.parent / f'detail-{index}-after.png')
                return result

            image,mask,records=refine_regions(image,regions,regenerate,settings['detail_resolution'])
            image.save(request_path.parent / 'detailed.png')
            mask.save(request_path.parent / 'detail-mask.png')
            detail_stage.update(status='SUCCEEDED' if regions else 'SKIPPED',reason=None if regions else 'NO_DETECTIONS',regions=records)
            for key in ('face_detector','hand_detector'):
                if key in resolved and fingerprint(resolved[key]) != components[key]['fingerprint']:
                    raise ValueError('DETECTOR_CHANGED during detail')
            report['phase_seconds']=status.timings()
            report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        except Exception as exc:
            if report['stages'][-1]['name']=='detail':
                report['stages'][-1].update(status='FAILED',error=str(exc))
            else:
                report['stages'].append(dict(name='detail',status='FAILED',error=str(exc)))
            report['phase_seconds']=status.timings()
            report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            raise
    if family=='krea2':report['weight_cache']=[mapping.statistics() for mapping in pipe.weight_maps]
    if 'upscaler' in resolved:
        try:
            status.update('upscale')
            enter_phase(pipe,'postprocess')
            retained=cache is not None and cache.suspend()
            if not retained:
                if manager:
                    manager.disable_auto_cpu_offload()
                    pipe.to('cpu')
                else:pipe.maybe_free_model_hooks()
            report['model_retained_on_cpu']=retained
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            image = upscale(image, resolved['upscaler'], components['upscaler']['profile']['scale'],progress=status)
            if fingerprint(resolved['upscaler']) != components['upscaler']['fingerprint']:
                raise ValueError('COMPONENT_CHANGED during upscale')
            report['stages'].append({'name':'upscale','status':'SUCCEEDED','size':list(image.size),'tile_size':256,'halo':32})
        except Exception as exc:
            report['stages'].append({'name':'upscale','status':'FAILED','error':str(exc)})
            report['phase_seconds']=status.timings()
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise
    if 'ip_adapter' in resolved:
        if hashlib.sha256(vision_config.read_bytes()).hexdigest()!=plan['vision_config_sha256']:raise ValueError('VISION_CONFIG_CHANGED during inference')
    status.update('save')
    image.save(request_path.parent / 'image.png')
    report.update(seconds=time.monotonic()-started, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  phase_seconds=status.timings())
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    from .offline import install_guard
    work=Path(sys.argv[1]).resolve().parent
    install_guard(work/'network-audit.jsonl')
    # Each job starts with an empty HF cache; only explicit prepared model paths may load.
    os.environ['HF_HOME']=str(work/'hf-cache')
    os.environ['HF_HUB_CACHE']=str(work/'hf-cache'/'hub')
    # The parent holds stdin open; a server crash must not leave a CUDA child alive.
    threading.Thread(target=parent_watch, daemon=True).start()
    run(sys.argv[1])
