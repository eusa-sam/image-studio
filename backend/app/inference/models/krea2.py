"""Offline Krea2 component assembly from the user's local single-file weights."""
import hashlib
import json
from pathlib import Path


def image_arguments(pipe, image, width, height, steps, strength, generator, *, mask=None, progress=None):
    """Start the existing Turbo denoiser from a noised, normalized VAE image."""
    import numpy as np
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from backend.app.inference.quantization.quantized_weights import weight_readers
    from backend.app.inference.runtime.execution import phase
    count=int(steps*strength)
    if not 0<strength<=1 or count<1:raise ValueError('Steps × strength must be at least one')
    sigmas=np.linspace(1.,1/steps,steps)[steps-count:]
    scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config)
    scheduler.set_timesteps(sigmas=sigmas,mu=1.15,device='cpu')
    sigma=float(scheduler.sigmas[0])
    device=pipe._execution_device
    with torch.inference_mode(),weight_readers(),phase(pipe,'encode'):
        pixels=pipe.image_processor.preprocess(image,height=height,width=width).to(device=device,dtype=pipe.vae.dtype).unsqueeze(2)
        latent=pipe.vae.encode(pixels).latent_dist.mode()
        mean=torch.tensor(pipe.vae.config.latents_mean,device=device,dtype=latent.dtype).view(1,-1,1,1,1)
        std=torch.tensor(pipe.vae.config.latents_std,device=device,dtype=latent.dtype).view(1,-1,1,1,1)
        latent=((latent-mean)/std)[:,:,0]
        noise=randn_tensor(latent.shape,generator=generator,device=device,dtype=latent.dtype)
        original=latent
        latent=(1-sigma)*latent+sigma*noise
        packed=pipe._pack_latents(latent,*latent.shape)
    arguments=dict(latents=packed,sigmas=sigmas,width=width,height=height)
    info=dict(strength=strength,start_sigma=sigma,denoising_steps=count,vae_sampling='mode')
    if mask is not None:
        from PIL import Image
        mask_values=np.asarray(mask.convert('L').resize((width,height),Image.Resampling.NEAREST),dtype=np.float32)/255
        mask_tensor=torch.from_numpy(mask_values.copy()).to(device=device,dtype=original.dtype)[None,None]
        mask_tensor=torch.nn.functional.interpolate(mask_tensor,size=original.shape[-2:],mode='nearest')
        mask_tensor=mask_tensor.expand_as(original)
        packed_mask=pipe._pack_latents(mask_tensor,*mask_tensor.shape)
        original=pipe._pack_latents(original,*original.shape)
        noise=pipe._pack_latents(noise,*noise.shape)
        def preserve(pipeline,step,timestep,kwargs):
            sigma=pipeline.scheduler.sigmas[step+1].to(device=original.device,dtype=original.dtype)
            source=(1-sigma)*original+sigma*noise
            kwargs['latents']=packed_mask*kwargs['latents']+(1-packed_mask)*source
            return progress(pipeline,step,timestep,kwargs) if progress else kwargs
        arguments['callback_on_step_end']=preserve
        info['mask_mode']='per_step_latent_blend_and_pixel_composite'
    return arguments,info


def prompt_arguments(pipe, positive, cache):
    """Reuse exact conditioning within one worker, including each detail target's prefix."""
    import torch
    if positive not in cache:
        from backend.app.inference.quantization.quantized_weights import weight_readers
        encoder=pipe.weight_maps[1] if getattr(pipe,'memory_policy',{}).get('mode')=='partial' else None
        try:
            if encoder is not None:
                from backend.app.inference.runtime.memory_policy import available_vram
                reusable=max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
                encoder.transient_gpu_budget=max(0,available_vram()[0]+reusable-2*1024**3)
            with torch.inference_mode(),weight_readers():
                embeddings,mask=pipe.encode_prompt(positive,device=pipe._execution_device)
        finally:
            if encoder is not None:encoder.transient_gpu_budget=0
        cache[positive]=dict(prompt_embeds=embeddings,prompt_embeds_mask=mask)
    return {name:value.to(pipe._execution_device) for name,value in cache[positive].items()}


def partial_cache_budgets(pipe,policy):
    maps=pipe.weight_maps
    cpu_budget=max(0,policy['cpu_cache_budget_bytes']-sum(getattr(mapping,'extra_cpu_bytes',0) for mapping in maps))
    vae_cache=min(cpu_budget,sum(p.numel()*p.element_size() for p in pipe.vae.parameters()))
    maps[0].limit_cpu_cache(cpu_budget-vae_cache)
    maps[2].limit_cpu_cache(vae_cache)
    vae_gpu=min(vae_cache,policy['resident_budget_bytes'])
    maps[0].budget=policy['resident_budget_bytes']-vae_gpu
    maps[2].budget=vae_gpu
    for index in (0,2):maps[index].reserve=policy['activation_reserve_bytes']
    policy.update(vae_cpu_cache_budget_bytes=vae_cache,vae_gpu_cache_budget_bytes=vae_gpu)


def resume_pipeline(pipe,resolution):
    import torch
    from backend.app.inference.runtime.memory_policy import available_vram, available_ram, residency_plan
    torch.cuda.empty_cache()
    free,total,measurement=available_vram();host=available_ram()
    from backend.app.inference.runtime.memory_policy import cached_storage_bytes
    retained=cached_storage_bytes(pipe,'cpu')
    policy=residency_plan(free,total,pipe.memory_policy['weight_bytes'],*resolution,free_ram=host+retained)
    policy.update(mode='partial',measurement=measurement,free_ram_bytes=host)
    partial_cache_budgets(pipe,policy)
    pipe.memory_policy.update(policy)
    pipe.memory_policy.pop('free_vram_with_allocator_bytes',None)


def count_prompt_tokens(positive, negative, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        from backend.app.models import MODEL_ROOT
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ROOT/'configs/krea2/tokenizer',local_files_only=True)
    prefix='<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n'
    count=max(0,len(tokenizer(prefix+positive)['input_ids'])-34)
    row=dict(encoder='Qwen3-VL 4B',positive=count,negative=0,limit=507,chunks=1,truncated=count>507)
    return dict(encoders=[row],truncated=row['truncated'],mode='native',negative_used=False)


def text_keys(state):
    result={}
    for key,value in state.items():
        name=key.removeprefix('model.')
        if not name.startswith('visual.'):name='language_model.'+name
        result[name]=value
    return result


def verify_config(config_root):
    root=Path(config_root).resolve()
    rows=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['files']
    required={'text_encoder/config.json','vae/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json'}
    if len(rows)!=4 or {row['path'] for row in rows}!=required:
        raise ValueError('Krea2 supporting config manifest incomplete')
    for row in rows:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Krea2 supporting config changed')
    return root


def prepare_encoder(path, config_root, positive):
    import torch
    from accelerate import cpu_offload
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel
    from diffusers import Krea2Pipeline
    from backend.app.inference.quantization.quantized_weights import FileWeights, weight_readers
    from backend.app.inference.runtime.memory_policy import available_vram, weight_validation_device
    # Larger GPUs retain the existing full-residency loading path.
    if available_vram()[1]>=32*1024**3:return None
    root=verify_config(config_root)
    with torch.device('meta'):
        text=Qwen3VLModel(Qwen3VLConfig.from_json_file(root/'text_encoder/config.json'))
    text.visual.rotary_pos_emb=type(text.visual.rotary_pos_emb)(text.config.vision_config)
    text.language_model.rotary_emb=type(text.language_model.rotary_emb)(text.config.text_config)
    from safetensors import safe_open
    with safe_open(str(path),framework='pt',device='cpu') as reader:
        has_int8=any(reader.get_slice(key).get_dtype()=='I8' for key in reader.keys())
    device=weight_validation_device(available_vram()[0],max(p.numel() for p in text.parameters()),has_int8)
    with weight_readers():
        mapping=FileWeights(text,str(path),text_keys,validation_device=device,defer_numeric_validation=True)
        if device=='cuda:0':torch.cuda.empty_cache()
        mapping.bind(text)
        cpu_offload(text,torch.device('cuda:0'),state_dict=mapping,offload_buffers=len(text._parameters)>0)
        from backend.app.inference.ops.native_fp8 import attach_native_linear
        attach_native_linear(text,mapping)
        tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
        temporary=Krea2Pipeline(transformer=None,text_encoder=text,vae=None,tokenizer=tokenizer,scheduler=None,is_distilled=True)
        temporary.weight_maps=[None,mapping]
        temporary.memory_policy={'mode':'partial'}
        arguments=prompt_arguments(temporary,positive,{})
        return text,mapping,tokenizer,arguments,device


def load_pipeline(checkpoint, text_encoder, vae_path, config_root, *, adaptive=False, loras=(), resolution=(1024,1024), experimental_int8_gpu=False, prepared_encoder=None):
    import torch
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel
    from diffusers import Krea2Pipeline, Krea2Transformer2DModel, AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
    from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers, convert_wan_vae_to_diffusers
    from backend.app.inference.quantization.quantized_weights import load_local_weights
    root=verify_config(config_root)
    if prepared_encoder is not None and not adaptive:raise ValueError('Prepared encoder requires adaptive loading')
    with torch.device('meta'):
        transformer=Krea2Transformer2DModel()
        text=prepared_encoder[0] if prepared_encoder is not None else Qwen3VLModel(Qwen3VLConfig.from_json_file(root/'text_encoder/config.json'))
        vae=AutoencoderKLQwenImage.from_config(json.loads((root/'vae/config.json').read_text()))
    # Rotary buffers are derived from config and intentionally absent from state_dict.
    if prepared_encoder is None:
        text.visual.rotary_pos_emb=type(text.visual.rotary_pos_emb)(text.config.vision_config)
        text.language_model.rotary_emb=type(text.language_model.rotary_emb)(text.config.text_config)
    sources=[(transformer,checkpoint,convert_krea2_transformer_checkpoint_to_diffusers),
             (text,text_encoder,text_keys),(vae,vae_path,convert_wan_vae_to_diffusers)]
    maps=[]
    from backend.app.inference.runtime.memory_policy import available_ram, int8_preload_budget
    preload_budget=int8_preload_budget(checkpoint,available_ram()) if adaptive and not experimental_int8_gpu else 0
    validation_devices={}
    for model,path,convert in sources:
        if model is text and prepared_encoder is not None:
            maps.append(prepared_encoder[1])
            validation_devices[type(text).__name__]=prepared_encoder[4]
            continue
        if adaptive:
            from backend.app.inference.quantization.quantized_weights import FileWeights
            from safetensors import safe_open
            from backend.app.inference.runtime.memory_policy import available_vram, weight_validation_device
            with safe_open(str(path),framework='pt',device='cpu') as reader:
                has_int8=any(reader.get_slice(key).get_dtype()=='I8' for key in reader.keys())
            # INT8 GPU rounding is enabled only by the caller after quality validation.
            free=available_vram()[0] if not has_int8 or experimental_int8_gpu else 0
            device=weight_validation_device(free,max(p.numel() for p in model.parameters()),has_int8,allow_int8=experimental_int8_gpu)
            validation_devices[type(model).__name__]=device
            maps.append(FileWeights(model,str(path),convert,preload_budget=preload_budget if model is transformer else 0,
                                    validation_device=device,defer_numeric_validation=True))
            if device=='cuda:0':torch.cuda.empty_cache()
        else:load_local_weights(model,str(path),convert)
    tokenizer=prepared_encoder[2] if prepared_encoder is not None else AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
    scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True,base_shift=.5,max_shift=1.15,
        base_image_seq_len=256,max_image_seq_len=6400)
    pipe=Krea2Pipeline(transformer=transformer,text_encoder=text,vae=vae,tokenizer=tokenizer,
                      scheduler=scheduler,is_distilled=True)
    if adaptive:
        from backend.app.inference.conditioning.lora import apply_loras
        from backend.app.inference.runtime.memory_policy import residency_plan, available_vram, available_ram
        from accelerate import cpu_offload
        # Diffusers otherwise detaches the already active encoder offload hooks.
        if prepared_encoder is not None:pipe.register_modules(text_encoder=None)
        try:apply_loras(pipe,loras)
        finally:
            if prepared_encoder is not None:pipe.register_modules(text_encoder=text)
        weights=sum(p.numel()*p.element_size() for model,_,_ in sources for p in model.parameters())
        free,total,measurement=available_vram()
        host_free=available_ram()
        retained=sum(mapping.cpu_bytes for mapping in maps)
        policy=residency_plan(free,total,weights,*resolution,free_ram=host_free+retained)
        if any(mapping.packed_sources for mapping in maps):policy['mode']='partial'
        policy.update(free_ram_bytes=host_free,preloaded_cpu_bytes=retained)
        policy['family']='krea2'
        policy['measurement']=measurement
        policy['validation_devices']=validation_devices
        policy['packed_int8_gpu']=experimental_int8_gpu
        if any(info[2] for info in maps[0].packed_sources.values()):policy['int8_restore']='rotate_then_scale'
        for (model,_,_),mapping in zip(sources,maps):mapping.bind(model)
        pipe.weight_maps=maps
        if policy['mode']=='full':
            for (model,_,_),mapping in zip(sources,maps):
                for name in mapping:model.load_state_dict({name:mapping[name].to('cuda')},strict=False,assign=True)
                model.to('cuda')
        else:
            # Cache the repeatedly used denoiser, leaving encoders/VAE demand-loaded.
            maps[0].packed_enabled=True
            maps[0].packed_int8_enabled=experimental_int8_gpu
            partial_cache_budgets(pipe,policy)
            for (model,_,_),mapping in zip(sources,maps):
                if model is text and prepared_encoder is not None:continue
                cpu_offload(model,torch.device('cuda:0'),state_dict=mapping,
                            offload_buffers=len(model._parameters)>0)
                from backend.app.inference.ops.native_fp8 import attach_native_linear
                attach_native_linear(model,mapping)
            if experimental_int8_gpu:
                from backend.app.inference.ops.native_int8 import attach_native_linear
                attach_native_linear(transformer,maps[0])
                policy['int8_compute']='native_convrot_with_bf16_fallback'
            policy['fp8_compute']='scaled_mm_dynamic_input_with_bf16_fallback'
        if loras:
            from backend.app.inference.models.krea2_lora import skip_unit_scaling
            skip_unit_scaling(transformer)
        pipe.memory_policy=policy
        pipe.weight_maps=maps
    return pipe
