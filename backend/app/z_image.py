"""Pinned original and locally prepared BF16 Z-Image-Turbo bundles."""
import json
from pathlib import Path

SPEC = json.loads((Path(__file__).resolve().parents[2]/'scripts/manifests/z-image-turbo.json').read_text(encoding='utf-8'))
BF16_SPEC = json.loads((Path(__file__).resolve().parents[2]/'scripts/manifests/z-image-turbo-bf16.json').read_text(encoding='utf-8'))


def bundle_spec(folder):
    manifest=json.loads((Path(folder)/'download_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest,dict):raise ValueError('Invalid Z-Image manifest')
    variant=manifest.get('variant')
    if variant is None:return SPEC
    if variant==BF16_SPEC['variant']:return BF16_SPEC
    raise ValueError('Unsupported Z-Image bundle variant')


def activation_reserve(width,height):
    # Calibrated for this BF16 runner, batch one, 512 prompt tokens, through 1024 square.
    if max(width,height)>1024:return None
    return max(2*1024**3,int(3*1024**3*width*height/(1024*1024)))


def configure_pipeline(pipe,width,height):
    import torch
    from .memory_policy import configure_pipeline as configure
    policy=configure(pipe,width,height,activation_reserve_bytes=activation_reserve(width,height))
    if policy['mode']!='sequential_offload' or getattr(pipe,'text_encoder',None) is not None:
        return policy
    from accelerate import infer_auto_device_map,dispatch_model,cpu_offload
    vae_bytes=sum(value.numel()*value.element_size() for value in list(pipe.vae.parameters())+list(pipe.vae.buffers()))
    budget=max(0,policy['resident_budget_bytes']-vae_bytes)
    cpu_budget=max(0,policy['free_ram_bytes']-8*1024**3)
    if not budget or not cpu_budget:return policy
    mapping=infer_auto_device_map(pipe.transformer,max_memory={0:budget,'cpu':cpu_budget},
        no_split_module_classes=pipe.transformer._no_split_modules,dtype=pipe.transformer.dtype)
    # Keep the existing safe path when RAM cannot hold the planned offloaded blocks.
    if set(mapping.values())!={0,'cpu'}:return policy
    pipe.remove_all_hooks();pipe.to('cpu')
    dispatch_model(pipe.transformer,mapping,main_device='cuda:0')
    cpu_offload(pipe.vae,execution_device=torch.device('cuda:0'))
    policy.update(mode='block_offload',device_map=mapping,transformer_budget_bytes=budget)
    return policy


def verify_bundle(folder, check_hashes=True, cache=None):
    from .bundle import verify_bundle as verify
    return verify(folder,bundle_spec(folder),'Z-Image',check_hashes,cache)


def bundle_fingerprint(folder):
    from .bundle import bundle_fingerprint as fingerprint
    return fingerprint(folder,bundle_spec(folder),'Z-Image')


def load_pipeline(folder,width,height,loras=()):
    import torch
    from diffusers import ZImagePipeline
    pipe=ZImagePipeline.from_pretrained(str(folder),torch_dtype=torch.bfloat16,local_files_only=True)
    from .lora import apply_loras
    apply_loras(pipe,loras)
    pipe.memory_policy=configure_pipeline(pipe,width,height)
    pipe.memory_policy['family']='zimage'
    pipe._encoder_source=Path(folder)/'text_encoder'
    pipe._generation_size=(width,height)
    return pipe


def resume_pipeline(pipe,width,height):
    from .memory_policy import available_ram
    if available_ram()<8*1024**3:raise RuntimeError('모델을 다시 사용할 여유 RAM이 부족합니다. 최소 8 GiB를 확보하세요.')
    pipe.memory_policy=configure_pipeline(pipe,width,height)
    pipe.memory_policy['family']='zimage'
    pipe.memory_policy['text_encoder_released']=getattr(pipe,'text_encoder',None) is None
    pipe._generation_size=(width,height)


def execution_plan(model,settings,inventory):
    if settings['flow']!='txt2img':raise ValueError('Z-Image-Turbo는 현재 텍스트 생성을 지원합니다.')
    if settings['prediction_type']!='flow' or settings['guidance']!=0 or settings['prompt_mode']!='native':
        raise ValueError('Z-Image-Turbo: flow · Guidance 0 · 기본 프롬프트 모드를 선택하세요. 권장 Steps는 9입니다.')
    if any(settings[key] for key in ('clip','clip2','vae','ip_adapter','clip_vision','face_detector','hand_detector','source_job','mask_id','reference_job','reference_attachment')):
        raise ValueError('Z-Image-Turbo는 공식 번들의 인코더/VAE를 사용합니다. 참고 이미지·디테일러·마스크 조합은 아직 미지원입니다.')
    components={}
    if settings['upscaler']:
        item=next((row for row in inventory if row['id']==settings['upscaler'] and row.get('component')=='upscaler'),None)
        if item is None:raise ValueError('호환되는 업스케일러를 선택하세요.')
        components['upscaler']={key:item[key] for key in ('path','fingerprint','profile')}
    loras=[]
    for selection in settings['loras']:
        if any(row['id']==selection['model'] for row in loras):raise ValueError('같은 LoRA를 중복 선택할 수 없습니다.')
        item=next((row for row in inventory if row['id']==selection['model'] and row.get('component')=='lora' and row['profile']['family']=='zimage'),None)
        if item is None:raise ValueError('Z-Image와 호환되는 Z-Image LoRA 프로필이 필요합니다.')
        loras.append({key:item[key] for key in ('id','path','fingerprint','profile')} | {'weight':selection['weight']})
    return dict(schema_version=1,runner='zimage_txt2img',model_path=model['path'],fingerprint=model['fingerprint'],settings=settings,components=components,loras=loras)


def count_prompt_tokens(folder,positive,negative,tokenizer=None):
    if tokenizer is None:
        from transformers import Qwen2TokenizerFast
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(Path(folder)/'tokenizer'),local_files_only=True)
    text=tokenizer.apply_chat_template([dict(role='user',content=positive)],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=True)
    count=len(tokenizer(text,truncation=False)['input_ids'])
    return dict(encoders=[dict(encoder='Qwen3 4B',positive=count,negative=0,limit=512,truncated=count>512)],
                truncated=count>512,negative_used=False)


def reconfigure_after_encoder_change(pipe):
    import torch
    from .memory_policy import available_vram
    width,height=pipe._generation_size
    policy=configure_pipeline(pipe,width,height)
    for key in ('device_map','transformer_budget_bytes'):pipe.memory_policy.pop(key,None)
    pipe.memory_policy.update(policy)
    pipe.memory_policy['free_vram_with_allocator_bytes']=available_vram()[0]+torch.cuda.memory_reserved()


def prompt_arguments(pipe,positive,cache):
    import torch
    import gc
    from .memory_policy import available_ram
    managed=isinstance(getattr(pipe,'_encoder_source',None),Path)
    if positive not in cache:
        if managed and pipe.text_encoder is None:
            from transformers import Qwen3Model
            # A residency change may have put the transformer on GPU since the last prompt.
            pipe.remove_all_hooks();pipe.to('cpu');torch.cuda.empty_cache()
            pipe.text_encoder=Qwen3Model.from_pretrained(str(pipe._encoder_source),torch_dtype=torch.bfloat16,local_files_only=True)
            reconfigure_after_encoder_change(pipe)
            pipe.memory_policy['text_encoder_released']=False
        with torch.inference_mode():
            embeddings,_=pipe.encode_prompt(prompt=positive,device=pipe._execution_device,
                                             do_classifier_free_guidance=False,max_sequence_length=512)
        if len(embeddings)!=1:raise ValueError('Expected one Z-Image prompt embedding')
        cache.clear()
        cache[positive]={'prompt_embeds':embeddings[0]}
    if managed and pipe.text_encoder is not None and pipe.memory_policy['mode']!='full':
        transformer_bytes=sum(value.numel()*value.element_size() for value in list(pipe.transformer.parameters())+list(pipe.transformer.buffers()))
        # Drop the idle encoder before choosing faster denoiser residency.
        if pipe.memory_policy['mode']=='sequential_offload' or available_ram()<8*1024**3+transformer_bytes:
            pipe.remove_all_hooks();pipe.to('cpu')
            pipe.text_encoder=None
            gc.collect();torch.cuda.empty_cache()
            reconfigure_after_encoder_change(pipe)
            pipe.memory_policy['text_encoder_released']=True
    return {'prompt_embeds':[cache[positive]['prompt_embeds'].to(pipe._execution_device)]}
