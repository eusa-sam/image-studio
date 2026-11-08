"""Local, revision-pinned FLUX.2 Klein 4B bundle validation and loading."""
import json
from pathlib import Path

SPEC = json.loads((Path(__file__).resolve().parents[4]/'scripts/manifests/flux2-klein-4b.json').read_text(encoding='utf-8'))


def verify_bundle(folder, check_hashes=True, cache=None):
    from backend.app.inference.models.bundle import verify_bundle as verify
    return verify(folder,SPEC,'Klein',check_hashes,cache)


def bundle_fingerprint(folder):
    from backend.app.inference.models.bundle import bundle_fingerprint as fingerprint
    return fingerprint(folder,SPEC,'Klein')


def load_pipeline(folder,width,height,loras=()):
    import torch
    from diffusers import Flux2KleinPipeline
    from backend.app.inference.runtime.memory_policy import configure_pipeline
    pipe=Flux2KleinPipeline.from_pretrained(str(folder),torch_dtype=torch.bfloat16,local_files_only=True)
    from backend.app.inference.conditioning.lora import apply_loras
    apply_loras(pipe,loras)
    from backend.app.inference.runtime.cpu_weights import remember
    remember(pipe)
    pipe.memory_policy=configure_pipeline(pipe,width,height)
    pipe.memory_policy['family']='flux2'
    return pipe


def resume_pipeline(pipe,width,height):
    from backend.app.inference.runtime.memory_policy import configure_pipeline, available_ram
    if available_ram()<8*1024**3:raise RuntimeError('모델을 다시 사용할 여유 RAM이 부족합니다. 최소 8 GiB를 확보하세요.')
    from backend.app.inference.runtime.cpu_weights import remember
    remember(pipe)
    pipe.memory_policy=configure_pipeline(pipe,width,height)
    pipe.memory_policy['family']='flux2'


def execution_plan(model,settings,inventory):
    from uuid import UUID
    if settings['flow'] not in ('txt2img','reference_edit'):
        raise ValueError('Klein은 텍스트 생성 또는 참고 이미지 편집을 선택하세요. 강도 기반 img2img/마스크 수정은 미지원입니다.')
    if settings['prediction_type']!='flow' or settings['guidance']!=1 or settings['prompt_mode']!='native':
        raise ValueError('Klein distilled: flow · Guidance 1 · 기본 프롬프트 모드를 선택하세요. 권장 Steps는 4입니다.')
    if any(settings[key] for key in ('clip','clip2','vae','ip_adapter','clip_vision','face_detector','hand_detector','source_job','mask_id')):
        raise ValueError('Klein은 공식 번들의 인코더/VAE를 사용합니다. IP-Adapter·디테일러·마스크 조합은 아직 미지원입니다.')
    refs=[settings['reference_job'],settings['reference_attachment']]
    if settings['flow']=='reference_edit':
        if sum(value is not None for value in refs)!=1:raise ValueError('생성 결과 또는 사용자 참고 이미지 하나를 선택하세요.')
        for value in refs:
            if value is not None and str(UUID(value))!=value:raise ValueError('잘못된 참고 이미지 ID')
    elif any(value is not None for value in refs):raise ValueError('참고 이미지를 사용하려면 참고 이미지 편집을 선택하세요.')
    components={}
    if settings['upscaler']:
        item=next((row for row in inventory if row['id']==settings['upscaler'] and row.get('component')=='upscaler'),None)
        if item is None:raise ValueError('호환되는 업스케일러를 선택하세요.')
        components['upscaler']={key:item[key] for key in ('path','fingerprint','profile')}
    loras=[]
    for selection in settings['loras']:
        if any(row['id']==selection['model'] for row in loras):raise ValueError('같은 LoRA를 중복 선택할 수 없습니다.')
        item=next((row for row in inventory if row['id']==selection['model'] and row.get('component')=='lora' and row['profile']['family']=='flux2'),None)
        if item is None:raise ValueError('Klein과 호환되는 FLUX.2 LoRA 프로필이 필요합니다.')
        loras.append({key:item[key] for key in ('id','path','fingerprint','profile')} | {'weight':selection['weight']})
    return dict(schema_version=1,runner='flux2_'+settings['flow'],model_path=model['path'],fingerprint=model['fingerprint'],settings=settings,components=components,loras=loras)


def count_prompt_tokens(folder,positive,negative,tokenizer=None):
    if tokenizer is None:
        from transformers import Qwen2TokenizerFast
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(Path(folder)/'tokenizer'),local_files_only=True)
    messages=[{'role':'user','content':positive}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    count=len(tokenizer(text,truncation=False)['input_ids'])
    return dict(encoders=[dict(encoder='Qwen3 4B',positive=count,negative=0,limit=512,truncated=count>512)],truncated=count>512,negative_used=False)


def prompt_arguments(pipe,positive,cache):
    import torch
    if positive not in cache:
        with torch.inference_mode():
            embeddings,_=pipe.encode_prompt(prompt=positive,device=pipe._execution_device)
        cache[positive]={'prompt_embeds':embeddings}
    return {name:value.to(pipe._execution_device) for name,value in cache[positive].items()}
