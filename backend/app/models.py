"""CPU-only discovery and conservative SD1/SDXL execution preflight."""
import hashlib
import json
from typing import Literal
from uuid import UUID
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from scripts.inspect_models import inspect

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / 'models'


class LoraSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model: str
    weight: float = Field(default=1.0, ge=-2, le=2)


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model: str
    width: int = Field(default=1024, ge=256, le=1024, multiple_of=64)
    height: int = Field(default=1024, ge=256, le=1024, multiple_of=64)
    steps: int = Field(default=20, ge=1, le=60)
    guidance: float = Field(default=5.0, ge=0, le=20)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    random_seed: bool = False
    prediction_type: str = Field(default='epsilon', pattern='^(epsilon|v_prediction|flow)$')
    vae: str | None = None
    clip: str | None = None
    clip2: str | None = None
    ip_adapter: str | None = None
    clip_vision: str | None = None
    ip_adapter_scale: float = Field(default=0.7, ge=0, le=2)
    reference_job: str | None = None
    reference_artifact: Literal['image.png', 'original.png'] = 'image.png'
    reference_attachment: str | None = None
    upscaler: str | None = None
    prompt_mode: Literal['native','weighted'] = 'native'
    flow: Literal['txt2img', 'img2img', 'inpaint', 'reference_edit'] = 'txt2img'
    mask_id: str | None = None
    source_job: str | None = None
    source_artifact: Literal['image.png', 'original.png'] = 'image.png'
    strength: float = Field(default=0.5, gt=0, le=1)
    loras: list[LoraSelection] = Field(default_factory=list, max_length=4)
    embeddings: list[str] = Field(default_factory=list, max_length=4)
    face_detector: str | None = None
    hand_detector: str | None = None
    detail_strength: float = Field(default=0.3, gt=0, le=0.8)
    detail_steps: int = Field(default=20, ge=4, le=60)
    detail_confidence: float = Field(default=0.35, ge=0.1, le=0.95)
    detail_max_regions: int = Field(default=4, ge=1, le=8)
    detail_resolution: Literal[512,768,1024] = 1024


class ComponentProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    schema_version: Literal[1]
    kind: Literal['vae', 'upscaler', 'lora', 'detector', 'text_encoder', 'ip_adapter', 'clip_vision', 'embedding']
    family: Literal['sdxl', 'sd1', 'sd2', 'anima', 'krea2', 'flux2', 'zimage', 'spandrel', 'yolov8']
    sha256: str = Field(pattern='^[0-9a-f]{64}$')
    scale: int = Field(ge=1, le=4)
    source: str
    target: Literal['face', 'hand'] | None = None
    slot: Literal[1,2] | None = None
    tokens: list[str] | None = None


class AnimaProfile(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    schema_version: Literal[1]
    variant: Literal['base','turbo','aesthetic']
    sha256: str = Field(pattern='^[0-9a-f]{64}$')
    bundle: str
    source: str


class KreaProfile(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    schema_version: Literal[1]
    variant: Literal['turbo']
    sha256: str = Field(pattern='^[0-9a-f]{64}$')
    text_encoder: str
    vae: str
    source: str


def bundle_family(path):
    if not (path/'model_index.json').is_file():return 'anima'
    index=json.loads((path/'model_index.json').read_text(encoding='utf-8'))
    family={'Flux2KleinPipeline':'flux2','ZImagePipeline':'zimage'}.get(index.get('_class_name')) if isinstance(index,dict) else None
    if family is None:raise ValueError('이 Diffusers bundle의 실행기는 아직 미지원입니다.')
    return family


def fingerprint(path):
    if path.is_dir():
        family=bundle_family(path)
        if family=='flux2':
            from .flux2 import bundle_fingerprint
        elif family=='zimage':
            from .z_image import bundle_fingerprint
        else:
            from .anima import bundle_fingerprint
        return bundle_fingerprint(path)
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def scan():
    result = []
    paths=[]
    for folder,directories,files in MODEL_ROOT.walk():
        paths.extend(folder/name for name in directories+files)
        directories[:]=[name for name in directories if not (
            (folder==MODEL_ROOT and name.casefold()=='sources') or
            (folder.name in ('checkpoint','checkpoints') and any(
                (folder/name/marker).is_file() for marker in ('modular_model_index.json','model_index.json'))))]
    for path in sorted(paths):
        if path.relative_to(MODEL_ROOT).parts[0].casefold()=='sources':
            continue
        if path.is_dir() and ((path/'modular_model_index.json').is_file() or (path/'model_index.json').is_file()) and path.parent.name in ('checkpoint','checkpoints'):
            relative=path.relative_to(MODEL_ROOT).as_posix()
            family='unknown'
            row=dict(id=hashlib.sha256(relative.encode()).hexdigest()[:24],name=path.name,path=relative,family=family,available=False)
            try:
                if not path.resolve().is_relative_to(MODEL_ROOT.resolve()):raise ValueError('모델 루트 밖 경로')
                family=bundle_family(path);row['family']=family
                if family=='zimage':
                    from .z_image import bundle_spec
                    row['variant']=bundle_spec(path).get('variant','original')
                row.update(fingerprint=fingerprint(path),available=True,reason={'flux2':'FLUX.2 Klein 4B','zimage':'Z-Image-Turbo','anima':'ANIMA'}[family]+' bundle · 실행 시 전체 해시 검사')
            except (OSError,ValueError,KeyError,TypeError) as exc:row['reason']=str(exc)
            result.append(row)
            continue
        if any((parent/'modular_model_index.json').is_file() or (parent/'model_index.json').is_file() for parent in path.parents if parent!=MODEL_ROOT and parent.is_relative_to(MODEL_ROOT) and parent.parent.name in ('checkpoint','checkpoints')):
            continue
        if not path.is_file() or path.suffix.lower() not in ('.safetensors', '.pth', '.pt', '.gguf', '.onnx'):
            continue
        relative = path.relative_to(MODEL_ROOT).as_posix()
        row = dict(id=hashlib.sha256(relative.encode()).hexdigest()[:24], name=path.name,
                   path=relative, family='unknown', available=False, reason='실행기가 아직 구현되지 않은 구성요소입니다.')
        try:
            if not path.resolve().is_relative_to(MODEL_ROOT.resolve()):
                raise ValueError('모델 폴더 밖의 링크는 사용할 수 없습니다.')
            rejected=path.with_name(path.name+'.rejected.json')
            if rejected.exists():
                raise ValueError('검증 거절 기록이 있는 파일입니다. 재검증 후 정상 프로필을 준비해야 선택할 수 있습니다.')
            before = fingerprint(path)
            if path.suffix.lower() == '.safetensors':
                info = inspect(path, root=MODEL_ROOT)
                row['family'] = info['family']
                if info['family']=='vision_encoder':
                    row['reason']='이미지 특징 인코더입니다. CLIP 텍스트 슬롯에는 쓸 수 없으며, 호환 이미지 조건 실행기가 아직 없습니다.'
                if info['family'] in ('krea2','krea2_candidate'):
                    row['reason'] = 'Krea2 구조 후보: Turbo 종류와 인코더/VAE를 지정한 .krea2.json 프로필이 필요합니다.'
                    profile_path=path.with_name(path.name+'.krea2.json')
                    if profile_path.exists():
                        if not profile_path.resolve().is_relative_to(MODEL_ROOT.resolve()):raise ValueError('프로필 경로가 모델 루트 밖입니다.')
                        profile=KreaProfile.model_validate_json(profile_path.read_text(encoding='utf-8')).model_dump()
                        row.update(family='krea2',available=True,krea2=profile,reason='Krea2 Turbo · Qwen3-VL + Qwen Image VAE · 텍스트 생성')
                if info['family']=='anima_candidate':
                    profile_path=path.with_name(path.name+'.anima.json')
                    if not profile_path.resolve().is_relative_to(MODEL_ROOT.resolve()):raise ValueError('프로필 경로가 모델 루트 밖입니다.')
                    if not profile_path.exists():raise ValueError('ANIMA 구조 후보: Base/Turbo 종류와 로컬 bundle을 지정한 .anima.json 프로필이 필요합니다.')
                    profile=AnimaProfile.model_validate_json(profile_path.read_text(encoding='utf-8')).model_dump()
                    bundle=(MODEL_ROOT/profile['bundle']).resolve()
                    if not bundle.is_relative_to(MODEL_ROOT.resolve()):raise ValueError('ANIMA bundle 경로가 모델 루트 밖입니다.')
                    support=fingerprint(bundle)
                    row.update(family='anima',available=True,anima=profile,bundle_fingerprint=support,reason='ANIMA '+profile['variant']+' 단일 파일 · 실행 시 해시/전체 key 검사')
                if info['family'] in ('sdxl_candidate','sd1_candidate','sd2_candidate','sd1_inpaint_candidate') and relative.split('/')[0] in ('checkpoint', 'checkpoints'):
                    row.update(available=True, reason=info['family']+' 구조 후보. prediction type을 확인하세요. 로드 시 최종 검사합니다.')
            if fingerprint(path) != before:
                raise ValueError('파일이 변경 중입니다. 복사가 끝나면 다시 검색하세요.')
            row['fingerprint'] = before
            profile_path = path.with_name(path.name + '.profile.json')
            if profile_path.exists():
                if not profile_path.resolve().is_relative_to(MODEL_ROOT.resolve()):
                    raise ValueError('프로필 경로가 모델 폴더 밖입니다.')
                profile = ComponentProfile.model_validate_json(profile_path.read_text(encoding='utf-8')).model_dump()
                folder = relative.split('/')[0]
                valid = (profile['kind']=='embedding' and profile['family'] in ('sd1','sd2','sdxl')
                         and folder=='embeddings' and path.suffix=='.safetensors' and profile['scale']==1) or (profile['kind'] in ('ip_adapter','clip_vision') and profile['family']=='sdxl'
                         and folder==profile['kind'] and path.suffix=='.safetensors' and profile['scale']==1) or (profile['kind']=='text_encoder' and profile['family'] in ('sdxl','sd1','sd2','krea2')
                         and folder in ('clip','text_encoder','text_encoders') and path.suffix=='.safetensors'
                         and profile['scale']==1 and profile['slot'] in ((1,2) if profile['family']=='sdxl' else (1,))) or (
                         profile['kind'] == 'vae' and profile['family'] in ('sdxl','sd1','sd2','krea2') and folder == 'vae'
                         and path.suffix == '.safetensors' and profile['scale'] == 1) or (
                         profile['kind'] == 'upscaler' and profile['family'] == 'spandrel'
                         and folder in ('upscale_models', 'upscalers') and path.suffix in ('.pth','.safetensors')) or (
                         profile['kind'] == 'lora' and profile['family'] in ('sdxl','sd1','sd2','anima','krea2','flux2','zimage') and folder in ('lora','loras')
                         and path.suffix == '.safetensors' and profile['scale'] == 1 and (row['family'] != 'krea2_lora' or profile['family']=='krea2')) or (
                         profile['kind'] == 'detector' and profile['family'] == 'yolov8' and folder in ('detector','detectors')
                         and path.suffix == '.onnx' and profile['scale'] == 1 and profile['target'] in ('face','hand'))
                if not valid:
                    raise ValueError('구성요소 프로필과 폴더/형식이 맞지 않습니다.')
                if profile['kind']=='embedding':
                    from .textual_inversion import read_embedding
                    read_embedding(path,profile)
                row.update(component=profile['kind'], profile=profile, reason='로컬 프로필 있음. 사용 시 파일 해시와 로더 호환성을 검사합니다.')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(available=False, reason=str(exc))
        result.append(row)
    return result


def embedding_components(selected, inventory, family):
    if not selected:return {}
    if family not in ('sd1','sd2','sdxl'):raise ValueError('임베딩은 SD1/SD2/SDXL만 지원합니다.')
    if len(set(selected))!=len(selected):raise ValueError('같은 임베딩을 중복 선택할 수 없습니다.')
    components={}
    for index,identifier in enumerate(selected):
        item=next((m for m in inventory if m['id']==identifier and m.get('component')=='embedding'),None)
        if not item or item['profile']['family']!=family:raise ValueError('임베딩 계열/프로필을 확인하세요.')
        path=(MODEL_ROOT/item['path']).resolve()
        if not path.is_relative_to(MODEL_ROOT.resolve()) or fingerprint(path)!=item['fingerprint']:raise ValueError('EMBEDDING_CHANGED')
        components['embedding'+str(index)]={k:item[k] for k in ('path','fingerprint','profile')}
    from .textual_inversion import validated_embeddings
    from .prompt_tokens import tokenizers
    validated_embeddings([(row,MODEL_ROOT/row['path']) for row in components.values()],family,tokenizers(family))
    return components


def preflight(settings):
    settings = Settings.model_validate(settings).model_dump()
    if settings['flow'] in ('img2img','inpaint'):
        if not settings['source_job'] or str(UUID(settings['source_job'])) != settings['source_job']:
            raise ValueError('변형할 이미지 작업을 선택하세요.')
        if int(settings['steps'] * settings['strength']) < 1:
            raise ValueError('Steps × 변형 강도가 1 이상이어야 합니다.')
    inventory = scan()
    model = next((m for m in inventory if m['id'] == settings['model']), None)
    if not model or not model['available']:
        raise ValueError(model['reason'] if model else '모델 파일을 찾을 수 없습니다.')
    if settings['embeddings'] and model.get('family') not in ('sd1_candidate','sd1_inpaint_candidate','sd2_candidate','sdxl_candidate'):
        raise ValueError('임베딩은 SD1/SD2/SDXL만 지원합니다.')
    if model.get('family')=='flux2':
        from .flux2 import execution_plan
        return execution_plan(model,settings,inventory)
    if model.get('family')=='zimage':
        from .z_image import execution_plan
        return execution_plan(model,settings,inventory)
    if settings['flow']=='reference_edit':raise ValueError('참고 이미지 편집은 FLUX.2 Klein 모델을 선택하세요.')
    if settings['ip_adapter']:
        if model.get('family')!='sdxl_candidate':raise ValueError('IP-Adapter는 현재 SDXL만 지원합니다.')
        if not settings['clip_vision']:raise ValueError('CLIP Vision 인코더를 선택하세요.')
        refs=[settings['reference_job'],settings['reference_attachment']]
        if sum(value is not None for value in refs)!=1:raise ValueError('생성 결과 또는 사용자 참고 이미지 하나를 선택하세요.')
        for value in refs:
            if value is not None and str(UUID(value))!=value:raise ValueError('잘못된 참고 이미지 ID')
    elif any(settings[key] for key in ('clip_vision','reference_job','reference_attachment')):
        raise ValueError('참고 이미지 조건을 쓰려면 IP-Adapter를 선택하세요.')
    if model.get('family')=='krea2':
        return krea2_plan(model, settings, inventory)
    family = 'anima' if model.get('family') == 'anima' else 'sd2' if model.get('family') == 'sd2_candidate' else 'sd1' if model.get('family') in ('sd1_candidate','sd1_inpaint_candidate') else 'sdxl'
    dedicated = model.get('family') == 'sd1_inpaint_candidate'
    if dedicated and settings['flow'] != 'inpaint':
        raise ValueError('9채널 전용 모델은 마스크가 있는 부분 수정에서만 사용할 수 있습니다.')
    if settings['flow']=='inpaint':
        if family=='anima':raise ValueError('ANIMA 인페인팅은 아직 미지원입니다.')
        if not settings['mask_id'] or str(UUID(settings['mask_id']))!=settings['mask_id']:raise ValueError('저장한 마스크를 선택하세요.')
    if (family=='anima') != (settings['prediction_type']=='flow'):
        raise ValueError('ANIMA는 flow, SD 계열은 epsilon/v_prediction을 선택하세요.')
    if family=='anima' and settings['prompt_mode']=='weighted':raise ValueError('ANIMA 가중치 인코딩은 아직 미지원입니다. 기본 모드를 선택하세요.')
    if model.get('anima',{}).get('variant')=='turbo' and settings['guidance']!=1:
        raise ValueError('ANIMA Turbo는 Guidance 1을 사용하세요. 권장 Steps는 8~12입니다.')
    config = MODEL_ROOT / 'configs' / (family+'-inpaint' if dedicated else family)
    for component in ('model_index.json', 'scheduler/scheduler_config.json', 'unet/config.json',
                      'vae/config.json', 'text_encoder/config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt') + (
                      ('text_encoder_2/config.json','tokenizer_2/vocab.json','tokenizer_2/merges.txt') if family == 'sdxl' else ()):
        if family != 'anima' and not (config / component).is_file():
            raise ValueError('로컬 '+family+' 구성 누락: ' + component)
    components = embedding_components(settings['embeddings'],inventory,family)
    if settings['ip_adapter']:
        for key in ('ip_adapter','clip_vision'):
            item=next((m for m in inventory if m['id']==settings[key] and m.get('component')==key and m['profile']['family']=='sdxl'),None)
            if not item:raise ValueError('SDXL 호환 구성요소를 선택하세요: '+key)
            components[key]={k:item[k] for k in ('path','fingerprint','profile')}
    for key,slot in (('clip',1),('clip2',2)):
        if not settings[key]:continue
        if family=='anima' or (slot==2 and family!='sdxl'):raise ValueError('이 계열은 선택한 CLIP 슬롯을 지원하지 않습니다.')
        encoder=next((m for m in inventory if m['id']==settings[key] and m.get('component')=='text_encoder'),None)
        if not encoder or encoder['profile']['family']!=family or encoder['profile']['slot']!=slot:
            raise ValueError('CLIP 계열/슬롯이 Checkpoint와 다릅니다.')
        components[key]={k:encoder[k] for k in ('path','fingerprint','profile')}
    if (settings['face_detector'] or settings['hand_detector']) and int(settings['detail_steps']*settings['detail_strength']) < 1:
        raise ValueError('디테일러 Steps × 강도가 1 이상이어야 합니다.')
    for target in ('face','hand'):
        key = target + '_detector'
        if settings[key]:
            detector = next((m for m in inventory if m['id'] == settings[key] and m.get('component') == 'detector'
                             and m['profile']['target'] == target),None)
            if not detector:
                raise ValueError(key + ': 대상에 맞는 검출기 프로필이 없습니다.')
            components[key] = {k:detector[k] for k in ('path','fingerprint','profile')}
    for kind in ('vae','upscaler'):
        if settings[kind]:
            component = next((m for m in inventory if m['id'] == settings[kind] and m.get('component') == kind), None)
            if component and kind == 'vae' and component['profile'].get('family','sdxl') != family:
                raise ValueError('VAE 계열이 Checkpoint와 다릅니다.')
            if not component:
                raise ValueError(f'{kind}: 호환되는 로컬 구성요소 프로필이 없습니다.')
            components[kind] = {key:component[key] for key in ('path','fingerprint','profile')}
    loras = []
    for selection in settings['loras']:
        if any(item['id'] == selection['model'] for item in loras):
            raise ValueError('같은 LoRA를 중복 선택할 수 없습니다.')
        model_lora = next((m for m in inventory if m['id'] == selection['model'] and m.get('component') == 'lora'), None)
        if not model_lora:
            raise ValueError('LoRA 호환 프로필이 없습니다.')
        if model_lora['profile'].get('family','sdxl') != family:
            raise ValueError('LoRA 계열이 Checkpoint와 다릅니다.')
        loras.append(dict(id=model_lora['id'], weight=selection['weight'],
                          **{key:model_lora[key] for key in ('path','fingerprint','profile')}))
    plan = dict(schema_version=1, runner=family + '_' + settings['flow'], model_path=model['path'],
                fingerprint=model['fingerprint'], settings=settings, components=components, loras=loras)
    if settings['ip_adapter']:
        vision_config=MODEL_ROOT/'configs/ip_adapter/clip_vision_h/config.json'
        if not vision_config.resolve().is_relative_to(MODEL_ROOT.resolve()):raise ValueError('인코더 설정 경로 이탈')
        plan['vision_config_sha256']=hashlib.sha256(vision_config.read_bytes()).hexdigest()
    if dedicated:plan['inpaint_channels']=9
    if 'anima' in model:plan.update(anima=model['anima'],bundle_fingerprint=model['bundle_fingerprint'])
    return plan


def krea2_plan(model, settings, inventory):
    if settings['flow']=='inpaint':
        if not settings['mask_id'] or str(UUID(settings['mask_id']))!=settings['mask_id']:
            raise ValueError('저장한 마스크를 선택하세요.')
    if settings['prediction_type']!='flow' or settings['guidance']!=0 or settings['prompt_mode']!='native':
        raise ValueError('Krea2 Turbo: flow · Guidance 0 · 기본 프롬프트 모드를 선택하세요. Negative는 사용하지 않습니다.')
    if settings['clip2']:
        raise ValueError('Krea2는 Qwen3-VL 인코더 하나를 사용합니다.')
    components={}
    if (settings['face_detector'] or settings['hand_detector']) and int(settings['detail_steps']*settings['detail_strength'])<1:
        raise ValueError('디테일러 Steps × 강도가 1 이상이어야 합니다.')
    for target in ('face','hand'):
        key=target+'_detector'
        if settings[key]:
            item=next((m for m in inventory if m['id']==settings[key] and m.get('component')=='detector' and m['profile']['target']==target),None)
            if not item:raise ValueError('대상에 맞는 검출기 프로필이 없습니다: '+key)
            components[key]={k:item[k] for k in ('path','fingerprint','profile')}
    for key,kind,default in (('clip','text_encoder','text_encoder'),('vae','vae','vae')):
        item=next((m for m in inventory if (m['id']==settings[key] if settings[key] else m['path']==model['krea2'][default])
                   and m.get('component')==kind and m['profile']['family']=='krea2'),None)
        if not item:raise ValueError('Krea2 호환 구성요소를 준비하세요: '+kind)
        components[key]={k:item[k] for k in ('path','fingerprint','profile')}
    if settings['upscaler']:
        item=next((m for m in inventory if m['id']==settings['upscaler'] and m.get('component')=='upscaler'),None)
        if not item:raise ValueError('호환 업스케일 프로필이 없습니다.')
        components['upscaler']={k:item[k] for k in ('path','fingerprint','profile')}
    loras=[]
    for selection in settings['loras']:
        if any(item['id']==selection['model'] for item in loras):raise ValueError('같은 LoRA를 중복 선택할 수 없습니다.')
        item=next((m for m in inventory if m['id']==selection['model'] and m.get('component')=='lora' and m['profile']['family']=='krea2'),None)
        if not item:raise ValueError('Krea2 계열의 LoRA 프로필을 선택하세요.')
        loras.append(dict(id=item['id'],weight=selection['weight'],**{k:item[k] for k in ('path','fingerprint','profile')}))
    manifest=MODEL_ROOT/'configs/krea2/manifest.json'
    if not manifest.resolve().is_relative_to(MODEL_ROOT.resolve()):raise ValueError('설정 경로가 모델 루트 밖입니다.')
    from .krea2 import verify_config
    verify_config(manifest.parent)
    return dict(schema_version=1,runner='krea2_'+settings['flow'],model_path=model['path'],fingerprint=model['fingerprint'],
                settings=settings,components=components,loras=loras,krea2=model['krea2'],
                config_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
