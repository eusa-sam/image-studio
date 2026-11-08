"""Strict local CLIP Vision loading for image-conditioned generation."""

def load_vision(path, config_path):
    import torch
    from safetensors import safe_open
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    config=CLIPVisionConfig.from_json_file(config_path)
    with torch.device('meta'):
        encoder=CLIPVisionModelWithProjection(config)
    count=(config.image_size//config.patch_size)**2+1
    encoder.vision_model.embeddings.position_ids=torch.arange(count).view(1,-1)
    expected=encoder.state_dict()
    with safe_open(str(path),framework='pt',device='cpu') as reader:
        keys=set(reader.keys())
        position='vision_model.embeddings.position_ids'
        if position in keys:
            ids=reader.get_tensor(position)
            if ids.dtype not in (torch.int32,torch.int64) or ids.shape!=(1,count) or not torch.equal(ids,torch.arange(count).view(1,-1)):
                raise ValueError('Invalid CLIP Vision position IDs')
            keys.remove(position)
        if keys!=expected.keys():raise ValueError('CLIP Vision key mismatch')
        for key in sorted(keys):
            value=reader.get_tensor(key)
            if value.shape!=expected[key].shape:raise ValueError('CLIP Vision shape mismatch: '+key)
            if value.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(value).all():
                raise ValueError('Unsupported or nonfinite CLIP Vision weights')
            value=value.to(dtype=torch.float16,copy=True)
            if not torch.isfinite(value).all():raise ValueError('CLIP Vision conversion overflow')
            encoder.load_state_dict({key:value},strict=False,assign=True)
    return encoder.eval()


def attach_adapter(pipe, adapter_path, vision_path, config_path, scale):
    import math
    import torch
    from safetensors.torch import load_file
    if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or not 0<=scale<=2:
        raise ValueError('IP-Adapter strength must be between 0 and 2')
    if pipe.unet.config.cross_attention_dim!=2048:
        raise ValueError('Only the SDXL standard ViT-H IP-Adapter is supported')
    vision=load_vision(vision_path,config_path)
    expected={'image_proj.proj.weight':(8192,vision.config.projection_dim),'image_proj.proj.bias':(8192,),
              'image_proj.norm.weight':(2048,),'image_proj.norm.bias':(2048,)}
    index=1
    for name in pipe.unet.attn_processors:
        if name.endswith('attn1.processor'):continue
        attention=pipe.unet.get_submodule(name.removesuffix('.processor'))
        for part in ('k','v'):
            expected[f'ip_adapter.{index}.to_{part}_ip.weight']=tuple(getattr(attention,'to_'+part).weight.shape)
        index+=2
    weights=load_file(str(adapter_path),device='cpu')
    if weights.keys()!=expected.keys():raise ValueError('IP-Adapter key mismatch')
    parts={'image_proj':{},'ip_adapter':{}}
    for key,value in weights.items():
        if tuple(value.shape)!=expected[key]:raise ValueError('IP-Adapter shape mismatch: '+key)
        if value.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(value).all():
            raise ValueError('Unsupported or nonfinite IP-Adapter weights')
        if value.abs().max()>torch.finfo(pipe.unet.dtype).max:
            raise ValueError('IP-Adapter conversion overflow')
        group,name=key.split('.',1);parts[group][name]=value
    pipe.register_modules(image_encoder=vision)
    pipe.load_ip_adapter(parts,subfolder='',weight_name='',image_encoder_folder=None,local_files_only=True)
    pipe.set_ip_adapter_scale(scale)


def reference_image(store,session,settings):
    from backend.app.storage import digest
    if settings['reference_attachment']:
        from backend.app.references import read_reference
        path,meta=read_reference(store,session,settings['reference_attachment'])
        if settings.get('flow')=='reference_edit':
            return path.parent/meta['original'],dict(kind='user_reference',id=meta['id'],sha256=meta['sha256'],original_sha256=meta['sha256'],representation='original')
        return path,dict(kind='user_reference',id=meta['id'],sha256=meta['preview_sha256'],original_sha256=meta['sha256'])
    job=settings['reference_job']
    artifact=settings.get('reference_artifact','image.png')
    if artifact not in ('image.png','original.png'):raise ValueError('Invalid reference artifact')
    if artifact=='image.png' and store.job(session,job)['state']['status']!='SUCCEEDED':raise ValueError('완료된 생성 결과를 참고 이미지로 선택하세요.')
    path=store.artifact(session,job,artifact)
    return path,dict(kind='generated_output',id=job,artifact=artifact,sha256=digest(path.read_bytes()))
