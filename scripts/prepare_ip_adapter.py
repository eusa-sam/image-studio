"""Prepare public, pinned SDXL ViT-H IP-Adapter assets; no inference downloads."""
import json
import shutil
from pathlib import Path
if __package__:
    from .downloads import hf_hub_download
else:
    from downloads import hf_hub_download
from scripts.prepare_sd1 import checksum

ROOT=Path(__file__).resolve().parents[1]
REPO='h94/IP-Adapter'
REVISION='018e402774aeeddd60609b4ecdb7e298259dc729'
FILES=[
 ('models/image_encoder/config.json','configs/ip_adapter/clip_vision_h/config.json','625d37b31afbf2f0792a87846b3654ee23f20568409e35b78a1f795b04e1a7a1'),
 ('sdxl_models/ip-adapter_sdxl_vit-h.safetensors','ip_adapter/ip-adapter_sdxl_vit-h.safetensors','ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831'),
]


def main():
    records=[]
    for name,destination,expected in FILES:
        target=ROOT/'models'/destination
        if target.exists():
            if checksum(target)!=expected:raise ValueError('Existing preparation file differs: '+destination)
        else:
            source=Path(hf_hub_download(REPO,name,revision=REVISION,token=False))
            if checksum(source)!=expected:raise ValueError('Downloaded file hash mismatch: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            if checksum(target)!=expected:raise ValueError('Copied file hash mismatch: '+destination)
        records.append(dict(path=destination,sha256=expected,url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'))
        print('Verified',destination,flush=True)
    manifest=ROOT/'models/configs/ip_adapter/download_manifest.json'
    content=json.dumps(dict(schema_version=1,repository=REPO,revision=REVISION,files=records),indent=2)
    if manifest.exists():
        if manifest.read_text(encoding='utf-8')!=content:raise ValueError('Existing manifest differs')
    else:
        with manifest.open('x',encoding='utf-8') as stream:stream.write(content)

    for relative,kind in [('ip_adapter/ip-adapter_sdxl_vit-h.safetensors','ip_adapter'),('clip_vision/clip_vision_h.safetensors','clip_vision')]:
        path=ROOT/'models'/relative
        if not path.exists():continue
        if kind=='clip_vision':
            from backend.app.ip_adapter import load_vision
            encoder=load_vision(path,ROOT/'models/configs/ip_adapter/clip_vision_h/config.json')
            del encoder
        profile=path.with_name(path.name+'.profile.json')
        data=dict(schema_version=1,kind=kind,family='sdxl',scale=1,sha256=checksum(path),
                  source=f'{REPO}@{REVISION}; standard SDXL ViT-H adapter' if kind=='ip_adapter' else 'Local ViT-H; full CLIP Vision weights validated against pinned IP-Adapter config')
        content=json.dumps(data,indent=2)
        if profile.exists():
            if profile.read_text(encoding='utf-8')!=content:raise ValueError('Existing profile differs')
        else:
            with profile.open('x',encoding='utf-8') as stream:stream.write(content)


if __name__=='__main__':main()
