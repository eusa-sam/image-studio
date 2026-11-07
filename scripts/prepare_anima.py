"""Prepare the pinned official Anima Base bundle without inference downloads."""
import hashlib
import json
import shutil
from pathlib import Path
if __package__:
    from .downloads import hf_hub_download
else:
    from downloads import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'circlestone-labs/Anima-Base-v1.0-Diffusers'
REVISION = '073c3a9db359c31ad0e8aa268d15775473c2176c'
FILES = ['LICENSE.md','README.md','modular_model_index.json','scheduler/scheduler_config.json',
         't5_tokenizer/tokenizer.json','t5_tokenizer/tokenizer_config.json','tokenizer/chat_template.jinja',
         'tokenizer/tokenizer.json','tokenizer/tokenizer_config.json']
for component in ('text_conditioner','text_encoder','transformer','vae'):
    FILES += [component+'/config.json',component+('/model.safetensors' if component=='text_encoder' else '/diffusion_pytorch_model.safetensors')]


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    folder=ROOT/'models/checkpoint/anima-base-v1.0'
    records=[]
    for name in FILES:
        source=Path(hf_hub_download(REPO,name,revision=REVISION,token=False))
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
        sha=checksum(source)
        if target.exists():
            if checksum(target)!=sha:raise ValueError('Existing file differs: '+str(target))
        else:
            temporary=target.with_suffix(target.suffix+'.download')
            shutil.copyfile(source,temporary)
            assert checksum(temporary)==sha
            temporary.replace(target)
        records.append(dict(path=name,sha256=sha))
        print(name,sha,flush=True)
    content=json.dumps(dict(repository=REPO,revision=REVISION,files=records),indent=2)
    manifest=folder/'download_manifest.json'
    if manifest.exists() and manifest.read_text(encoding='utf-8')!=content:raise ValueError('Existing manifest differs')
    manifest.write_text(content,encoding='utf-8')


if __name__=='__main__':main()
