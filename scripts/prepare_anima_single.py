"""Download an official single-file Anima variant with its explicit local profile."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
if __package__:
    from .downloads import hf_hub_download
else:
    from downloads import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
REPO='circlestone-labs/Anima'
REVISION='f973fc41ec7545364ac9776c2440285f43ff2a30'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['base','turbo'],default='turbo')
    args=parser.parse_args()
    version='1.0' if args.variant=='base' else '1.1'
    name=f'anima-{args.variant}-v{version}.safetensors'
    source=Path(hf_hub_download(REPO,'split_files/diffusion_models/'+name,revision=REVISION,token=False))
    with source.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    target=ROOT/'models/checkpoint'/name
    if target.exists():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:raise ValueError('Existing model differs')
    else:
        temporary=target.with_suffix('.download');shutil.copyfile(source,temporary)
        with temporary.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
        temporary.replace(target)
    profile=dict(schema_version=1,variant=args.variant,sha256=sha,bundle='checkpoint/anima-base-v1.0',
                 source=f'https://huggingface.co/{REPO}/blob/{REVISION}/split_files/diffusion_models/{name}')
    destination=target.with_name(target.name+'.anima.json');content=json.dumps(profile,indent=2)
    if destination.exists() and destination.read_text(encoding='utf-8')!=content:raise ValueError('Existing profile differs')
    destination.write_text(content,encoding='utf-8');print(name,sha,flush=True)


if __name__=='__main__':main()
