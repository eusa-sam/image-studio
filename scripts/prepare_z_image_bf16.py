"""Prepare verified BF16 weights offline, preserving the official source bundle."""
import json
import os
import tempfile
from pathlib import Path

from .prepare_flux2_klein import ROOT,verify

SPEC=json.loads((Path(__file__).parent/'manifests/z-image-turbo-bf16.json').read_text(encoding='utf-8'))


def convert(source,target):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(str(source),framework='pt',device='cpu') as reader:
        tensors={key:reader.get_tensor(key).to(torch.bfloat16) for key in reader.keys()}
        save_file(tensors,str(target),metadata=reader.metadata())


def prepare(source,folder,spec=SPEC):
    from backend.app.inference.models.z_image import verify_bundle
    source=Path(source).resolve();folder=Path(folder).resolve()
    if source==folder:raise ValueError('BF16 output must not replace source')
    records=[];source_verified=False
    for row in spec['files']:
        name=row['path'];target=(folder/name).resolve();original=(source/name).resolve()
        if not target.is_relative_to(folder) or not original.is_relative_to(source):raise ValueError('Invalid bundle path')
        if target.exists():digest=verify(target,row)
        else:
            if not source_verified:
                verify_bundle(source);source_verified=True
            target.parent.mkdir(parents=True,exist_ok=True)
            if name.startswith('transformer/') and target.suffix=='.safetensors':
                with tempfile.NamedTemporaryFile(dir=target.parent,delete=False,suffix='.prepare') as stream:temporary=Path(stream.name)
                try:
                    convert(original,temporary)
                    digest=verify(temporary,row)
                    os.link(temporary,target)
                finally:temporary.unlink(missing_ok=True)
            elif name=='transformer/diffusion_pytorch_model.safetensors.index.json':
                value=json.loads(original.read_text(encoding='utf-8'));value['metadata']['total_size']//=2
                with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=target.parent,delete=False,suffix='.prepare') as stream:
                    temporary=Path(stream.name);stream.write(json.dumps(value,indent=2))
                try:
                    digest=verify(temporary,row);os.link(temporary,target)
                finally:temporary.unlink(missing_ok=True)
            else:
                digest=verify(original,row);os.link(original,target)
        records.append(dict(path=name,sha256=digest))
        print('Verified',name,flush=True)
    content=json.dumps(dict(repository=spec['repository'],revision=spec['revision'],variant=spec['variant'],files=records),indent=2)
    manifest=folder/'download_manifest.json'
    if manifest.exists():
        if manifest.read_text(encoding='utf-8')!=content:raise ValueError('Existing manifest differs')
    else:
        with manifest.open('x',encoding='utf-8') as stream:stream.write(content)


def main():
    import torch
    torch.set_num_threads(4)
    folder=ROOT/'models/checkpoint/z-image-turbo-bf16'
    if not folder.exists():folder=ROOT/'models/sources/z-image-turbo-bf16'
    prepare(ROOT/'models/checkpoint/z-image-turbo',folder)


if __name__=='__main__':main()
