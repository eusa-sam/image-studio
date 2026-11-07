"""Prepare the pinned official Klein 4B bundle; inference stays offline."""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .downloads import hf_hub_download
else:
    from downloads import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((Path(__file__).parent/'manifests/flux2-klein-4b.json').read_text(encoding='utf-8'))


def verify(path, record):
    if path.stat().st_size != record['size']:
        raise ValueError('File size mismatch: '+record['path'])
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {record['size']}\0".encode())
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            sha.update(block)
            if not record['sha256']:blob.update(block)
    digest = sha.hexdigest()
    if (digest != record['sha256'] if record['sha256'] else blob.hexdigest() != record['blob_id']):
        raise ValueError('File hash mismatch: '+record['path'])
    return digest


def main(*, spec=None, slug='flux2-klein-4b'):
    spec = SPEC if spec is None else spec
    if not slug or Path(slug).name != slug or slug in ('.','..'):
        raise ValueError('Invalid bundle directory')
    # Keep incomplete downloads outside inventory until the runner is connected.
    folder = ROOT/'models/checkpoint'/slug
    if not folder.exists():folder = ROOT/'models/sources'/slug
    records = []
    for record in spec['files']:
        name = record['path']
        target = (folder/name).resolve()
        if not target.is_relative_to(folder.resolve()):raise ValueError('Invalid bundle path')
        if target.exists():
            digest = verify(target,record)
        else:
            source = Path(hf_hub_download(spec['repository'],name,revision=spec['revision'],token=False))
            digest = verify(source,record)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent,delete=False,suffix='.download') as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(source,temporary)
                verify(temporary,record)
                # A hard link publishes atomically without replacing another file.
                os.link(temporary,target)
            finally:
                temporary.unlink(missing_ok=True)
        records.append(dict(path=name,sha256=digest))
        print('Verified',name,flush=True)
    manifest = folder/'download_manifest.json'
    content = json.dumps(dict(repository=spec['repository'],revision=spec['revision'],files=records),indent=2)
    if manifest.exists():
        if manifest.read_text(encoding='utf-8') != content:raise ValueError('Existing manifest differs')
    else:
        with manifest.open('x',encoding='utf-8') as stream:stream.write(content)


if __name__ == '__main__':main()
