"""Explicit, pinned SD1 setup; inference never downloads files."""
import hashlib
import json
import shutil
from pathlib import Path
if __package__:
    from .downloads import hf_hub_download
else:
    from downloads import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(repo=REPO, revision=REVISION, family='sd1', checkpoint='v1-5-pruned-emaonly.safetensors'):
    configs = ['model_index.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json',
               'unet/config.json', 'vae/config.json', 'tokenizer/merges.txt', 'tokenizer/vocab.json',
               'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json']
    records = []
    for name in configs + [checkpoint]:
        source = Path(hf_hub_download(repo, name, revision=revision, token=False))
        target = ROOT / 'models' / ('configs/'+family if name in configs else 'checkpoint') / name
        sha = checksum(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if checksum(target) != sha:
                raise ValueError('Existing file differs: ' + str(target))
        else:
            temporary = target.with_suffix(target.suffix + '.download')
            shutil.copyfile(source, temporary)
            assert checksum(temporary) == sha
            temporary.replace(target)
        records.append(dict(path=target.relative_to(ROOT).as_posix(), sha256=sha))
        print(name, sha, flush=True)
    manifest = ROOT / ('models/configs/'+family+'/download_manifest.json')
    content = json.dumps(dict(repository=repo, revision=revision, files=records), indent=2)
    if manifest.exists() and manifest.read_text(encoding='utf-8') != content:
        raise ValueError('Existing manifest differs')
    manifest.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
