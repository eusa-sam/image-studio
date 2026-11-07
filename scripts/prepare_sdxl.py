"""Explicit setup download; inference never calls this script."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'stabilityai/stable-diffusion-xl-base-1.0'


def main():
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}'))
    revision = info['sha']
    folder = ROOT / 'models/configs/sdxl'
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for entry in info['siblings']:
        name = entry['rfilename']
        if name.endswith(('.json', '.txt')) and not name.startswith('.'):
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix(target.suffix + '.download')
                urllib.request.urlretrieve(f'https://huggingface.co/{REPO}/resolve/{revision}/{name}', temporary)
                temporary.replace(target)
            files.append(name)
    (folder / 'download_manifest.json').write_text(json.dumps(dict(repository=REPO, revision=revision, files=files), indent=2), encoding='utf-8')
    assert (folder / 'tokenizer/merges.txt').exists()
    assert (folder / 'tokenizer_2/tokenizer_config.json').exists()
    print('Prepared local SDXL configs:', len(files), revision)


if __name__ == '__main__':
    main()
