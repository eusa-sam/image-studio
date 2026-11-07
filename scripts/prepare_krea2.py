"""Download public, revision-pinned Krea2 supporting configs/tokenizer only."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'models/configs/krea2'
SOURCES = [
    ('Qwen/Qwen3-VL-4B-Instruct','ebb281ec70b05090aa6165b016eac8ec08e71b17','config.json','text_encoder/config.json'),
    ('Qwen/Qwen3-VL-4B-Instruct','ebb281ec70b05090aa6165b016eac8ec08e71b17','tokenizer.json','tokenizer/tokenizer.json'),
    ('Qwen/Qwen3-VL-4B-Instruct','ebb281ec70b05090aa6165b016eac8ec08e71b17','tokenizer_config.json','tokenizer/tokenizer_config.json'),
    ('Qwen/Qwen-Image','75e0b4be04f60ec59a75f475837eced720f823b6','vae/config.json','vae/config.json'),
]


def main():
    manifest=ROOT/'manifest.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text(encoding='utf-8'))['files']:
            if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Prepared config changed: '+row['path'])
        print('Verified existing Krea2 supporting files');return
    rows=[]
    for repo,revision,source,destination in SOURCES:
        url=f'https://huggingface.co/{repo}/resolve/{revision}/{source}'
        with urllib.request.urlopen(url,timeout=120) as response:content=response.read(32*1024*1024+1)
        if len(content)>32*1024*1024:raise ValueError('Config/tokenizer too large')
        json.loads(content)
        target=ROOT/destination;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=content:raise ValueError('Existing preparation file differs')
        else:
            with target.open('xb') as stream:stream.write(content)
        rows.append(dict(path=destination,repo=repo,revision=revision,url=url,sha256=hashlib.sha256(content).hexdigest()))
    with manifest.open('x',encoding='utf-8') as stream:json.dump(dict(schema_version=1,files=rows),stream,indent=2)
    print('Prepared',len(rows),'files')


if __name__=='__main__':main()
