"""Prepare public HandRefiner ControlNet weights; this does not install the MANO pipeline."""
import hashlib
import json
from pathlib import Path
try:
    from .downloads import hf_hub_download
except ImportError:
    from downloads import hf_hub_download


def main():
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'scripts/manifests/handrefiner-controlnet.json').read_text(encoding='utf8'))
    target=root/'models/sources/handrefiner'/spec['revision']
    target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in spec['files']:
        if item['path'] not in ('README.md','control_sd15_inpaint_depth_hand_fp16.safetensors'):continue
        path=target/item['path']
        if not path.exists():
            path=Path(hf_hub_download(spec['repository'],item['path'],revision=spec['revision'],local_dir=target))
        with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if path.stat().st_size!=item['size'] or (item['sha256'] and checksum!=item['sha256']):
            raise ValueError('Downloaded file does not match pinned manifest: '+item['path'])
        rows.append(dict(path=item['path'],bytes=path.stat().st_size,sha256=checksum))
    assert len(rows)==2
    from safetensors import safe_open
    weights=target/'control_sd15_inpaint_depth_hand_fp16.safetensors'
    with safe_open(weights,framework='pt',device='cpu') as reader:
        shapes={key:list(reader.get_slice(key).get_shape()) for key in reader.keys()}
    report=dict(repository=spec['repository'],revision=spec['revision'],license=spec['license'],files=rows,
                tensors=len(shapes),shapes=shapes,status='SOURCE_PREPARED_NOT_RUNTIME_SUPPORTED')
    (target/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(target=str(target),tensors=len(shapes),status=report['status'])),flush=True)

if __name__=='__main__':main()
