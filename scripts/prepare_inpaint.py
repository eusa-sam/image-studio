"""Pinned SD1 inpaint setup; restricted checkpoint conversion happens only here."""
import json
from pathlib import Path
from scripts.prepare_sd1 import main as prepare, checksum, ROOT

REPO = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
REVISION = '8a4288a76071f7280aedbdb3253bdb9e9d5d84bb'


def main():
    prepare(REPO, REVISION, 'sd1-inpaint', 'sd-v1-5-inpainting.ckpt')
    import torch
    from safetensors.torch import save_file
    source = ROOT/'models/checkpoint/sd-v1-5-inpainting.ckpt'
    target = source.with_suffix('.safetensors')
    manifest = target.with_name(target.name+'.source.json')
    provenance = dict(repository=REPO, revision=REVISION, source_sha256=checksum(source),
                      conversion='torch.load(weights_only=True), state_dict to safetensors')
    if target.exists():
        existing=json.loads(manifest.read_text(encoding='utf-8'))
        if existing != dict(provenance, sha256=checksum(target)):
            raise ValueError('Existing inpaint export differs')
        return
    state = torch.load(source, map_location='cpu', weights_only=True)['state_dict']
    if state['model.diffusion_model.input_blocks.0.0.weight'].shape != (320,9,3,3):
        raise ValueError('Unexpected inpaint input channels')
    state={key:value.contiguous() for key,value in state.items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Non-finite checkpoint')
    temporary=target.with_suffix('.safetensors.preparing')
    save_file(state,str(temporary))
    temporary.rename(target)
    manifest.write_text(json.dumps(dict(provenance,sha256=checksum(target)),indent=2),encoding='utf-8')
    print(target,flush=True)


if __name__ == '__main__':
    main()
