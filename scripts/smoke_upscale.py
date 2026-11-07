"""Offline learned 2x upscale on a real generated image."""
import json
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from spandrel import ModelLoader

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = sorted((ROOT / 'data/validation/sdxl/waiIllustriousSDXL_v160').glob('*/txt2img.png'))[0]
    target = ROOT / 'data/validation/upscale' / time.strftime('%Y%m%d-%H%M%S')
    target.mkdir(parents=True)
    state = torch.load(ROOT / 'models/upscale_models/BSRGANx2.pth', map_location='cpu', weights_only=True)
    model = ModelLoader().load_from_state_dict(state).eval().cuda()
    source_image = Image.open(source).convert('RGB')
    tensor = torch.from_numpy(np.asarray(source_image).copy()).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
    start = time.monotonic()
    with torch.inference_mode():
        result = model(tensor)
    assert torch.isfinite(result).all()
    assert result.shape[-2:] == (source_image.height * 2, source_image.width * 2)
    output = result.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray((output * 255).round().astype('uint8')).save(target / 'upscaled.png')
    (target / 'report.json').write_text(json.dumps(dict(status='passed', source=str(source), model='BSRGANx2', scale=model.scale, seconds=time.monotonic()-start), indent=2))
    print(target)


if __name__ == '__main__':
    main()
