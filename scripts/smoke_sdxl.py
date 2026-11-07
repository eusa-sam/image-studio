"""Real offline SDXL generation; one model per process to release GPU memory."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import argparse
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--vae')
    args = parser.parse_args()
    import torch
    import numpy as np
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
    path = ROOT / 'models/checkpoint' / args.model
    destination = ROOT / 'data/validation/sdxl' / path.stem / time.strftime('%Y%m%d-%H%M%S')
    destination.mkdir(parents=True)
    report = dict(model=str(path), steps=args.steps, size=args.size, torch=torch.__version__, tests=[])
    start = time.monotonic()
    try:
        assert torch.cuda.is_available(), 'CUDA unavailable'
        pipe = StableDiffusionXLPipeline.from_single_file(str(path), config=str(ROOT / 'models/configs/sdxl'), local_files_only=True, torch_dtype=torch.float16)
        if args.vae:
            pipe.vae = AutoencoderKL.from_single_file(str(ROOT / 'models/vae' / args.vae), config=str(ROOT / 'models/configs/sdxl/vae'), local_files_only=True, torch_dtype=torch.float16)
            report['external_vae'] = args.vae
        report['prediction_type'] = pipe.scheduler.config.prediction_type
        report['prediction_source'] = 'base SDXL config; checkpoint-specific semantics not yet verified'
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        prompt = 'a red ceramic teapot on a wooden table, flowers, soft daylight, detailed illustration'
        image = pipe(prompt=prompt, negative_prompt='blurry, low quality', height=args.size, width=args.size, num_inference_steps=args.steps, generator=torch.Generator('cpu').manual_seed(42)).images[0]
        image.save(destination / 'txt2img.png')
        pixels = np.asarray(image)
        assert pixels.std() > 2, 'Flat output'
        report['tests'].append(dict(flow='txt2img', status='passed', std=float(pixels.std())))
        edit = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
        result = edit(prompt=prompt + ', blue ceramic glaze', image=image, strength=0.5, num_inference_steps=args.steps, generator=torch.Generator('cpu').manual_seed(43)).images[0]
        result.save(destination / 'img2img.png')
        assert np.asarray(result).std() > 2
        report['tests'].append(dict(flow='img2img', status='passed'))
        from PIL import Image, ImageDraw, ImageFilter
        box = (args.size // 4, args.size // 4, args.size * 3 // 4, args.size * 3 // 4)
        crop = image.crop(box).resize((args.size, args.size))
        corrected = edit(prompt=prompt, image=crop, strength=0.5, num_inference_steps=args.steps, generator=torch.Generator('cpu').manual_seed(44)).images[0]
        corrected = corrected.resize((box[2]-box[0], box[3]-box[1]))
        mask = Image.new('L', corrected.size)
        ImageDraw.Draw(mask).ellipse((8, 8, corrected.width-8, corrected.height-8), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(4))
        combined = image.copy()
        combined.paste(corrected, box, mask)
        combined.save(destination / 'manual_crop_detail.png')
        assert np.array_equal(np.asarray(combined)[:box[1]], pixels[:box[1]])
        report['tests'].append(dict(flow='manual_crop_detail_no_detector', status='passed'))
        report['status'] = 'smoke_passed_quality_not_certified'
        report['peak_vram_bytes'] = torch.cuda.max_memory_allocated()
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        print(report['error'])
    finally:
        report['seconds'] = time.monotonic() - start
        (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(destination, report['status'], flush=True)
    return 0 if report['status'].startswith('smoke_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
