"""Export an owned checkpoint's standard CLIP weights without changing the original."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint');parser.add_argument('output');parser.add_argument('--family',choices=['sd1','sd2','sdxl'],required=True)
    parser.add_argument('--slot',type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    if args.slot==2 and args.family!='sdxl':raise ValueError('Slot2 requires SDXL')
    root=(Path(__file__).resolve().parents[1]/'models').resolve()
    source=(root/args.checkpoint).resolve();target=(root/args.output).resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root) or target.parent.name not in ('clip','text_encoder','text_encoders') or target.suffix!='.safetensors':raise ValueError('Use models paths and a text_encoder output folder')
    profile=target.with_name(target.name+'.profile.json')
    if target.exists() or profile.exists():raise ValueError('Output already exists')
    import torch
    import diffusers
    from safetensors.torch import save_file
    cls=diffusers.StableDiffusionXLPipeline if args.family=='sdxl' else diffusers.StableDiffusionPipeline
    extra={} if args.family=='sdxl' else dict(safety_checker=None,feature_extractor=None,requires_safety_checker=False)
    pipe=cls.from_single_file(str(source),config=str(root/'configs'/args.family),local_files_only=True,torch_dtype=torch.float16,**extra)
    encoder=pipe.text_encoder if args.slot==1 else pipe.text_encoder_2
    for key,value in encoder.state_dict().items():
        if not torch.isfinite(value).all():raise ValueError('Source CLIP contains nonfinite weights: '+key)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.preparing')
    save_file({key:value.contiguous() for key,value in encoder.state_dict().items()},str(temporary))
    with source.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    with temporary.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    temporary.rename(target)
    record=dict(schema_version=1,kind='text_encoder',family=args.family,slot=args.slot,scale=1,sha256=sha,
                source=f'local:{args.checkpoint} sha256:{source_hash}; diffusers:{diffusers.__version__}')
    profile.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
