"""Read safetensors headers without loading weights or executing pickle."""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect(path, root=ROOT):
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Missing header length')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(64 * 1024 * 1024, size - 8):
            raise ValueError('Invalid header length')
        header = json.loads(stream.read(length))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    for value in tensors.values():
        start, end = value['data_offsets']
        if not 0 <= start <= end <= size - length - 8:
            raise ValueError('Tensor offset outside file')
    metadata = header.get('__metadata__', {})
    keys = list(tensors)
    family = 'unresolved'
    if metadata.get('architecture') == 'krea2':
        family = 'krea2'
    elif (tensors.get('first.weight',{}).get('shape')==[6144,64]
          and tensors.get('txtfusion.projector.weight',{}).get('shape')==[1,12]
          and tensors.get('blocks.0.attn.wk.weight',{}).get('shape')==[1536,6144]
          and tensors.get('blocks.27.attn.wq.weight',{}).get('shape')==[6144,6144]):
        family = 'krea2_candidate'
    elif any(k.endswith('llm_adapter.embed.weight') and v.get('shape')==[32128,1024] for k,v in tensors.items()) and any(k.endswith('blocks.0.cross_attn.k_proj.weight') and v.get('shape')==[2048,1024] for k,v in tensors.items()):
        family = 'anima_candidate'
    elif (tensors.get('conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight',{}).get('shape')==[49408,768]
          and tensors.get('conditioner.embedders.1.model.token_embedding.weight',{}).get('shape')==[49408,1280]
          and tensors.get('model.diffusion_model.input_blocks.0.0.weight',{}).get('shape')==[320,4,3,3]
          and any(k.endswith('attn2.to_k.weight') and v.get('shape')==[640,2048] for k,v in tensors.items())):
        family = 'sdxl_candidate'
    elif (tensors.get('cond_stage_model.model.token_embedding.weight', {}).get('shape') == [49408,1024]
          and tensors.get('model.diffusion_model.input_blocks.0.0.weight', {}).get('shape') == [320,4,3,3]
          and any(k.endswith('attn2.to_k.weight') and v.get('shape') == [320,1024] for k,v in tensors.items())):
        family = 'sd2_candidate'
    elif (tensors.get('cond_stage_model.transformer.text_model.embeddings.token_embedding.weight', {}).get('shape') == [49408, 768]
          and tensors.get('model.diffusion_model.input_blocks.0.0.weight', {}).get('shape') in ([320, 4, 3, 3], [320, 9, 3, 3])
          and any(k.endswith('attn2.to_k.weight') and v.get('shape') == [320, 768] for k,v in tensors.items())):
        family = 'sd1_candidate' if tensors['model.diffusion_model.input_blocks.0.0.weight']['shape'][1] == 4 else 'sd1_inpaint_candidate'
    elif metadata.get('ss_network_module') == 'networks.lora_krea2':
        family = 'krea2_lora'
    elif any(k.startswith('lora_') for k in keys):
        family = 'lora_unresolved'
    elif any(k.startswith('vision_model.') for k in keys):
        family = 'vision_encoder'
    return dict(schema_version=1, path=str(path.relative_to(root)), bytes=size,
                tensor_count=len(tensors), dtypes=sorted({v['dtype'] for v in tensors.values()}),
                family=family, metadata=metadata, sample_keys=keys[:10],
                status='header_checked_not_inference_verified')


def main():
    destination = ROOT / 'data/models/inventory'
    destination.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'models').rglob('*.safetensors'):
        result = inspect(path)
        # Preserve directory identity when filenames repeat.
        import hashlib
        identifier = hashlib.sha256(result['path'].encode()).hexdigest()[:24]
        target = destination / (identifier + '.json')
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(target)
        print(path.name, result['family'], result['tensor_count'])


if __name__ == '__main__':
    main()
