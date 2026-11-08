"""Strict replacement of standard CLIP safetensors before LoRA and offloading."""


def replace_encoder(encoder,path):
    import torch
    from safetensors.torch import load_file
    weights=load_file(str(path),device='cpu')
    expected=encoder.state_dict()
    converted={}
    for key,value in weights.items():
        target=key
        if target not in expected and target.startswith('text_model.') and target[len('text_model.'):] in expected:
            target=target[len('text_model.'):]
        if target in converted:raise ValueError('Duplicate CLIP key')
        if target not in expected or value.shape!=expected[target].shape:raise ValueError('CLIP key/shape mismatch: '+key)
        if value.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(value).all():
            raise ValueError('Unsupported or nonfinite CLIP weights')
        if value.dtype!=expected[target].dtype:
            value=value.to(dtype=expected[target].dtype)
            if not torch.isfinite(value).all():
                raise ValueError('CLIP weights overflow target precision: '+key)
        converted[target]=value
    if converted.keys()!=expected.keys():raise ValueError('Incomplete CLIP weights')
    encoder.load_state_dict(converted,strict=True)
