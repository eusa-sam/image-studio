"""Convert local Kohya Krea2 adapters to the installed Diffusers key format."""
import torch


def convert_lora(state):
    from diffusers.loaders import Krea2LoraLoaderMixin
    for key,value in state.items():
        if any(tag in key.lower() for tag in ('dora','hada','lokr')):
            raise ValueError('Unsupported Krea2 adapter format: '+key)
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Invalid Krea2 LoRA tensor: '+key)
    if not any(key.startswith('lora_unet_') for key in state):
        return Krea2LoraLoaderMixin.lora_state_dict(state)
    groups={}
    for key,value in state.items():
        name,separator,suffix=key.partition('.')
        if not separator or not name.startswith('lora_unet_') or suffix not in ('alpha','lora_down.weight','lora_up.weight'):
            raise ValueError('Unsupported Krea2 LoRA key: '+key)
        groups.setdefault(name,{})[suffix]=value
    converted={}
    for name,weights in groups.items():
        down,up=weights.get('lora_down.weight'),weights.get('lora_up.weight')
        if down is None or up is None or down.ndim!=2 or up.ndim!=2 or down.shape[0]<1 or down.shape[0]!=up.shape[1]:
            raise ValueError('Invalid Krea2 LoRA A/B pair: '+name)
        rank=down.shape[0]
        alpha=weights.get('alpha',torch.tensor(float(rank)))
        if alpha.numel()!=1 or alpha.item()<0:raise ValueError('Invalid Krea2 LoRA alpha: '+name)
        module=name.removeprefix('lora_unet_').replace('_','.')
        module=module.replace('layerwise.blocks','layerwise_blocks').replace('refiner.blocks','refiner_blocks')
        prefix='diffusion_model.'+module
        converted[prefix+'.lora_A.weight']=down
        converted[prefix+'.lora_B.weight']=up.float()*(alpha.item()/rank)
        if not torch.isfinite(converted[prefix+'.lora_B.weight']).all():
            raise ValueError('Krea2 LoRA scale overflow: '+name)
    return Krea2LoraLoaderMixin.lora_state_dict(converted)
