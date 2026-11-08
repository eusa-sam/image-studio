"""Convert local Kohya Krea2 adapters to the installed Diffusers key format."""
import torch


def skip_unit_scaling(model):
    from peft.tuners.lora.layer import Linear
    for module in model.modules():
        if not isinstance(module,Linear):continue
        attribute='_old_forward' if hasattr(module,'_old_forward') else 'forward'
        original=getattr(module,'_unit_scale_forward',getattr(module,attribute))
        module._unit_scale_forward=original
        def forward(x,*args,_module=module,_old=original,**kwargs):
            names=_module.active_adapters
            if (args or kwargs or torch.is_grad_enabled() or _module.disable_adapters or _module.merged
                    or len(names)!=1 or names[0] not in _module.lora_A or names[0] in _module.lora_variant):return _old(x,*args,**kwargs)
            if _module.scaling[names[0]]==0:return _module.base_layer(x)
            if _module.scaling[names[0]]!=1:return _old(x,*args,**kwargs)
            name=names[0];result=_module.base_layer(x)
            x=_module._cast_input_dtype(x,_module.lora_A[name].weight.dtype)
            # A unit multiplier adds a full activation pass without changing any value.
            return (result+_module.lora_B[name](_module.lora_A[name](_module.lora_dropout[name](x)))).to(result.dtype)
        setattr(module,attribute,forward)


def replace_loras(pipe,selections):
    """Rebind adapter hooks without materializing or replacing packed base weights."""
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module
    from backend.app.inference.conditioning.lora import apply_loras
    from backend.app.inference.runtime.execution import finish_transfers
    from backend.app.inference.ops.native_fp8 import attach_native_linear as fp8
    from backend.app.inference.ops.native_int8 import attach_native_linear as int8
    finish_transfers(pipe)
    model=pipe.transformer;mapping=pipe.weight_maps[0]
    if hasattr(mapping,'derived_identity'):
        from backend.app.inference.quantization.lora_weights import invalidate
        invalidate(mapping)
    for module in model.modules():
        hook=getattr(module,'_hf_hook',None)
        for item in getattr(hook,'hooks',(hook,)):
            if getattr(item,'offload',False):
                # Detach must not decode the complete source model into CPU tensors.
                item.original_devices={name:torch.device('meta') for name in item.original_devices}
    remove_hook_from_module(model,recurse=True)
    placeholders={}
    for name,value in model.state_dict().items():
        if name in mapping.sources:
            placeholders[name]=torch.zeros((),dtype=mapping.sources[name][1]).expand(value.shape)
        else:placeholders[name]=mapping.extra[name]
    model.load_state_dict(placeholders,strict=True,assign=True)
    encoder,vae=pipe.text_encoder,pipe.vae
    pipe.register_modules(text_encoder=None,vae=None)
    try:
        adapters=list(getattr(model,'peft_config',{}))
        if adapters:pipe.delete_adapters(adapters)
        apply_loras(pipe,selections)
    finally:pipe.register_modules(text_encoder=encoder,vae=vae)
    mapping.bind(model)
    from .krea2 import partial_cache_budgets
    partial_cache_budgets(pipe,pipe.memory_policy)
    pipe.memory_policy['weight_bytes']=sum(value.numel()*value.element_size() for component in (model,encoder,vae) for value in component.parameters())
    cpu_offload(model,torch.device('cuda:0'),state_dict=mapping,offload_buffers=len(model._parameters)>0)
    fp8(model,mapping)
    if mapping.packed_int8_enabled:int8(model,mapping)
    skip_unit_scaling(model)


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
