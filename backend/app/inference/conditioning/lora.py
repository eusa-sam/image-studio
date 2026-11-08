"""Validate local LoRA tensors and targets before adapter injection."""
def validate_lora(pipe, path, *, raw=None, require_dora=False):
    import torch
    from torch.nn import Conv2d
    from safetensors.torch import load_file
    from diffusers.utils.state_dict_utils import convert_all_state_dict_to_peft, convert_unet_state_dict_to_peft
    if raw is None:
        raw = load_file(str(path), device='cpu')
        if type(getattr(pipe,'transformer',None)).__name__=='Krea2Transformer2DModel':
            from backend.app.inference.models.krea2_lora import convert_lora
            raw=convert_lora(raw)
    if any('lokr' in key or 'hada' in key for key in raw):
        raise ValueError('LoKr/LoHa는 미지원입니다.')
    if hasattr(pipe,'transformer') and any('dora' in key or 'lora_magnitude_vector' in key for key in raw):
        raise ValueError('DoRA는 현재 SD 계열 UNet만 지원합니다.')
    for key,value in raw.items():
        if key.endswith('.alpha') and (value.numel()!=1 or value.is_complex() or not torch.isfinite(value).all()):
            raise ValueError('Invalid LoRA alpha: '+key)
    if hasattr(pipe,'transformer'):
        state = pipe.lora_state_dict(dict(raw))
        allowed = ('transformer','text_conditioner')
    else:
        state, _ = pipe.lora_state_dict(dict(raw), unet_config=pipe.unet.config)
        unet = convert_unet_state_dict_to_peft({key:value for key,value in state.items() if key.startswith('unet.')})
        other = {key:value for key,value in state.items() if not key.startswith('unet.')}
        state = unet | (convert_all_state_dict_to_peft(other) if other else {})
        allowed = ('unet','text_encoder','text_encoder_2')
    pairs = {}
    for key, tensor in state.items():
        suffix = next((suffix for suffix in ('.lora_A.weight','.lora_B.weight','.lora_magnitude_vector.weight','.lora_magnitude_vector') if key.endswith(suffix)),None)
        if suffix is None:
            raise ValueError('지원하지 않는 LoRA 키: '+key)
        if tensor.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid or nonfinite LoRA weight: '+key)
        target = key[:-len(suffix)]
        normalized='.lora_magnitude_vector' if 'magnitude' in suffix else suffix
        if normalized in pairs.setdefault(target,{}):raise ValueError('Duplicate LoRA tensor: '+key)
        pairs[target][normalized] = tensor
    if not pairs:
        raise ValueError('LoRA 가중치가 없습니다.')
    for target, weights in pairs.items():
        component, separator, module = target.partition('.')
        if component not in allowed or not separator:
            raise ValueError('LoRA 대상 구성요소 오류: '+target)
        try:
            encoder = getattr(pipe,component)
            # Match Diffusers' legacy CLIP key handling for Transformers >=5.6.
            if component.startswith('text_encoder') and not hasattr(encoder,'text_model'):
                module = module.removeprefix('text_model.')
            layer = encoder.get_submodule(module)
        except (AttributeError, KeyError):
            raise ValueError('LoRA 대상 레이어 없음: '+target) from None
        if hasattr(layer,'get_base_layer'):layer=layer.get_base_layer()
        base = layer.weight
        a, b = weights.get('.lora_A.weight'), weights.get('.lora_B.weight')
        if a is None or b is None:
            raise ValueError('완전한 LoRA A/B 쌍이 필요합니다: '+target)
        linear = a.ndim == b.ndim == base.ndim == 2
        convolution = (isinstance(layer,Conv2d) and a.ndim == b.ndim == base.ndim == 4 and
                       a.shape[2:] == base.shape[2:] and b.shape[2:] == (1,1) and
                       layer.groups == 1 and layer.dilation == (1,1) and layer.padding_mode == 'zeros')
        if not (linear or convolution):
            raise ValueError('Linear 또는 원본 커널 down + 1x1 up Conv2d LoRA가 필요합니다 (groups=1, dilation=1, zero padding): '+target)
        if a.shape[0]<1 or a.shape[0] != b.shape[1] or (b.shape[0],a.shape[1]) != tuple(base.shape[:2]):
            raise ValueError('LoRA shape 불일치: '+target)
        magnitude=weights.get('.lora_magnitude_vector')
        if require_dora and component=='unet' and magnitude is None:
            raise ValueError('DoRA metadata requires magnitude: '+target)
        if magnitude is not None:
            if component!='unet':raise ValueError('DoRA는 현재 SD 계열 UNet만 지원합니다.')
            expected=(base.shape[0],) if linear else (1,base.shape[0],1,1)
            if tuple(magnitude.shape)!=expected:raise ValueError('DoRA magnitude shape 불일치: '+target)
    return len(pairs)


def apply_loras(pipe, selections):
    import json, math
    from safetensors import safe_open
    from safetensors.torch import load_file
    for index, (selection,path) in enumerate(selections):
        require_dora=False
        before=path.stat()
        stamp=(before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_ino)
        with safe_open(str(path),framework='pt',device='cpu') as reader:
            metadata=(reader.metadata() or {}).get('lora_adapter_metadata')
        if metadata:
            config=json.loads(metadata)
            if not isinstance(config,dict):raise ValueError('Invalid LoRA adapter metadata')
            for key,value in config.items():
                option=key.rsplit('.',1)[-1]
                if option=='use_dora' and value:
                    if key!='unet.use_dora' or not hasattr(pipe,'unet') or type(value) is not bool:
                        raise ValueError('Unsupported LoRA metadata: '+key)
                    require_dora=True
                if option in ('use_qalora','layer_replication','modules_to_save','target_parameters','trainable_token_indices','alora_invocation_tokens') and value:
                    raise ValueError('Unsupported LoRA metadata: '+key)
                if option=='init_lora_weights' and type(value) is not bool:
                    raise ValueError('Unsupported LoRA initialization: '+key)
                if option=='lora_alpha' and (type(value) not in (int,float) or not math.isfinite(value)):
                    raise ValueError('Invalid LoRA alpha: '+key)
        raw=load_file(str(path),device='cpu')
        if type(getattr(pipe,'transformer',None)).__name__=='Krea2Transformer2DModel':
            from backend.app.inference.models.krea2_lora import convert_lora
            raw=convert_lora(raw)
        validate_lora(pipe,path,raw=raw,require_dora=require_dora)
        # Diffusers reads rank/alpha/rsLoRA metadata only from a safetensors path.
        pipe.load_lora_weights(str(path) if metadata else raw,adapter_name=f'lora_{index}',local_files_only=True,use_safetensors=True)
        after=path.stat()
        if stamp!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino):
            raise ValueError('COMPONENT_CHANGED during LoRA loading')
    set_lora_weights(pipe,selections)


def set_lora_weights(pipe,selections):
    if selections:
        if any(getattr(config,'use_dora',False) for config in getattr(getattr(pipe,'unet',None),'peft_config',{}).values()):
            # Zero DoRA scaling still changes magnitude; omit the adapter entirely.
            active=[(f'lora_{i}',selection['weight']) for i,(selection,_) in enumerate(selections) if selection['weight']!=0]
            if not active:
                pipe.disable_lora()
                return
            pipe.enable_lora()
            pipe.set_adapters([name for name,_ in active],adapter_weights=[weight for _,weight in active])
            return
        if hasattr(pipe,'unet'):
            # A removed zero-strength DoRA can leave PEFT layers disabled.
            pipe.enable_lora()
        pipe.set_adapters([f'lora_{i}' for i in range(len(selections))],
                          adapter_weights=[selection['weight'] for selection,_ in selections])
