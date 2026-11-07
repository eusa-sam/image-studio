"""Restore scaled FP8 and rotated INT8 without rewriting source weights."""
import json
import torch
from collections.abc import Mapping
from contextlib import contextmanager,ExitStack,nullcontext
from contextvars import ContextVar
from pathlib import Path

_readers=ContextVar('weight_readers',default=None)


@contextmanager
def weight_readers():
    if _readers.get() is not None:
        yield
        return
    with ExitStack() as stack:
        token=_readers.set((stack,{}))
        try:yield
        finally:_readers.reset(token)


def open_weights(path,device='cpu'):
    from safetensors import safe_open
    scope=_readers.get()
    if scope is None:return safe_open(str(path),framework='pt',device=device)
    stack,readers=scope
    key=(str(Path(path).resolve()),str(device))
    if key not in readers:
        readers[key]=stack.enter_context(safe_open(str(path),framework='pt',device=device))
    return nullcontext(readers[key])


def allocator_free_bytes():
    # Unused allocator blocks can back a new weight without claiming more driver memory.
    return torch.cuda.mem_get_info()[0]+max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())


def undo_convrot(tensor, group):
    if type(group) is not int or group not in (16,64,256) or tensor.ndim!=2 or tensor.shape[1]%group:
        raise ValueError('Invalid ConvRot group/shape')
    # ConvRot uses regular H4, not the usual Sylvester Hadamard basis.
    h4=torch.ones(4,4,device=tensor.device)-2*torch.eye(4,device=tensor.device).flip(1)
    h=h4
    while h.shape[0]<group:h=torch.kron(h,h4)
    return (tensor.reshape(tensor.shape[0],-1,group) @ (h/(group**.5))).reshape(tensor.shape)


def quantization(reader,name,tensor):
    marker = name.removesuffix('.weight') + '.comfy_quant'
    keys = set(reader.keys())
    if not name.endswith('.weight') or marker not in keys:
        raise ValueError('Quantized tensor has no supported metadata: '+name)
    config_tensor = reader.get_tensor(marker)
    if config_tensor.dtype != torch.uint8 or config_tensor.ndim != 1 or config_tensor.numel()>4096:
        raise ValueError('Invalid quantization metadata')
    config = json.loads(bytes(config_tensor.tolist()))
    if not isinstance(config,dict):
        raise ValueError('Unsupported quantization options')
    scale_name = name.removesuffix('.weight') + '.weight_scale'
    if scale_name not in keys:raise ValueError('Missing quantization scale')
    scale = reader.get_tensor(scale_name)
    if scale.dtype != torch.float32 or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid weight scale')
    if config.get('format')=='float8_e4m3fn':
        if set(config)-{'format','full_precision_matrix_mult'} or tensor.dtype!=torch.float8_e4m3fn or tensor.ndim!=2 or scale.numel()!=1:
            raise ValueError('Unsupported FP8 format/options')
    elif config.get('format')=='int8_tensorwise':
        if set(config)!={'format','convrot','convrot_groupsize'} or config['convrot'] is not True or tensor.dtype!=torch.int8 or tensor.ndim!=2 or scale.shape!=(tensor.shape[0],1):
            raise ValueError('Unsupported INT8 format/options')
        group=config['convrot_groupsize']
        if type(group) is not int or group not in (16,64,256) or tensor.shape[1]%group:
            raise ValueError('Invalid ConvRot group/shape')
    else:raise ValueError('Unsupported quantization format')
    return config,scale


def read_tensor(reader, name, *, dtype=torch.bfloat16):
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Unsupported compute dtype')
    tensor = reader.get_tensor(name)
    marker = name.removesuffix('.weight') + '.comfy_quant'
    if name.endswith('.weight') and marker in reader.keys():
        config,scale=quantization(reader,name,tensor)
        # Integer rotation first makes the power-of-two sums exact on CPU and GPU.
        tensor=undo_convrot(tensor.float(),config['convrot_groupsize'])*scale if config['format']=='int8_tensorwise' else tensor.float()*scale.reshape(())
    elif tensor.dtype not in (torch.float32,torch.float16,torch.bfloat16):
        raise ValueError('Quantized tensor has no supported metadata: ' + name)
    if not torch.isfinite(tensor).all():
        raise ValueError('Nonfinite tensor: ' + name)
    result = tensor.to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Tensor overflows compute dtype: ' + name)
    return result


def load_local_weights(model, path, convert, *, dtype=torch.bfloat16):
    """Validate all names/shapes on meta first, then materialize one source tensor at a time."""
    from safetensors import safe_open
    expected = model.state_dict()
    mapping = {}
    with safe_open(path, framework='pt', device='cpu') as reader:
        for key in reader.keys():
            if key.endswith(('.comfy_quant','.weight_scale')):
                continue
            tensor = torch.empty(reader.get_slice(key).get_shape(), device='meta')
            converted = convert({key:tensor})
            if len(converted)!=1:
                raise ValueError('Conversion must preserve each source tensor')
            name,value = next(iter(converted.items()))
            if name in mapping or name not in expected or value.shape != expected[name].shape:
                raise ValueError('Weight name/shape mismatch: '+name)
            mapping[name] = key
        if set(mapping)!=set(expected):
            raise ValueError('Missing model weights: '+str(sorted(set(expected)-set(mapping))))
        keep_fp32 = getattr(model, '_keep_in_fp32_modules', None) or ()
        for name,key in mapping.items():
            compute_dtype = torch.float32 if any(part in keep_fp32 for part in name.split('.')) else dtype
            tensor = read_tensor(reader,key,dtype=compute_dtype)
            values = convert({key:tensor})
            model.load_state_dict(values, strict=False, assign=True)
    if any(parameter.is_meta for parameter in list(model.parameters())+list(model.buffers())):
        raise ValueError('Model contains uninitialized weights')
    return model.eval().requires_grad_(False)


class FileWeights(Mapping):
    """Validated source weights decoded on demand; never retain the full BF16 CPU copy."""
    def __init__(self, model, path, convert, *, dtype=torch.bfloat16, preload_budget=0, validation_device="cpu", defer_numeric_validation=False):
        self.path, self.convert = Path(path), convert
        self.validate_on_read=defer_numeric_validation and not preload_budget
        self.stamp = self._stamp()
        self.sources, self.extra, self.cache = {}, {}, {}
        self.extra_gpu = {}
        self.file_views=set()
        self.packed_sources={}
        self.packed_cache={}
        self.packed_enabled=False
        self.packed_int8_enabled=False
        self.packed_hits=0
        self.fused_hits=0
        self.native_hits=0
        self.native_layers={}
        self.native_cpu_cache={}
        self.int8_bound_validations=0
        self.fp8_bound_validations=0
        self.transient_gpu_budget=0
        self.transient_gpu_reads=0
        self.cache_bytes = self.budget = 0
        self.reserve = 0
        self.cpu_budget, self.cpu_bytes = preload_budget, 0
        expected = model.state_dict()
        keep = getattr(model, '_keep_in_fp32_modules', None) or ()
        placeholders={}
        with open_weights(path,validation_device) as reader:
            for key in reader.keys():
                if key.endswith(('.comfy_quant','.weight_scale')):continue
                converted = convert({key:torch.empty(reader.get_slice(key).get_shape(),device='meta')})
                if len(converted)!=1:raise ValueError('Conversion must preserve each source tensor')
                name,value = next(iter(converted.items()))
                if name in self.sources or name not in expected or value.shape!=expected[name].shape:
                    raise ValueError('Weight name/shape mismatch: '+name)
                target_dtype=torch.float32 if any(part in keep for part in name.split('.')) else dtype
                self.sources[name]=(key,target_dtype)
                if reader.get_slice(key).get_dtype()=='F8_E4M3' and key.endswith('.weight'):
                    self.packed_sources[name]=(expected[name].numel(),key.removesuffix('.weight')+'.weight_scale',None,1)
                if reader.get_slice(key).get_dtype()=={torch.float32:'F32',torch.float16:'F16',torch.bfloat16:'BF16'}[target_dtype]:
                    self.file_views.add(name)
            if set(self.sources)!=set(expected):raise ValueError('Missing model weights')
            # Reuse validated denoiser weights within the same bounded CPU cache.
            for name,(key,target_dtype) in self.sources.items():
                config=None;bounded=False
                int8=reader.get_slice(key).get_dtype()=='I8'
                if int8 and not preload_budget:
                    shape=reader.get_slice(key).get_shape()
                    config,scale=quantization(reader,key,torch.empty(shape,device='meta',dtype=torch.int8))
                    # Each normalized row has group entries of magnitude 1/sqrt(group); INT8 includes -128.
                    bounded=bool(scale.numel() and float(scale.max())<=torch.finfo(target_dtype).max/(128*config['convrot_groupsize']**.5))
                    if bounded:self.int8_bound_validations+=1
                elif not preload_budget and reader.get_slice(key).get_dtype()=='F8_E4M3':
                    raw=(torch.empty(reader.get_slice(key).get_shape(),device='meta',dtype=torch.float8_e4m3fn)
                         if self.validate_on_read else reader.get_tensor(key))
                    config,scale=quantization(reader,key,raw)
                    # E4M3FN has only two NaN encodings; finite magnitudes are at most 448.
                    if not self.validate_on_read and not ((raw.view(torch.uint8)&127)!=127).all():raise ValueError('Nonfinite tensor: '+key)
                    bounded=float(scale.reshape(()))<=torch.finfo(target_dtype).max/448
                    if bounded:self.fp8_bound_validations+=1
                    del raw
                plain_on_read=(self.validate_on_read
                               and reader.get_slice(key).get_dtype() in ('F32','F16','BF16')
                               and key.removesuffix('.weight')+'.comfy_quant' not in reader.keys())
                # Plain weights pass read_tensor's finite/overflow checks before their first use.
                if not bounded and not plain_on_read:
                    value=read_tensor(reader,key,dtype=target_dtype)
                    self._cache_cpu(name,next(iter(convert({key:value}).values())),reserve=16*1024**3)
                    del value
                if int8:
                    if config is None:config=json.loads(bytes(reader.get_tensor(key.removesuffix('.weight')+'.comfy_quant').tolist()))
                    self.packed_sources[name]=(expected[name].numel(),key.removesuffix('.weight')+'.weight_scale',config['convrot_groupsize'],expected[name].shape[0])
                # PEFT needs CPU device/dtype during injection, not materialized base weights.
                placeholder=torch.zeros((),dtype=target_dtype).expand(expected[name].shape)
                placeholders[name]=placeholder
        model.load_state_dict(placeholders,strict=True,assign=True)
        model.eval().requires_grad_(False)

    def _stamp(self):
        stat=self.path.stat()
        return stat.st_size,stat.st_mtime_ns

    def _checked_raw(self,name,tensor):
        if self.validate_on_read and tensor.dtype==torch.float8_e4m3fn:
            if not ((tensor.view(torch.uint8)&127)!=127).all():
                raise ValueError('Nonfinite tensor: '+name)
        return tensor

    def bind(self, model):
        sources,cache,views,packed,native={},{},set(),{},{}
        for name,value in model.state_dict().items():
            original=name.replace('.base_layer.','.')
            if original in self.sources:
                sources[name]=self.sources[original]
                if original in self.cache:cache[name]=self.cache[original]
                if original in self.file_views:views.add(name)
                if original in self.packed_sources:packed[name]=self.packed_sources[original]
                if original in self.native_cpu_cache:native[name]=self.native_cpu_cache[original]
            elif not value.is_meta:self.extra[name]=value
            else:raise ValueError('Uninitialized adapter: '+name)
        if len(sources)!=len(self.sources):raise ValueError('Lost base weights during adapter injection')
        self.sources=sources
        self.cache=cache
        self.file_views=views
        self.packed_sources=packed
        self.native_cpu_cache=native

    def limit_cpu_cache(self, budget):
        self.cpu_budget=budget
        for name in list(self.cache):
            if self.cpu_bytes<=budget:break
            tensor=self.cache[name]
            if tensor.device.type=='cpu':
                self.cpu_bytes-=tensor.numel()*tensor.element_size()
                del self.cache[name]
        for name in list(self.native_cpu_cache):
            if self.cpu_bytes<=budget:break
            self.cpu_bytes-=sum(v.numel()*v.element_size() for v in self.native_cpu_cache.pop(name))

    def offload_cache(self):
        from .memory_policy import available_ram
        self.extra_gpu.clear()
        for name,(raw,scale,group) in list(self.packed_cache.items()):
            size=raw.numel()*raw.element_size()+scale.numel()*scale.element_size()
            if self.cpu_bytes+size<=self.cpu_budget and available_ram()>=8*1024**3+size:
                self.native_cpu_cache[name]=(raw.cpu(),scale.cpu())
                self.cpu_bytes+=size
        self.packed_cache.clear()
        for name in list(self.cache):
            tensor=self.cache[name]
            if tensor.device.type=='cuda':
                del self.cache[name]
                self._cache_cpu(name,tensor)
        self.cache_bytes=0

    def _cache_cpu(self,name,tensor,*,reserve=8*1024**3):
        size=tensor.numel()*tensor.element_size()
        if self.cpu_bytes+size<=self.cpu_budget:
            from .memory_policy import available_ram
            if available_ram()>=reserve+size:
                # A safetensors view can otherwise retain the mapping of the entire source file.
                tensor=tensor.cpu()
                self.cache[name]=tensor.clone() if name in self.file_views else tensor
                self.cpu_bytes+=size

    def __iter__(self):return iter(self.sources.keys() | self.extra.keys())
    def __len__(self):return len(self.sources)+len(self.extra)
    def __contains__(self,key):return key in self.sources or key in self.extra

    def statistics(self):
        return dict(gpu_bytes=self.cache_bytes,cpu_bytes=self.cpu_bytes,validate_on_read=bool(self.validate_on_read),
                    adapter_gpu_bytes=sum(value.numel()*value.element_size() for value in self.extra_gpu.values()),
                    packed_fp8_tensors=sum(weight.dtype==torch.float8_e4m3fn for weight,_,_ in self.packed_cache.values()),
                    packed_int8_tensors=sum(weight.dtype==torch.int8 for weight,_,_ in self.packed_cache.values()),packed_hits=self.packed_hits,fused_int8_hits=self.fused_hits,
                    transient_gpu_reads=self.transient_gpu_reads,native_int8_linear_hits=self.native_hits,
                    int8_bound_validations=self.int8_bound_validations,
                    fp8_bound_validations=self.fp8_bound_validations,
                    native_cpu_bytes=sum(v.numel()*v.element_size() for pair in self.native_cpu_cache.values() for v in pair),
                    native_int8_layers=dict(sorted(self.native_layers.items())))

    def linear_packed(self,name,device):
        if self._stamp()!=self.stamp:raise ValueError('COMPONENT_CHANGED during native inference')
        if name in self.packed_cache:return self.packed_cache[name]
        if self._pack(name):return self.packed_cache[name]
        key,_=self.sources[name]
        _,scale_name,group,_=self.packed_sources[name]
        if name in self.native_cpu_cache:raw,scale=self.native_cpu_cache[name]
        else:
            with open_weights(self.path) as reader:
                raw,scale=self._checked_raw(name,reader.get_tensor(key)),reader.get_tensor(scale_name)
            previous=self.cache.pop(name,None)
            if previous is not None:
                size=previous.numel()*previous.element_size()
                if previous.device.type=='cpu':self.cpu_bytes-=size
                else:self.cache_bytes-=size
            del previous
            size=raw.numel()*raw.element_size()+scale.numel()*scale.element_size()
            from .memory_policy import available_ram
            if self.cpu_bytes+size<=self.cpu_budget and available_ram()>=8*1024**3+size:
                raw,scale=raw.clone(),scale.clone()
                self.native_cpu_cache[name]=(raw,scale)
                self.cpu_bytes+=size
        result=(raw.to(device),scale.to(device),group)
        if self._stamp()!=self.stamp:raise ValueError('COMPONENT_CHANGED during native transfer')
        return result

    def _unpack(self,name):
        weight,scale,group=self.packed_cache[name]
        key,dtype=self.sources[name]
        if group and dtype==torch.bfloat16:
            from .cuda_decode import decode_int8
            restored=decode_int8(weight,scale,group)
            if restored is not None:
                self.fused_hits+=1
                return next(iter(self.convert({key:restored}).values()))
        restored=undo_convrot(weight.float(),group)*scale if group else weight.float()*scale.reshape(())
        restored=restored.to(dtype)
        return next(iter(self.convert({key:restored}).values()))

    def _pack(self,name):
        if not self.packed_enabled or name not in self.packed_sources:return False
        elements,scale_name,group,scale_elements=self.packed_sources[name]
        if group and not self.packed_int8_enabled:return False
        size=elements+4*scale_elements
        if self.cache_bytes+size>self.budget:return False
        if allocator_free_bytes()<self.reserve+size+elements*(16 if group else 10):return False
        if name in self.native_cpu_cache:
            weight,scale=(value.to('cuda') for value in self.native_cpu_cache[name])
        else:
            key,_=self.sources[name]
            with open_weights(self.path) as reader:
                weight=reader.get_tensor(key).to('cuda')
                scale=reader.get_tensor(scale_name).to('cuda')
        self._checked_raw(name,weight)
        self.packed_cache[name]=(weight,scale,group)
        self.cache_bytes+=size
        previous=self.cache.pop(name,None)
        if previous is not None:self.cpu_bytes-=previous.numel()*previous.element_size()
        previous_raw=self.native_cpu_cache.pop(name,None)
        if previous_raw is not None:self.cpu_bytes-=sum(v.numel()*v.element_size() for v in previous_raw)
        return True

    def __getitem__(self,name):
        if self._stamp()!=self.stamp:raise ValueError('COMPONENT_CHANGED during inference')
        if name in self.extra:
            if name in self.extra_gpu:return self.extra_gpu[name]
            tensor=self.extra[name]
            size=tensor.numel()*tensor.element_size()
            if self.cache_bytes+size<=self.budget and allocator_free_bytes()>=self.reserve+size:
                self.extra_gpu[name]=tensor.to('cuda')
                self.cache_bytes+=size
                return self.extra_gpu[name]
            return tensor
        if name in self.packed_cache:
            self.packed_hits+=1
            return self._unpack(name)
        cached=name in self.cache
        if cached:
            tensor=self.cache[name]
            if tensor.device.type=='cuda':return tensor
        if self._pack(name):return self._unpack(name)
        if not cached:
            key,dtype=self.sources[name]
            packed=self.packed_sources.get(name)
            device='cuda:0' if packed and packed[2] is None and packed[0]*10<=self.transient_gpu_budget else 'cpu'
            with open_weights(self.path,device) as reader:
                if name in self.packed_sources:
                    # Scale bounds are checked at construction; deferred NaN checks precede use.
                    _,scale_name,group,_=self.packed_sources[name]
                    raw=self._checked_raw(name,reader.get_tensor(key)).float()
                    restored=undo_convrot(raw,group)*reader.get_tensor(scale_name) if group else raw*reader.get_tensor(scale_name).reshape(())
                    value=self.convert({key:restored.to(dtype)})
                else:
                    value=self.convert({key:read_tensor(reader,key,dtype=dtype)})
                tensor=next(iter(value.values()))
            if device=='cuda:0':
                self.transient_gpu_reads+=1
                return tensor
        size=tensor.numel()*tensor.element_size()
        if self.cache_bytes+size<=self.budget and allocator_free_bytes()>=self.reserve+size:
            tensor=tensor.to('cuda')
            self.cache[name]=tensor
            self.cache_bytes+=size
            if cached:self.cpu_bytes-=size
        elif not cached:self._cache_cpu(name,tensor)
        return self.cache.get(name,tensor)
