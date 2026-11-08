import json
import unittest
import torch
import tempfile
from pathlib import Path
from safetensors.torch import save_file
from backend.app.inference.quantization.quantized_weights import read_tensor, load_local_weights, undo_convrot


class Reader:
    def __init__(self, tensors):self.tensors=tensors
    def keys(self):return self.tensors.keys()
    def get_tensor(self,key):return self.tensors[key]


class QuantizedWeightsTests(unittest.TestCase):
    def test_budget_counters_match_owned_storage_through_derived_eviction(self):
        from backend.app.inference.quantization.quantized_weights import FileWeights
        from backend.app.inference.runtime.memory_policy import retained_bytes,cached_storage_bytes
        from backend.app.inference.quantization.lora_weights import invalidate
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors';save_file(self.fixture().tensors,path)
            model=torch.nn.ModuleDict({'layer':torch.nn.Linear(2,2,bias=False)})
            mapping=FileWeights(model,path,lambda state:state);mapping.bind(model);mapping.cpu_budget=1024
            pipe=SimpleNamespace(weight_maps=[mapping])
            with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                mapping._cache_cpu('layer.weight',torch.ones(2,2))
            invalidate(mapping)
            mapping.derived_cpu_cache['test']=(torch.ones(2,2,dtype=torch.int8),torch.ones(()),16)
            mapping.derived_cpu_bytes=8
            self.assertEqual(retained_bytes(pipe,'cpu'),cached_storage_bytes(pipe,'cpu'))
            self.assertEqual(retained_bytes(pipe,'cpu'),24)
            mapping.limit_cpu_cache(16)
            self.assertFalse(mapping.derived_cpu_cache)
            self.assertEqual(retained_bytes(pipe,'cpu'),cached_storage_bytes(pipe,'cpu'))
            mapping.clear_cache();self.assertEqual(retained_bytes(pipe,'cpu'),0)

    def test_rebinding_adapters_reuses_base_storage_and_discards_adapter_storage(self):
        from backend.app.inference.quantization.quantized_weights import FileWeights
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors';save_file(self.fixture().tensors,path)
            model=torch.nn.ModuleDict({'layer':torch.nn.Linear(2,2,bias=False)})
            mapping=FileWeights(model,path,lambda state:state)
            base=torch.ones(2,2);packed=(base,torch.ones(()),None)
            mapping.cache['layer.weight']=base
            mapping.packed_cache['layer.weight']=packed
            mapping.extra_gpu['stale']=torch.ones(1)
            bound=SimpleNamespace(state_dict=lambda:{'layer.base_layer.weight':base,'layer.lora_A.a.weight':torch.ones(1,2)})
            mapping.bind(bound)
            self.assertIs(mapping.cache['layer.base_layer.weight'],base)
            self.assertIs(mapping.packed_cache['layer.base_layer.weight'],packed)
            self.assertEqual(set(mapping.extra),{'layer.lora_A.a.weight'})
            self.assertFalse(mapping.extra_gpu)
            mapping.bind(model)
            self.assertIs(mapping.cache['layer.weight'],base)
            self.assertFalse(mapping.extra)
            self.assertEqual(set(mapping.packed_sources),{'layer.weight'})

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_int8_nonbinary_scales_restore_identically_on_cpu_and_gpu(self):
        generator=torch.Generator().manual_seed(123)
        for group in (16,64,256):
            raw=torch.randint(-127,128,(32,group*4),dtype=torch.int8,generator=generator)
            scale=torch.rand((32,1),generator=generator)*.017+.001
            metadata={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':group}
            values={'layer.weight':raw,'layer.weight_scale':scale,'layer.comfy_quant':torch.tensor(list(json.dumps(metadata).encode()),dtype=torch.uint8)}
            cpu=read_tensor(Reader(values),'layer.weight')
            gpu=read_tensor(Reader({k:v.cuda() for k,v in values.items()}),'layer.weight')
            torch.testing.assert_close(cpu,gpu.cpu(),rtol=0,atol=0)
            precise=(undo_convrot(raw.float(),group).double()*scale.double()).to(torch.bfloat16)
            torch.testing.assert_close(cpu,precise,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_rotation_matches_representable_cpu_reference(self):
        for group in (16,64,256):
            values=(torch.arange(2*group,dtype=torch.float32).reshape(2,group)%17)-8
            expected=undo_convrot(values,group)
            actual=undo_convrot(values.cuda(),group)
            self.assertEqual(actual.device.type,'cuda')
            torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)

    def test_int8_rotation_reconstructs_linear_output_and_preserves_source(self):
        h4=torch.tensor([[1.,1.,1.,-1.],[1.,1.,-1.,1.],[1.,-1.,1.,1.],[-1.,1.,1.,1.]])/2
        h=torch.kron(h4,h4)
        torch.testing.assert_close(h@h,torch.eye(16))
        quantized=torch.arange(-32,32,dtype=torch.int8).reshape(2,32)
        scale=torch.tensor([[.25],[.5]])
        metadata={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':16}
        reader=Reader({'layer.weight':quantized,'layer.weight_scale':scale,
                       'layer.comfy_quant':torch.tensor(list(json.dumps(metadata).encode()),dtype=torch.uint8)})
        restored=read_tensor(reader,'layer.weight',dtype=torch.float32)
        x=torch.arange(32,dtype=torch.float32).reshape(1,32)/32
        rotated_x=(x.reshape(1,2,16)@h).reshape(1,32)
        torch.testing.assert_close(x@restored.T,rotated_x@(quantized.float()*scale).T)
        torch.testing.assert_close(undo_convrot(restored,16),quantized.float()*scale)
        self.assertTrue(torch.equal(reader.get_tensor('layer.weight'),quantized))
        for group in (64,256):
            weight=torch.arange(group,dtype=torch.float32).reshape(1,group)/group
            torch.testing.assert_close(undo_convrot(undo_convrot(weight,group),group),weight)
        for invalid in (8,True,16.,1024):
            with self.assertRaises(ValueError):undo_convrot(restored,invalid)
        reader.tensors['layer.weight_scale']=torch.ones(2)
        with self.assertRaises(ValueError):read_tensor(reader,'layer.weight')

    def test_norm_dtype_follows_model_contract(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'norm.safetensors'
            save_file({'norm.weight':torch.ones(4)},path)
            with torch.device('meta'):
                model=torch.nn.ModuleDict({'norm':Qwen3VLTextRMSNorm(4)})
            load_local_weights(model,str(path),lambda state:state)
            value=model['norm'](torch.ones(1,4,dtype=torch.bfloat16))
            self.assertEqual(value.dtype,torch.bfloat16)
            self.assertTrue(torch.isfinite(value).all())
            model._keep_in_fp32_modules=['norm']
            load_local_weights(model,str(path),lambda state:state)
            self.assertEqual(model['norm'].weight.dtype,torch.float32)

    def test_strict_meta_loading_materializes_only_matching_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            reader=self.fixture();reader.tensors['layer.bias']=torch.tensor([1.,2.])
            save_file(reader.tensors,path)
            with torch.device('meta'):model=torch.nn.ModuleDict({'layer':torch.nn.Linear(2,2)})
            load_local_weights(model,str(path),lambda state:state,dtype=torch.float32)
            expected=torch.nn.functional.linear(torch.ones(1,2),reader.tensors['layer.weight'].float()*.25,reader.tensors['layer.bias'])
            torch.testing.assert_close(model['layer'](torch.ones(1,2)),expected)
            with torch.device('meta'):wrong=torch.nn.ModuleDict({'layer':torch.nn.Linear(3,2)})
            with self.assertRaisesRegex(ValueError,'shape mismatch'):load_local_weights(wrong,str(path),lambda state:state)
            self.assertTrue(wrong['layer'].weight.is_meta)

    def fixture(self, **config):
        values=torch.tensor([[1.,-2.],[.5,3.]])
        return Reader({'layer.weight':values.to(torch.float8_e4m3fn),'layer.weight_scale':torch.tensor(.25),
            'layer.comfy_quant':torch.tensor(list(json.dumps({'format':'float8_e4m3fn',**config}).encode()),dtype=torch.uint8)})

    def test_fp8_scalar_scale_and_dtype_preserve_source(self):
        reader=self.fixture();source=reader.get_tensor('layer.weight').float().clone()
        decoded=read_tensor(reader,'layer.weight',dtype=torch.float32)
        torch.testing.assert_close(decoded,source*.25)
        torch.testing.assert_close(reader.get_tensor('layer.weight').float(),source)
        self.assertEqual(read_tensor(reader,'layer.weight').dtype,torch.bfloat16)

    def test_unsafe_scales_unmarked_fp8_and_rotation_are_rejected(self):
        for scale in (torch.tensor(float('nan')),torch.tensor(-1.),torch.ones(2),torch.tensor(1,dtype=torch.int32)):
            reader=self.fixture();reader.tensors['layer.weight_scale']=scale
            with self.assertRaises(ValueError):read_tensor(reader,'layer.weight')
        with self.assertRaises(ValueError):read_tensor(self.fixture(convrot=True),'layer.weight')
        reader=self.fixture();reader.tensors.pop('layer.comfy_quant')
        with self.assertRaises(ValueError):read_tensor(reader,'layer.weight')
        reader=Reader({'weight':torch.tensor(float('inf'))})
        with self.assertRaises(ValueError):read_tensor(reader,'weight')
