import unittest
import torch

from backend.app.inference.quantization.lora_weights import merge_packed
from backend.app.inference.quantization.quantized_weights import undo_convrot


class MergedWeightsTests(unittest.TestCase):
    def test_multiple_adapters_and_full_precision_layers_keep_original_forward(self):
        from types import SimpleNamespace
        from peft import LoraConfig,inject_adapter_in_model
        from backend.app.inference.quantization.lora_weights import configure
        model=torch.nn.ModuleDict({'layer':torch.nn.Linear(4,4,bias=False,dtype=torch.bfloat16)})
        inject_adapter_in_model(LoraConfig(target_modules=['layer'],r=2),model,adapter_name='lora_0')
        layer=model['layer'];key='layer.base_layer.weight'
        mapping=SimpleNamespace(packed_sources={key:(16,'scale',None,1)},full_precision_sources=set(),
                                sources={key:('layer.weight',torch.bfloat16)})
        pipe=SimpleNamespace(transformer=model,weight_maps=[mapping])
        selected=[(dict(profile={'sha256':'adapter'},weight=1.),None)]
        x=torch.ones(1,4,dtype=torch.bfloat16);expected=layer(x)
        original=layer.forward
        configure(pipe,'model',selected)
        torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)
        configure(pipe,'model',selected+selected)
        self.assertEqual(layer.forward,original);self.assertFalse(hasattr(layer,'_derived_original'))
        mapping.full_precision_sources.add(key)
        configure(pipe,'model',selected)
        self.assertEqual(layer.forward,original);self.assertEqual(mapping.derived_layers,0)
        torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)

    def test_configuration_resets_request_counts_and_preserves_same_identity_cache(self):
        from types import SimpleNamespace
        from peft import LoraConfig,inject_adapter_in_model
        from backend.app.inference.quantization.lora_weights import configure
        model=torch.nn.ModuleDict({'layer':torch.nn.Linear(4,4,bias=False,dtype=torch.bfloat16)})
        inject_adapter_in_model(LoraConfig(target_modules=['layer'],r=2),model,adapter_name='lora_0')
        key='layer.base_layer.weight'
        mapping=SimpleNamespace(packed_sources={key:(16,'scale',None,1)},full_precision_sources=set(),
                                sources={key:('layer.weight',torch.bfloat16)})
        pipe=SimpleNamespace(transformer=model,weight_maps=[mapping])
        selected=[(dict(profile={'sha256':'adapter'},weight=1.),None)]
        configure(pipe,'model',selected)
        sentinel=object();mapping.derived_cpu_cache['saved']=sentinel
        mapping.derived_request_calls=7;mapping.derived_request_fallback_calls=2
        configure(pipe,'model',selected)
        self.assertIs(mapping.derived_cpu_cache['saved'],sentinel)
        self.assertEqual((mapping.derived_request_calls,mapping.derived_request_fallback_calls),(0,0))
        self.assertEqual((mapping.lora_layers,mapping.derived_layers),(1,1))
        configure(pipe,'model',[])
        self.assertFalse(mapping.derived_cpu_cache)
        self.assertFalse(hasattr(model['layer'],'_derived_original'))
        self.assertEqual((mapping.lora_layers,mapping.derived_layers),(1,0))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA native merged LoRA')
    def test_native_wrapper_applies_lora_once_and_invalidates_strength(self):
        import json,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from accelerate import cpu_offload
        from peft import LoraConfig,inject_adapter_in_model
        from backend.app.inference.quantization.quantized_weights import FileWeights
        from backend.app.inference.quantization.lora_weights import configure
        from backend.app.inference.ops.native_fp8 import attach_native_linear,native_linear
        from backend.app.inference.runtime.memory_policy import retained_bytes,cached_storage_bytes
        with tempfile.TemporaryDirectory() as directory,torch.inference_mode():
            path=Path(directory)/'weights.safetensors'
            raw=(torch.arange(4096).reshape(64,64)%17-8).to(torch.float8_e4m3fn);scale=torch.tensor(.03125)
            save_file({'layer.weight':raw,'layer.weight_scale':scale,
                'layer.comfy_quant':torch.tensor(list(json.dumps({'format':'float8_e4m3fn'}).encode()),dtype=torch.uint8)},path)
            model=torch.nn.ModuleDict({'layer':torch.nn.Linear(64,64,bias=False,dtype=torch.bfloat16)})
            mapping=FileWeights(model,path,lambda state:state)
            inject_adapter_in_model(LoraConfig(target_modules=['layer'],r=4,lora_alpha=4),model,adapter_name='lora_0')
            layer=model['layer'];layer.lora_A['lora_0'].weight.fill_(.03125);layer.lora_B['lora_0'].weight.fill_(.0625)
            mapping.bind(model);mapping.budget=mapping.cpu_budget=1024**2
            mapping.packed_enabled=True
            a=mapping.extra['layer.lora_A.lora_0.weight'].clone();b=mapping.extra['layer.lora_B.lora_0.weight'].clone()
            cpu_offload(model,torch.device('cuda:0'),state_dict=mapping)
            attach_native_linear(model,mapping)
            pipe=SimpleNamespace(transformer=model,weight_maps=[mapping],memory_policy=dict(
                resident_budget_bytes=1024**2,cpu_cache_budget_bytes=1024**2,activation_reserve_bytes=1))
            x=torch.ones(32,64,device='cuda',dtype=torch.bfloat16)
            for strength in (1.,.5,1.):
                layer.scaling['lora_0']=strength
                configure(pipe,'model-hash',[(dict(profile={'sha256':'adapter-hash'},weight=strength),path)])
                packed=merge_packed(raw.cuda(),scale.cuda(),None,a.cuda(),b.cuda(),strength)
                expected=native_linear(x,*packed[:2])
                torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)
                self.assertEqual(len(mapping.derived_cpu_cache),1)
                for device in ('cpu','cuda'):
                    self.assertEqual(retained_bytes(pipe,device),cached_storage_bytes(pipe,device))
            layer.enable_adapters(False)
            torch.testing.assert_close(layer(x),native_linear(x,raw.cuda(),scale.cuda()),rtol=0,atol=0)
            layer.enable_adapters(True)
            mapping.derived_gpu_cache.clear();mapping.derived_gpu_bytes=0;mapping.budget=0
            from backend.app.inference.quantization.staging import staging_weights
            packed=merge_packed(raw.cuda(),scale.cuda(),None,a.cuda(),b.cuda(),1.)
            expected=native_linear(x,*packed[:2])
            with staging_weights(pipe):
                for _ in range(16):torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)
            self.assertGreater(pipe._staging_statistics['hits'],0)
            self.assertFalse(mapping.derived_gpu_cache)
            layer.scaling['lora_0']=.3;pipe.memory_policy['cpu_cache_budget_bytes']=0
            configure(pipe,'model-hash',[(dict(profile={'sha256':'adapter-hash'},weight=.3),path)])
            expected=layer._derived_original(x)
            torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)
            self.assertFalse(mapping.derived_cpu_cache);self.assertEqual(len(mapping.derived_failed),1)
            from unittest.mock import patch
            pipe.memory_policy['cpu_cache_budget_bytes']=1024**2
            configure(pipe,'model-hash',[(dict(profile={'sha256':'adapter-hash'},weight=.3),path)])
            with patch('backend.app.inference.quantization.lora_weights.merge_packed',side_effect=RuntimeError('preparation failed')):
                torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)
            self.assertFalse(mapping.derived_cpu_cache);self.assertEqual(len(mapping.derived_failed),1)
            from backend.app.inference.runtime.execution import release
            release(pipe)

    def test_fp8_and_convrot_restore_lora_once_without_changing_source(self):
        generator=torch.Generator().manual_seed(321)
        for group in (None,16,64,256):
            width=group or 64
            values=torch.randn(32,width,generator=generator)
            if group:
                rotated=undo_convrot(values,group)
                scale=rotated.abs().amax(1,keepdim=True)/127
                raw=(rotated/scale).round().clamp(-127,127).to(torch.int8)
                base=undo_convrot(raw.float(),group)*scale
            else:
                scale=values.abs().amax()/448;raw=(values/scale).to(torch.float8_e4m3fn)
                base=raw.float()*scale
            original=raw.view(torch.uint8).clone();original_scale=scale.clone()
            a=torch.randn(4,width,generator=generator)*.2;b=torch.randn(32,4,generator=generator)*.2
            for strength in (.3,1):
                packed,new_scale,new_group=merge_packed(raw,scale,group,a,b,strength)
                actual=undo_convrot(packed.float(),group)*new_scale if group else packed.float()*new_scale
                expected=base+(b@a)*strength
                self.assertEqual(new_group,group);self.assertEqual(packed.dtype,raw.dtype)
                self.assertLess((actual-expected).square().mean().sqrt().item(),.04)
                self.assertGreater((actual-(expected+(b@a)*strength)).square().mean().sqrt().item(),.01)
            self.assertTrue(torch.equal(original,raw.view(torch.uint8)))
            self.assertTrue(torch.equal(original_scale,scale))

    def test_nonfinite_adapter_is_never_published(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            merge_packed(torch.ones(16,16,dtype=torch.int8),torch.ones(16,1),16,
                         torch.full((1,16),float('nan')),torch.ones(16,1),1)
