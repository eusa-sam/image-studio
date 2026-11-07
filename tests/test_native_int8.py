import os,unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from backend.app import native_int8
from backend.app.quantized_weights import undo_convrot


class NativeInt8Tests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt' and torch.cuda.is_available(),'Windows CUDA required')
    def test_wide_mlp_groups_and_upper_boundary(self):
        torch.manual_seed(123)
        with torch.inference_mode():
            for group in (16,64,256):
                x=torch.randn(17,16384,device='cuda',dtype=torch.bfloat16);x[0]=0
                weight=torch.randint(-127,128,(32,16384),device='cuda',dtype=torch.int8)
                scale=torch.rand(32,1,device='cuda')*.001
                expected=torch.nn.functional.linear(x,(undo_convrot(weight.float(),group)*scale).bfloat16())
                actual=native_int8.native_linear(x,weight,scale,group)
                self.assertTrue(torch.isfinite(actual).all());self.assertEqual(actual[0].count_nonzero(),0)
                relative=(actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()
                self.assertLess(float(relative),.02)
            self.assertIsNone(native_int8.native_linear(torch.empty(17,16640,device='cuda',dtype=torch.bfloat16),weight,scale,256))

    def test_cpu_inputs_keep_original_forward_and_compiler_is_not_loaded(self):
        model=torch.nn.Sequential(torch.nn.Linear(16,16))
        expected=model(torch.ones(17,16))
        model[0]._old_forward=model[0].forward
        mapping=SimpleNamespace(packed_sources={'0.weight':(256,'scale',16,16)},packed_cache={'0.weight':(torch.ones(16,16,dtype=torch.int8),torch.ones(16,1),16)},native_hits=0)
        native_int8.attach_native_linear(model,mapping)
        with patch.object(native_int8,'NativeLinear',side_effect=AssertionError('CPU must not compile')):
            torch.testing.assert_close(model(input=torch.ones(17,16)),expected)
        self.assertEqual(mapping.native_hits,0)

    @unittest.skipUnless(os.name=='nt' and torch.cuda.is_available(),'Windows CUDA required')
    def test_groups_zero_rows_noncontiguous_inputs_and_fallback(self):
        torch.manual_seed(123)
        with torch.inference_mode():
            for group in (16,64,256):
                x=torch.randn(2,17,256,device='cuda',dtype=torch.bfloat16).transpose(0,1);x[0]=0
                weight=torch.randint(-127,128,(32,256),device='cuda',dtype=torch.int8)
                scale=torch.rand(32,1,device='cuda')*.01
                expected=torch.nn.functional.linear(x,(undo_convrot(weight.float(),group)*scale).bfloat16())
                actual=native_int8.native_linear(x,weight,scale,group)
                self.assertIsNotNone(actual);self.assertEqual(actual.shape,expected.shape)
                self.assertTrue(torch.isfinite(actual).all());self.assertEqual(actual[0].count_nonzero(),0)
                relative=(actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()
                self.assertLess(float(relative),.02)
            self.assertIsNone(native_int8.native_linear(x.float(),weight,scale,256))
            self.assertIsNone(native_int8.native_linear(x,weight,scale,128))
            with patch.object(native_int8,'_kernel',None),patch.object(native_int8,'_unavailable',False),patch.object(native_int8,'NativeLinear',side_effect=OSError('no NVRTC')):
                with self.assertWarnsRegex(RuntimeWarning,'using BF16'):
                    self.assertIsNone(native_int8.native_linear(x,weight,scale,256))

    @unittest.skipUnless(os.name=='nt' and torch.cuda.is_available(),'Windows CUDA required')
    def test_native_base_keeps_offload_bias_and_lora_residual(self):
        import json,tempfile
        from pathlib import Path
        from safetensors.torch import save_file
        from accelerate import cpu_offload
        from peft import LoraConfig,get_peft_model
        from backend.app.quantized_weights import FileWeights
        with tempfile.TemporaryDirectory() as folder,torch.inference_mode():
            path=Path(folder)/'weights.safetensors'
            metadata={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':16}
            save_file({'0.weight':torch.randint(-30,31,(16,16),dtype=torch.int8),'0.weight_scale':torch.full((16,1),.02),
                '0.comfy_quant':torch.tensor(list(json.dumps(metadata).encode()),dtype=torch.uint8),'0.bias':torch.full((16,),.25)},path)
            with torch.device('meta'):model=torch.nn.Sequential(torch.nn.Linear(16,16))
            mapping=FileWeights(model,str(path),lambda x:x)
            model=get_peft_model(model,LoraConfig(target_modules=['0'],r=2,lora_alpha=2))
            for name,value in model.named_parameters():
                if 'lora_' in name:value.fill_(.125)
            inner=model.base_model.model
            mapping.bind(inner);mapping.packed_enabled=True;mapping.packed_int8_enabled=True;mapping.budget=4096
            cpu_offload(inner,torch.device('cuda:0'),state_dict=mapping)
            x=torch.randn(17,16,device='cuda',dtype=torch.bfloat16)
            baseline=model(x)
            native_int8.attach_native_linear(inner,mapping)
            actual=model(x);self.assertEqual(mapping.native_hits,1)
            self.assertEqual(mapping.statistics()['native_int8_layers'],{'0.base_layer.weight':1})
            torch.testing.assert_close(actual,baseline,rtol=.04,atol=.03)
            with model.disable_adapter():without=model(x)
            self.assertGreater(float((actual-without).abs().max()),.01)
            self.assertTrue(model.base_model.model[0].base_layer.weight.is_meta)
            with patch.object(native_int8,'native_linear',return_value=None):
                torch.testing.assert_close(model(x),baseline,rtol=0,atol=0)
            self.assertEqual(mapping.statistics()['native_int8_layers'],{'0.base_layer.weight':2})
            self.assertEqual(sum(mapping.native_layers.values()),mapping.native_hits)
            packed=mapping.packed_cache.pop('0.base_layer.weight')
            mapping.cache_bytes-=sum(v.numel()*v.element_size() for v in packed[:2])
            mapping.budget=0;mapping.limit_cpu_cache(2048)
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3):
                evicted=model(x)
            torch.testing.assert_close(evicted,actual,rtol=0,atol=0)
            self.assertEqual(mapping.native_hits,3)
            self.assertIn('0.base_layer.weight',mapping.native_cpu_cache)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):model(x)
