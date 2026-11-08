import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from backend.app.inference.ops import native_fp8
from backend.app.inference.quantization.quantized_weights import FileWeights


class NativeFp8Tests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_fused_scale_matches_torch_and_preserves_nan_and_zero(self):
        with torch.inference_mode():
            for dtype in (torch.bfloat16,torch.float16):
                bits=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.uint16).view(dtype)
                for values in (*bits.split(256),torch.zeros(4097,device='cuda',dtype=dtype),
                               torch.randn((513,2048),device='cuda',dtype=dtype).t()):
                    expected=(values.abs().amax().float()/448).clamp_min(1e-12)
                    actual=native_fp8.input_scale(values)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_fused_input_conversion_matches_torch_bits_and_missing_compiler_fallback(self):
        if torch.cuda.get_device_capability()<(8,9):self.skipTest('Native FP8 requires SM89+')
        import os
        with torch.inference_mode():
            for dtype in (torch.bfloat16,torch.float16):
                values=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.uint16).view(dtype).reshape(256,256)
                for factor in (.125,1.,1.235):
                    scale=torch.tensor(factor,device='cuda')
                    expected=(values.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
                    actual=native_fp8.quantize_input(values,scale)
                    finite=torch.isfinite(values)
                    torch.testing.assert_close(actual.view(torch.uint8)[finite],expected.view(torch.uint8)[finite],rtol=0,atol=0)
                    self.assertTrue(torch.equal(torch.isnan(actual.float()),torch.isnan(expected.float())))
            if os.name=='nt':
                self.assertIsNotNone(native_fp8._quantizer)
                with patch.object(native_fp8,'_quantizer',None),patch.object(native_fp8,'_quantizer_unavailable',False),patch('backend.app.inference.ops.cuda_decode.Kernel',side_effect=OSError('no compiler')):
                    with self.assertWarnsRegex(RuntimeWarning,'using Torch'):
                        actual=native_fp8.quantize_input(values,scale)
                    torch.testing.assert_close(actual.float(),expected.float(),rtol=0,atol=0,equal_nan=True)

    def fixture(self,folder,full_precision=False):
        path=Path(folder)/'weights.safetensors'
        torch.manual_seed(42)
        config={'format':'float8_e4m3fn','full_precision_matrix_mult':full_precision}
        save_file({'0.weight':torch.randn(32,32).to(torch.float8_e4m3fn),
                   '0.weight_scale':torch.tensor(.125),'0.bias':torch.full((32,),.25),
                   '0.comfy_quant':torch.tensor(list(json.dumps(config).encode()),dtype=torch.uint8)},path)
        with torch.device('meta'):model=torch.nn.Sequential(torch.nn.Linear(32,32))
        return path,model

    def test_full_precision_metadata_is_validated_and_kept_after_lora_binding(self):
        from peft import LoraConfig,get_peft_model
        for flag in (True,False):
            with tempfile.TemporaryDirectory() as folder:
                path,model=self.fixture(folder,flag)
                mapping=FileWeights(model,str(path),lambda x:x)
                model=get_peft_model(model,LoraConfig(target_modules=['0'],r=2,lora_alpha=2)).base_model.model
                mapping.bind(model)
                for module in model.modules():
                    if isinstance(module,torch.nn.Linear):module._old_forward=module.forward
                native_fp8.attach_native_linear(model,mapping)
                self.assertEqual('0.base_layer.weight' in mapping.full_precision_sources,flag)
                self.assertEqual('0.base_layer.weight' in mapping.native_fp8_sources,not flag)
        for flag in (1,'false',None):
            with tempfile.TemporaryDirectory() as folder:
                path,model=self.fixture(folder,flag)
                with self.assertRaisesRegex(ValueError,'full_precision_matrix_mult'):FileWeights(model,str(path),lambda x:x)

    def test_cpu_and_incompatible_input_keep_fallback(self):
        self.assertIsNone(native_fp8.native_linear(torch.ones(2,32),torch.ones(32,32),torch.tensor(1.)))
        with tempfile.TemporaryDirectory() as folder:
            path,model=self.fixture(folder)
            mapping=FileWeights(model,str(path),lambda x:x)
            model[0]._old_forward=model[0].forward
            expected=model(torch.ones(2,32,dtype=torch.bfloat16))
            native_fp8.attach_native_linear(model,mapping)
            with patch.object(mapping,'linear_packed',side_effect=AssertionError('CPU must not transfer')):
                torch.testing.assert_close(model(torch.ones(2,32,dtype=torch.bfloat16)),expected)
            self.assertEqual(mapping.fp8_fallbacks,1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_scaled_mm_noncontiguous_zero_outlier_and_odd_rows(self):
        if torch.cuda.get_device_capability()<(8,9):self.skipTest('Native FP8 requires SM89+')
        torch.manual_seed(14)
        with torch.inference_mode():
            for dtype in (torch.bfloat16,torch.float16):
                x=torch.randn(3,17,64,device='cuda',dtype=dtype).transpose(0,1)
                x[0]=0;x[-1]*=20
                raw=torch.randn(32,64,device='cuda').to(torch.float8_e4m3fn)
                scale=torch.tensor(.125,device='cuda')
                expected=torch.nn.functional.linear(x,(raw.float()*scale).to(dtype))
                actual=native_fp8.native_linear(x,raw,scale)
                self.assertEqual(actual.shape,expected.shape);self.assertEqual(actual.dtype,dtype)
                self.assertTrue(torch.isfinite(actual).all());self.assertEqual(actual[0].count_nonzero(),0)
                relative=(actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()
                self.assertLess(float(relative),.04)
                zero=native_fp8.native_linear(torch.zeros_like(x),raw,scale)
                self.assertEqual(zero.count_nonzero(),0)
                self.assertIsNone(native_fp8.native_linear(x.float(),raw,scale))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_offload_lora_bias_packed_cache_eviction_and_source_change(self):
        if torch.cuda.get_device_capability()<(8,9):self.skipTest('Native FP8 requires SM89+')
        from accelerate import cpu_offload
        from peft import LoraConfig,get_peft_model
        with tempfile.TemporaryDirectory() as folder,torch.inference_mode():
            path,model=self.fixture(folder)
            mapping=FileWeights(model,str(path),lambda x:x,defer_numeric_validation=True)
            model=get_peft_model(model,LoraConfig(target_modules=['0'],r=2,lora_alpha=2))
            for name,value in model.named_parameters():
                if 'lora_' in name:value.fill_(.125)
            inner=model.base_model.model;mapping.bind(inner)
            mapping.packed_enabled=True;mapping.budget=8192
            cpu_offload(inner,torch.device('cuda:0'),state_dict=mapping)
            x=torch.randn(17,32,device='cuda',dtype=torch.bfloat16)
            baseline=model(x)
            native_fp8.attach_native_linear(inner,mapping)
            with patch.object(mapping,'_unpack',side_effect=AssertionError('Native must not restore BF16 weights')):
                actual=model(x)
            self.assertEqual(mapping.fp8_hits,1)
            self.assertTrue(inner[0].base_layer.weight.is_meta)
            from backend.app.inference.models.krea2_lora import skip_unit_scaling
            skip_unit_scaling(inner)
            torch.testing.assert_close(model(x),actual,rtol=0,atol=0)
            relative=(actual.float()-baseline.float()).square().mean().sqrt()/baseline.float().square().mean().sqrt()
            self.assertLess(float(relative),.04)
            with model.disable_adapter():without=model(x)
            self.assertGreater(float((actual-without).abs().max()),.01)
            with patch.object(native_fp8,'native_linear',return_value=None):
                torch.testing.assert_close(model(x),baseline,rtol=0,atol=0)
            with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                mapping.limit_cpu_cache(8192);mapping.offload_cache();mapping.budget=0
                evicted=model(x)
            torch.testing.assert_close(evicted,actual,rtol=0,atol=0)
            self.assertTrue(mapping.native_cpu_cache)
            self.assertEqual(mapping.native_cpu_cache['0.base_layer.weight'][0].dtype,torch.float8_e4m3fn)
            with patch.object(native_fp8,'native_linear',side_effect=RuntimeError('kernel failure')):
                with self.assertRaisesRegex(RuntimeError,'kernel failure'):model(x)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):model(x)
