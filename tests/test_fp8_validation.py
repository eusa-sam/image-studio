import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from backend.app.quantized_weights import FileWeights,read_tensor


class FP8ValidationTests(unittest.TestCase):
    def test_all_encodings_and_small_scale_preserve_decoded_values(self):
        bits=torch.arange(256,dtype=torch.uint8).reshape(16,16)
        torch.testing.assert_close((bits&127)!=127,torch.isfinite(bits.view(torch.float8_e4m3fn).float()))
        finite=bits.clone();finite[(finite&127)==127]=0
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            with self.subTest(device=device),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'fp8.safetensors'
                self.save(path,finite,.125)
                with patch('backend.app.quantized_weights.read_tensor',side_effect=AssertionError('Validation must not decode')):
                    mapping=FileWeights(self.model(),str(path),lambda x:x,validation_device=device)
                self.assertEqual(mapping.statistics()['fp8_bound_validations'],1)
                torch.testing.assert_close(mapping['0.weight'],(finite.view(torch.float8_e4m3fn).float()*.125).bfloat16(),rtol=0,atol=0)
                for nan in (127,255):
                    invalid=finite.clone();invalid[0,0]=nan;self.save(path,invalid,.125)
                    with self.assertRaisesRegex(ValueError,'Nonfinite'):FileWeights(self.model(),str(path),lambda x:x,validation_device=device)

    def test_large_scale_uses_exact_fallback_and_rejects_real_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fp8.safetensors'
            zeros=torch.zeros((16,16),dtype=torch.uint8)
            self.save(path,zeros,torch.finfo(torch.float32).max)
            with patch('backend.app.quantized_weights.read_tensor',wraps=read_tensor) as decode:
                mapping=FileWeights(self.model(),str(path),lambda x:x)
            self.assertEqual(decode.call_count,1);self.assertEqual(mapping.fp8_bound_validations,0)
            self.assertTrue(torch.equal(mapping['0.weight'],torch.zeros((16,16),dtype=torch.bfloat16)))
            self.save(path,torch.full((16,16),126,dtype=torch.uint8),torch.finfo(torch.float32).max)
            with self.assertRaisesRegex(ValueError,'Nonfinite|overflows'):FileWeights(self.model(),str(path),lambda x:x)
            for scale in (0.,-1.,float('nan'),float('inf')):
                self.save(path,zeros,scale)
                with self.assertRaisesRegex(ValueError,'scale'):FileWeights(self.model(),str(path),lambda x:x)

    def test_fp16_boundary_and_requested_preload_keep_original_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fp8.safetensors';bits=torch.full((16,16),126,dtype=torch.uint8)
            self.save(path,bits,146.)
            mapping=FileWeights(self.model(),str(path),lambda x:x,dtype=torch.float16)
            self.assertEqual(mapping.fp8_bound_validations,1)
            torch.testing.assert_close(mapping['0.weight'],torch.full((16,16),448*146,dtype=torch.float16),rtol=0,atol=0)
            self.save(path,bits,147.)
            with self.assertRaisesRegex(ValueError,'overflows'):FileWeights(self.model(),str(path),lambda x:x,dtype=torch.float16)
            self.save(path,bits,.125)
            with patch('backend.app.quantized_weights.read_tensor',wraps=read_tensor) as decode:
                mapping=FileWeights(self.model(),str(path),lambda x:x,preload_budget=512)
            decode.assert_called_once();self.assertEqual(mapping.fp8_bound_validations,0)

    @staticmethod
    def model():
        with torch.device('meta'):return torch.nn.Sequential(torch.nn.Linear(16,16,bias=False))

    @staticmethod
    def save(path,bits,scale):
        save_file({'0.weight':bits.view(torch.float8_e4m3fn),'0.weight_scale':torch.tensor(scale,dtype=torch.float32),
                   '0.comfy_quant':torch.tensor(list(json.dumps({'format':'float8_e4m3fn'}).encode()),dtype=torch.uint8)},str(path))
