import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from backend.app.inference.quantization.quantized_weights import FileWeights, read_tensor


class PlainValidationTests(unittest.TestCase):
    def model(self):
        with torch.device('meta'):
            return torch.nn.Linear(2, 2, bias=False)

    def test_plain_values_checked_before_use_and_preload_remains_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            for value, error in ((float('nan'), 'Nonfinite'), (1e10, 'overflows')):
                save_file({'weight': torch.full((2, 2), value)}, path)
                mapping = FileWeights(self.model(), path, lambda x: x,
                                      dtype=torch.float16, defer_numeric_validation=True)
                with self.assertRaisesRegex(ValueError, error):
                    mapping['weight']
                self.assertFalse(mapping.cache)
                for options in ({}, {'defer_numeric_validation': True, 'preload_budget': 32}):
                    with self.assertRaisesRegex(ValueError, error):
                        FileWeights(self.model(), path, lambda x: x, dtype=torch.float16, **options)

    def test_valid_read_is_exact_and_constructor_does_not_materialize(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            expected = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
            save_file({'weight': expected}, path)
            with patch('backend.app.inference.quantization.quantized_weights.read_tensor', wraps=read_tensor) as read:
                mapping = FileWeights(self.model(), path, lambda x: x, defer_numeric_validation=True)
                read.assert_not_called()
                torch.testing.assert_close(mapping['weight'], expected, rtol=0, atol=0)
                self.assertEqual(read.call_count, 1)
            with patch.object(mapping, '_stamp', return_value=None):
                with self.assertRaisesRegex(ValueError, 'COMPONENT_CHANGED'):
                    mapping['weight']

    def test_shape_validation_is_not_deferred(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            save_file({'weight': torch.ones(3, 2)}, path)
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                FileWeights(self.model(), path, lambda x: x, defer_numeric_validation=True)

    def test_quantized_nan_validation_precedes_read_and_default_is_eager(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            with torch.device('meta'):
                model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
            metadata = json.dumps({'format': 'float8_e4m3fn'}).encode()
            save_file({'0.weight': torch.full((2, 2), float('nan')).to(torch.float8_e4m3fn),
                       '0.weight_scale': torch.tensor(1.),
                       '0.comfy_quant': torch.tensor(list(metadata), dtype=torch.uint8)}, path)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                FileWeights(model, path, lambda x: x)
            mapping = FileWeights(model, path, lambda x: x, defer_numeric_validation=True)
            for read in (lambda: mapping['0.weight'], lambda: mapping.linear_packed('0.weight', 'cpu')):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    read()
                self.assertFalse(mapping.cache)
                self.assertFalse(mapping.native_cpu_cache)

    def test_fp8_scale_and_overflow_validation_are_not_deferred(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            for scale in (float('nan'), 0., -1., 1e10):
                with torch.device('meta'):
                    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
                save_file({'0.weight': torch.ones(2, 2).to(torch.float8_e4m3fn),
                           '0.weight_scale': torch.tensor(scale),
                           '0.comfy_quant': torch.tensor(list(json.dumps({'format': 'float8_e4m3fn'}).encode()), dtype=torch.uint8)}, path)
                with self.assertRaises(ValueError):
                    FileWeights(model, path, lambda x: x, dtype=torch.float16, defer_numeric_validation=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fp8_packing_checks_before_publication_and_valid_output_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            for value in (float('nan'), 2.):
                with torch.device('meta'):
                    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
                save_file({'0.weight': torch.full((2, 2), value).to(torch.float8_e4m3fn),
                           '0.weight_scale': torch.tensor(.5),
                           '0.comfy_quant': torch.tensor(list(json.dumps({'format': 'float8_e4m3fn'}).encode()), dtype=torch.uint8)}, path)
                mapping = FileWeights(model, path, lambda x: x, defer_numeric_validation=True)
                mapping.packed_enabled = True
                mapping.budget = 1024
                if value != value:
                    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                        mapping['0.weight']
                    self.assertFalse(mapping.packed_cache)
                    self.assertEqual(mapping.cache_bytes, 0)
                    mapping.packed_enabled = False
                    mapping.transient_gpu_budget = 1024
                    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                        mapping['0.weight']
                    self.assertEqual(mapping.transient_gpu_reads, 0)
                else:
                    torch.testing.assert_close(mapping['0.weight'].cpu(), torch.ones(2, 2, dtype=torch.bfloat16), rtol=0, atol=0)
                    self.assertIn('0.weight', mapping.packed_cache)
                    mapping.packed_cache.clear()
                    mapping.cache_bytes = 0
                    mapping.packed_enabled = False
                    mapping.transient_gpu_budget = 1024
                    result = mapping['0.weight']
                    self.assertEqual(result.device.type, 'cuda')
                    torch.testing.assert_close(result.cpu(), torch.ones(2, 2, dtype=torch.bfloat16), rtol=0, atol=0)
                    self.assertEqual(mapping.transient_gpu_reads, 1)
