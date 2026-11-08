import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from backend.app.inference.quantization.quantized_weights import FileWeights

class AdapterResidencyTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_budget_reuse_release_and_source_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'base.safetensors'
            save_file({'weight':torch.ones(4,4)},path)
            with torch.device('meta'):model=torch.nn.Linear(4,4,bias=False)
            mapping=FileWeights(model,path,lambda values:values)
            original=torch.arange(16,dtype=torch.float32).reshape(4,4)
            mapping.extra['adapter.weight']=original
            size=original.numel()*original.element_size()
            mapping.budget=size-1
            self.assertIs(mapping['adapter.weight'],original)
            mapping.budget=size;mapping.reserve=128
            with patch('backend.app.inference.quantization.quantized_weights.allocator_free_bytes',return_value=128+size-1):
                self.assertIs(mapping['adapter.weight'],original)
            with patch('backend.app.inference.quantization.quantized_weights.allocator_free_bytes',return_value=128+size):
                first=mapping['adapter.weight']
            self.assertTrue(first.is_cuda)
            self.assertEqual(mapping.cache_bytes,size)
            self.assertEqual(mapping.statistics()['adapter_gpu_bytes'],size)
            torch.testing.assert_close(first.cpu(),original,rtol=0,atol=0)
            with patch('backend.app.inference.quantization.quantized_weights.allocator_free_bytes',side_effect=AssertionError('reuse existing tensor')):
                self.assertIs(mapping['adapter.weight'],first)
            mapping.extra['other.weight']=original.clone()
            self.assertIs(mapping['other.weight'],mapping.extra['other.weight'])
            del first
            mapping.offload_cache()
            self.assertEqual(mapping.cache_bytes,0)
            self.assertFalse(mapping.extra_gpu)
            self.assertIs(mapping.extra['adapter.weight'],original)
            self.assertEqual(mapping.cpu_bytes,0)
            mapping.budget=0
            self.assertIs(mapping['adapter.weight'],original)
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):mapping['adapter.weight']
