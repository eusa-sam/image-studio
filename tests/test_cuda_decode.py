import unittest
from unittest.mock import patch
import torch
from backend.app import cuda_decode
from backend.app.quantized_weights import undo_convrot


class FusedDecodeTests(unittest.TestCase):
    def test_unavailable_runtime_falls_back_once(self):
        with patch.object(cuda_decode,'_kernel',None),patch.object(cuda_decode,'_unavailable',False),patch.object(cuda_decode.os,'name','nt'),patch.object(cuda_decode,'Kernel',side_effect=OSError('missing')) as kernel:
            with self.assertWarnsRegex(RuntimeWarning,'using Torch'):self.assertIsNone(cuda_decode.decode_int8(None,None,16))
            self.assertIsNone(cuda_decode.decode_int8(None,None,16));kernel.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_exact_groups_boundary_blocks_and_stream_ordering(self):
        kernel=cuda_decode.Kernel()
        torch.manual_seed(42)
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for group in (16,64,256):
                for rows in (1,7,33):
                    raw=torch.randint(-128,128,(rows,group*3),device='cuda',dtype=torch.int8)
                    scale=torch.rand(rows,1,device='cuda')*.1
                    expected=(undo_convrot(raw.float(),group)*scale).bfloat16()
                    actual=kernel(raw,scale,group)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            with self.assertRaises(ValueError):kernel(raw,scale,True)
            with self.assertRaises(ValueError):kernel(raw,scale.flatten(),group)
            with self.assertRaises(ValueError):kernel(raw.float(),scale,group)
        stream.synchronize()
