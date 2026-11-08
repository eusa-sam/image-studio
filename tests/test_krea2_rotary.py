import os,unittest
from unittest.mock import patch
import torch
from diffusers.models.transformers import transformer_krea2
from backend.app.inference.models import krea2_rotary


class KreaRotaryTests(unittest.TestCase):
    def test_cpu_and_other_layouts_fall_back_and_restore_after_error(self):
        original=transformer_krea2.apply_rotary_emb
        x=torch.randn(2,3,5,8);freqs=(torch.randn(5,8),torch.randn(5,8))
        expected=original(x,freqs)
        with self.assertRaisesRegex(RuntimeError,'body'):
            with krea2_rotary.fused_rotary():
                torch.testing.assert_close(transformer_krea2.apply_rotary_emb(x,freqs),expected,rtol=0,atol=0)
                self.assertIsNone(krea2_rotary.native_rotary(x,freqs))
                raise RuntimeError('body')
        self.assertIs(transformer_krea2.apply_rotary_emb,original)

    @unittest.skipUnless(torch.cuda.is_available() and os.name=='nt','Windows CUDA required')
    def test_fused_formula_is_bit_exact_for_batches_heads_and_boundary_values(self):
        original=transformer_krea2.apply_rotary_emb
        with torch.inference_mode():
            for shape in ((2,17,3,32),(1,31,1,128),(1,1,4,8)):
                x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
                x.flatten()[:4]=torch.tensor([0.,-0.,1e20,-1e20],device='cuda',dtype=torch.bfloat16)
                angles=torch.randn(shape[1],shape[3],device='cuda')
                freqs=(angles.cos(),angles.sin())
                expected=original(x,freqs,sequence_dim=1)
                actual=krea2_rotary.native_rotary(x,freqs)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual.view(torch.int16),expected.view(torch.int16),rtol=0,atol=0)
                with krea2_rotary.fused_rotary():
                    torch.testing.assert_close(transformer_krea2.apply_rotary_emb(x,freqs,sequence_dim=1),expected,rtol=0,atol=0)
                self.assertIsNone(krea2_rotary.native_rotary(x.float(),freqs))
            with patch.object(krea2_rotary,'_kernel',None),patch.object(krea2_rotary,'_unavailable',False),patch('backend.app.inference.ops.cuda_decode.Kernel',side_effect=OSError('no compiler')):
                with self.assertWarnsRegex(RuntimeWarning,'using Torch'):
                    self.assertIsNone(krea2_rotary.native_rotary(x,freqs))
