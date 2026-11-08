import unittest
import torch
from backend.app.inference.runtime.offload import prefer_cudnn


@unittest.skipUnless(torch.cuda.is_available() and torch.backends.cudnn.is_available(),'CUDA/cuDNN required')
class KreaAttentionTests(unittest.TestCase):
    def test_cudnn_preserves_masked_attention_for_short_and_long_sequences(self):
        from diffusers.models.transformers.transformer_krea2 import Krea2Attention
        torch.manual_seed(123)
        layer=Krea2Attention(512,4,1).to(device='cuda',dtype=torch.bfloat16).eval()
        with torch.inference_mode():
            for length in (1,127,512,3584,4608):
                x=torch.randn(1,length,512,device='cuda',dtype=torch.bfloat16)
                mask=torch.ones(1,1,1,length,device='cuda',dtype=torch.bool)
                if length>1:mask[:,:,:,-length//3:]=False
                layer.set_attention_backend('native');expected=layer(x,attention_mask=mask)
                with prefer_cudnn():actual=layer(x,attention_mask=mask)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual,expected,rtol=.02,atol=.004)
                perturbed=x.clone();perturbed[:,mask.flatten()==False]*=2
                with prefer_cudnn():changed=layer(perturbed,attention_mask=mask)
                # Changing excluded keys must not affect unmasked queries.
                torch.testing.assert_close(changed[:,mask.flatten()],actual[:,mask.flatten()],rtol=0,atol=0)
