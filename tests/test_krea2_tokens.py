import unittest
import torch
from backend.app.inference.conditioning.krea2_tokens import compact,compact_tokens


class KreaTokenTests(unittest.TestCase):
    def test_masked_middle_is_removed_without_losing_valid_suffix(self):
        generator=torch.Generator().manual_seed(12)
        embeddings=torch.randn(1,8,2,16,generator=generator)
        mask=torch.tensor([[True,True,False,False,False,True,True,False]])
        small,valid,eligible=compact(embeddings,mask)
        self.assertTrue(eligible);self.assertTrue(valid.all())
        torch.testing.assert_close(small,embeddings[:,[0,1,5,6]])
        image=torch.randn(1,3,16,generator=generator)
        full=torch.cat((embeddings.mean(2),image),dim=1)[:,None]
        reduced=torch.cat((small.mean(2),image),dim=1)[:,None]
        full_mask=torch.cat((mask,torch.ones(1,3,dtype=torch.bool)),dim=1)[:,None,None]
        reference=torch.nn.functional.scaled_dot_product_attention(full,full,full,attn_mask=full_mask)[:,:,-3:]
        actual=torch.nn.functional.scaled_dot_product_attention(reduced,reduced,reduced)[:,:,-3:]
        torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
        self.assertFalse(compact(embeddings.repeat(2,1,1,1),mask.repeat(2,1))[2])
        self.assertFalse(compact(embeddings,torch.zeros_like(mask))[2])

    def test_context_preserves_unknown_masks_and_restores_after_failure(self):
        class Transformer(torch.nn.Module):
            def forward(self,encoder_attention_mask):return encoder_attention_mask
        class Pipe:
            do_classifier_free_guidance=False
            transformer=Transformer()
            def encode_prompt(self):return torch.ones(1,4,2,8),torch.tensor([[True,False,False,True]])
        pipe=Pipe()
        with torch.inference_mode(),self.assertRaisesRegex(RuntimeError,'cancel'):
            with compact_tokens(pipe):
                embeddings,mask=pipe.encode_prompt()
                self.assertEqual(embeddings.shape[1],2)
                self.assertIsNone(pipe.transformer(encoder_attention_mask=mask))
                other=mask.clone()
                self.assertIs(pipe.transformer(encoder_attention_mask=other),other)
                raise RuntimeError('cancel')
        self.assertNotIn('encode_prompt',vars(pipe));self.assertFalse(pipe.transformer._forward_pre_hooks)
        pipe.do_classifier_free_guidance=True
        with torch.inference_mode(),compact_tokens(pipe):
            embeddings,mask=pipe.encode_prompt()
            self.assertEqual(embeddings.shape[1],4)
            self.assertIs(pipe.transformer(encoder_attention_mask=mask),mask)
