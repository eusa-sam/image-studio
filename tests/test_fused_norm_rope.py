import unittest
import torch
from diffusers.models.normalization import RMSNorm
from backend.app.inference.ops.fused_norm_rope import native_norm_rope, FusedProcessor


class FusedNormRopeTests(unittest.TestCase):
    def test_install_preserves_custom_processors_and_projection_storage(self):
        from types import SimpleNamespace
        from diffusers.models.attention_processor import Attention,AttnProcessor2_0
        from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
        from backend.app.inference.ops.fused_norm_rope import install
        original=CosmosAttnProcessor2_0();custom=AttnProcessor2_0()
        layers=torch.nn.ModuleList([Attention(query_dim=64,heads=2,dim_head=32,processor=p) for p in (original,custom)])
        pointers=[p.data_ptr() for p in layers.parameters()]
        install(SimpleNamespace(transformer=layers),'anima')
        self.assertIsInstance(layers[0].processor,FusedProcessor)
        self.assertIs(layers[1].processor,custom)
        self.assertEqual(pointers,[p.data_ptr() for p in layers.parameters()])

    def test_processor_fallback_preserves_layout_mask_and_output(self):
        from diffusers.models.attention_processor import Attention
        from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor
        from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
        with torch.inference_mode():
            for interleaved in (False,True):
                processor=ZSingleStreamAttnProcessor() if interleaved else CosmosAttnProcessor2_0()
                attn=Attention(query_dim=64,heads=2,dim_head=32,qk_norm='rms_norm',processor=processor)
                x=torch.randn(2,9,64)
                phase=torch.randn(2,9,16) if interleaved else torch.randn(9,32)
                kwargs={'freqs_cis':torch.polar(torch.ones_like(phase),phase)} if interleaved else {'image_rotary_emb':(phase.cos(),phase.sin())}
                mask=torch.ones((2,9) if interleaved else (2,1,9,9),dtype=torch.bool);mask[...,-2:]=False
                kwargs['attention_mask']=mask
                expected=processor(attn,x,**kwargs);actual=FusedProcessor(processor,interleaved)(attn,x,**kwargs)
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                kwargs={} if interleaved else {'encoder_hidden_states':torch.randn(2,5,64)}
                torch.testing.assert_close(FusedProcessor(processor,interleaved)(attn,x,**kwargs),processor(attn,x,**kwargs),atol=1e-6,rtol=1e-6)

    def test_cpu_falls_back(self):
        self.assertIsNone(native_norm_rope(torch.ones(1,4,2,128),RMSNorm(128,eps=1e-6),None,False))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_matches_intermediate_rounding_for_both_layouts(self):
        from diffusers.models.embeddings import apply_rotary_emb
        from unittest.mock import patch
        with torch.inference_mode():
            for width in (32,64,128,256):
                torch.manual_seed(43)
                x=torch.randn(2,11,3,width,device='cuda',dtype=torch.bfloat16)
                norm=RMSNorm(width,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
                norm.weight.copy_(torch.randn_like(norm.weight))
                angle=torch.randn(11,width,device='cuda');frequencies=(angle.cos(),angle.sin())
                expected=apply_rotary_emb(norm(x),frequencies,use_real_unbind_dim=-2,sequence_dim=1)
                actual=native_norm_rope(x,norm,frequencies,False)
                self.assertIsNotNone(actual);torch.testing.assert_close(actual,expected,atol=.0625,rtol=.02)
                phase=torch.randn(2,11,width//2,device='cuda');complex_freq=torch.polar(torch.ones_like(phase),phase)
                expected=torch.view_as_real(torch.view_as_complex(norm(x).float().reshape(2,11,3,-1,2))*complex_freq.unsqueeze(2)).flatten(3).to(x.dtype)
                actual=native_norm_rope(x,norm,complex_freq,True)
                self.assertIsNotNone(actual);torch.testing.assert_close(actual,expected,atol=.0625,rtol=.02)
                actual=native_norm_rope(x,norm,None,False)
                self.assertIsNotNone(actual);torch.testing.assert_close(actual,norm(x),atol=.0625,rtol=.02)
                x[:,:,0,0]=float('nan')
                self.assertTrue(torch.equal(torch.isnan(native_norm_rope(x,norm,None,False)),torch.isnan(norm(x))))
