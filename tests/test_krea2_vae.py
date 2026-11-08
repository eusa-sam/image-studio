import unittest
from unittest.mock import Mock
import torch
from diffusers import AutoencoderKLQwenImage
from backend.app.inference.models.krea2_vae import blend, vectorized_blending


class VaeBlendTests(unittest.TestCase):
    def test_matches_scalar_operations_at_both_tile_edges(self):
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            for dim,name in ((-2,'blend_v'),(-1,'blend_h')):
                old=lambda a,b,n:getattr(AutoencoderKLQwenImage,name)(None,a,b,n)
                for extent in (0,16,31,64,128,256):
                    with self.subTest(dtype=dtype,dim=dim,extent=extent):
                        shape=[1,3,1,7,7];shape[dim]=max(1,extent)+3
                        a=torch.randn(shape,dtype=dtype);b=torch.randn(shape,dtype=dtype)
                        expected=old(a,b.clone(),extent)
                        actual=blend(a,b.clone(),extent,dim,old)
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_context_restores_class_and_instance_methods_after_failure(self):
        class Vae:
            def blend_v(self,*args):pass
            def blend_h(self,*args):pass
        vae=Vae();custom=Mock();vae.blend_h=custom
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            with vectorized_blending(vae):raise RuntimeError('cancelled')
        self.assertNotIn('blend_v',vars(vae))
        self.assertIs(vae.blend_h,custom)
