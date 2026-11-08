import unittest

import torch
from diffusers import (AutoencoderKL, DDIMScheduler, StableDiffusionPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
                       UNet2DConditionModel)

from backend.app.inference.runtime.memory_policy import convert_pipeline, fresh_scheduler


class PipelineConversionTests(unittest.TestCase):
    def test_scheduler_reset_discards_history_and_preserves_effective_config(self):
        from diffusers import PNDMScheduler
        scheduler=PNDMScheduler(skip_prk_steps=True)
        scheduler.register_to_config(steps_offset=1)
        scheduler.set_timesteps(20)
        scheduler.counter=9
        scheduler.ets=[torch.ones(1)]
        scheduler.cur_sample=torch.ones(1)
        scheduler.alphas_cumprod=scheduler.alphas_cumprod.half()
        fresh=fresh_scheduler(scheduler)
        self.assertEqual(fresh.config.steps_offset,1)
        self.assertEqual(fresh.counter,0)
        self.assertEqual(fresh.ets,[])
        self.assertIsNone(fresh.cur_sample)
        self.assertEqual(fresh.alphas_cumprod.dtype,torch.float32)
        self.assertEqual(scheduler.counter,9)

    def test_image_flows_preserve_shared_half_precision_weights(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                unet = UNet2DConditionModel(
                    block_out_channels=(32,), down_block_types=('DownBlock2D',),
                    up_block_types=('UpBlock2D',), cross_attention_dim=32,
                    layers_per_block=1, sample_size=64).to(dtype=dtype)
                vae = AutoencoderKL(block_out_channels=(32,), latent_channels=4).to(dtype=dtype)
                pipe = StableDiffusionPipeline(
                    unet=unet, vae=vae, scheduler=DDIMScheduler(),
                    text_encoder=None, tokenizer=None, safety_checker=None,
                    feature_extractor=None, requires_safety_checker=False)
                parameter = next(unet.parameters())
                address = parameter.data_ptr()
                expected = parameter.clone()
                for cls in (StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline):
                    converted = convert_pipeline(pipe, cls)
                    self.assertIs(converted.unet, unet)
                    self.assertIs(converted.vae, vae)
                    self.assertIsNot(converted.scheduler,pipe.scheduler)
                    public=lambda scheduler:{key:value for key,value in scheduler.config.items() if not key.startswith('_')}
                    self.assertEqual(public(converted.scheduler),public(pipe.scheduler))
                    self.assertEqual(converted.unet.dtype, dtype)
                    self.assertEqual(converted.vae.dtype, dtype)
                    self.assertEqual(next(unet.parameters()).data_ptr(), address)
                    torch.testing.assert_close(next(unet.parameters()), expected, rtol=0, atol=0)
