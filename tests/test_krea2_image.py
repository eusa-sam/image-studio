import math
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline
from backend.app.krea2 import image_arguments


class KreaImageTests(unittest.TestCase):
    def test_resume_rebudgets_existing_cache_from_current_vram_and_ram(self):
        from backend.app.krea2 import resume_pipeline
        gib=1024**3
        maps=[SimpleNamespace(cpu_bytes=2*gib,limit_cpu_cache=Mock()),SimpleNamespace(cpu_bytes=0,limit_cpu_cache=Mock()),SimpleNamespace(cpu_bytes=0,limit_cpu_cache=Mock())]
        pipe=SimpleNamespace(weight_maps=maps,vae=torch.nn.Linear(2,2,bias=False),memory_policy={'weight_bytes':35*gib,'int8_compute':'native'})
        with patch('backend.app.memory_policy.available_vram',return_value=(6*gib,16*gib,{'driver_free_bytes':6*gib})),patch('backend.app.memory_policy.available_ram',return_value=gib):
            resume_pipeline(pipe,(768,1024))
        self.assertEqual(pipe.memory_policy['mode'],'partial')
        self.assertEqual(pipe.memory_policy['int8_compute'],'native')
        self.assertEqual(maps[0].budget,2*gib);self.assertEqual(maps[2].budget,0)
        maps[0].limit_cpu_cache.assert_called_once_with(0);maps[2].limit_cpu_cache.assert_called_once_with(0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_suspended_cpu_conditioning_is_reused_on_execution_device(self):
        from backend.app.krea2 import prompt_arguments
        cache={'woman':{'prompt_embeds':torch.ones(1,4,2,3),'prompt_embeds_mask':torch.tensor([[True,True,False,False]])}}
        pipe=SimpleNamespace(_execution_device='cuda:0',encode_prompt=Mock(side_effect=AssertionError('must reuse conditioning')))
        result=prompt_arguments(pipe,'woman',cache)
        self.assertEqual(result['prompt_embeds'].device.type,'cuda')
        self.assertEqual(result['prompt_embeds_mask'].dtype,torch.bool)
        self.assertEqual(cache['woman']['prompt_embeds'].device.type,'cpu')
        torch.testing.assert_close(result['prompt_embeds'].cpu(),cache['woman']['prompt_embeds'])

    def test_encoder_temporary_budget_is_scoped_and_failure_does_not_cache(self):
        from backend.app.krea2 import prompt_arguments
        encoder=SimpleNamespace(transient_gpu_budget=0)
        def encode(*args,**kwargs):
            self.assertEqual(encoder.transient_gpu_budget,1024**3)
            raise ValueError('encoder failed')
        pipe=SimpleNamespace(memory_policy={'mode':'partial'},weight_maps=[None,encoder],_execution_device='cuda:0',encode_prompt=encode)
        cache={}
        with patch('backend.app.memory_policy.available_vram',return_value=(3*1024**3,16*1024**3,{})),patch('torch.cuda.memory_reserved',return_value=0),patch('torch.cuda.memory_allocated',return_value=0):
            with self.assertRaisesRegex(ValueError,'encoder failed'):prompt_arguments(pipe,'woman',cache)
        self.assertEqual(encoder.transient_gpu_budget,0);self.assertEqual(cache,{})

    def test_prompt_cache_preserves_exact_text_mask_and_is_scoped_to_one_job(self):
        from backend.app.krea2 import prompt_arguments
        calls=[]
        def hidden(prompt, length, device):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(prompt)
            return torch.full((1,4,2,3),float(len(prompt))),torch.tensor([[True,True,False,False]])
        pipe=SimpleNamespace(_execution_device=torch.device('cpu'),get_text_hidden_states=hidden)
        pipe.encode_prompt=lambda *args,**kwargs:Krea2Pipeline.encode_prompt(pipe,*args,**kwargs)
        cache={}
        prompts=['woman','detailed face, natural eyes, woman','natural hand, anatomically correct fingers, woman']
        for prompt in prompts+[prompts[1],prompts[2]]:
            args=prompt_arguments(pipe,prompt,cache)
            with torch.inference_mode():
                encoded,mask=pipe.encode_prompt(None,**args)
            torch.testing.assert_close(encoded,torch.full((1,4,2,3),float(len(prompt))),rtol=0,atol=0)
            torch.testing.assert_close(mask,torch.tensor([[True,True,False,False]]),rtol=0,atol=0)
            self.assertNotIn('prompt',args)
        self.assertEqual(calls,prompts)
        prompt_arguments(pipe,'woman ',cache)
        self.assertEqual(calls[-1],'woman ')
        prompt_arguments(pipe,'woman',{})
        self.assertEqual(calls[-1],'woman')
        self.assertEqual(len(calls),5)


    def test_strength_uses_shifted_sigma_normalized_image_and_short_schedule(self):
        captured=[]
        def encode(pixels):
            captured.append(pixels.shape)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.full((1,16,1,2,2),3.)))
        pipe=SimpleNamespace(_execution_device=torch.device('cpu'),patch_size=2,
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
            vae=SimpleNamespace(dtype=torch.float32,config=SimpleNamespace(latents_mean=[1.]*16,latents_std=[2.]*16),encode=encode),
            image_processor=SimpleNamespace(preprocess=lambda image,**kwargs:torch.zeros(1,3,16,16)))
        pipe._pack_latents=lambda *args:Krea2Pipeline._pack_latents(pipe,*args)
        for strength,count in ((.5,4),(1.,8)):
            args,info=image_arguments(pipe,None,16,16,8,strength,torch.Generator().manual_seed(42))
            raw=count/8
            sigma=math.exp(1.15)*raw/(1+(math.exp(1.15)-1)*raw)
            self.assertAlmostEqual(info['start_sigma'],sigma,places=6)
            self.assertEqual(len(args['sigmas']),count)
            self.assertEqual(info['denoising_steps'],count)
            noise=torch.randn((1,16,2,2),generator=torch.Generator().manual_seed(42))
            expected=(1-sigma)*torch.ones_like(noise)+sigma*noise
            torch.testing.assert_close(args['latents'],pipe._pack_latents(expected,*expected.shape))
        self.assertEqual(captured,[(1,3,1,16,16)]*2)
        from PIL import Image
        mask=Image.new('L',(16,16),0);mask.paste(255,(0,0,8,16))
        calls=[]
        args,info=image_arguments(pipe,None,16,16,8,.5,torch.Generator().manual_seed(42),
            mask=mask,progress=lambda p,i,t,k:(calls.append(i) or k))
        pipe.scheduler.set_timesteps(sigmas=args['sigmas'],mu=1.15,device='cpu')
        packed_mask=torch.zeros(1,16,2,2);packed_mask[:,:,:,0]=1
        packed_mask=pipe._pack_latents(packed_mask,*packed_mask.shape)
        noise=torch.randn((1,16,2,2),generator=torch.Generator().manual_seed(42))
        for step in (0,3):
            generated=torch.full_like(args['latents'],7)
            actual=args['callback_on_step_end'](pipe,step,None,{'latents':generated})['latents']
            sigma=pipe.scheduler.sigmas[step+1]
            expected=(1-sigma)*torch.ones_like(noise)+sigma*noise
            source=pipe._pack_latents(expected,*expected.shape)
            torch.testing.assert_close(actual,packed_mask*generated+(1-packed_mask)*source)
        self.assertEqual(calls,[0,3])
        self.assertEqual(info['mask_mode'],'per_step_latent_blend_and_pixel_composite')
        with self.assertRaises(ValueError):image_arguments(pipe,None,16,16,8,.01,torch.Generator())
