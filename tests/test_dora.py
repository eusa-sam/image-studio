import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from backend.app.lora import apply_loras, set_lora_weights, validate_lora


class DoraTests(unittest.TestCase):
    def pipeline(self):
        unet=UNet2DConditionModel(block_out_channels=(32,),down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',),cross_attention_dim=32,layers_per_block=1,sample_size=64)
        return StableDiffusionPipeline(unet=unet,vae=AutoencoderKL(block_out_channels=(32,)),
            scheduler=DDIMScheduler(clip_sample=False,steps_offset=1),text_encoder=None,tokenizer=None,
            safety_checker=None,feature_extractor=None,requires_safety_checker=False)

    def state(self,layer,target,dora=True):
        a=torch.full((2,*layer.weight.shape[1:]),.01)
        b=torch.full((layer.weight.shape[0],2,*([1,1] if a.ndim==4 else [])),.02)
        state={f'unet.{target}.lora_A.weight':a,f'unet.{target}.lora_B.weight':b}
        if dora:
            magnitude=layer.weight.detach().flatten(1).norm(dim=1)*1.2
            if a.ndim==4:magnitude=magnitude.reshape(1,-1,1,1)
            state[f'unet.{target}.lora_magnitude_vector']=magnitude
        return state

    def test_linear_and_conv_match_dora_math_and_zero_reactivation_removal(self):
        for target in ('time_embedding.linear_1','conv_in'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as root:
                pipe=self.pipeline();layer=pipe.unet.get_submodule(target)
                weight=layer.weight.detach().clone();bias=layer.bias.detach().clone()
                x=torch.randn(2,layer.in_features) if weight.ndim==2 else torch.randn(1,4,8,8)
                baseline=layer(x).detach();state=self.state(layer,target)
                path=Path(root)/'dora.safetensors';save_file(state,path)
                selected=[({'weight':1.},path)];apply_loras(pipe,selected)
                a,b=state[f'unet.{target}.lora_A.weight'],state[f'unet.{target}.lora_B.weight']
                delta=(b.flatten(1)@a.flatten(1)).reshape_as(weight)
                combined=weight+delta
                magnitude=state[f'unet.{target}.lora_magnitude_vector'].flatten()
                normalized=combined*(magnitude/combined.flatten(1).norm(dim=1)).reshape(-1,*([1]*(weight.ndim-1)))
                expected=torch.nn.functional.linear(x,normalized,bias) if weight.ndim==2 else torch.nn.functional.conv2d(x,normalized,bias,padding=1)
                for strength in (1.,0.,1.):
                    selected[0][0]['weight']=strength;set_lora_weights(pipe,selected)
                    actual=pipe.unet.get_submodule(target)(x)
                    torch.testing.assert_close(actual,expected if strength else baseline,rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(pipe.unet.get_submodule(target).base_layer.weight,weight,rtol=0,atol=0)
                pipe.unload_lora_weights()
                torch.testing.assert_close(pipe.unet.get_submodule(target)(x),baseline,rtol=0,atol=0)

    def test_zero_dora_does_not_affect_active_plain_adapter(self):
        with tempfile.TemporaryDirectory() as root:
            pipe=self.pipeline();target='time_embedding.linear_1';layer=pipe.unet.get_submodule(target)
            base=layer.weight.detach().clone();bias=layer.bias.detach().clone();x=torch.randn(2,layer.in_features)
            selections=[]
            for index,dora in enumerate((True,False)):
                state=self.state(layer,target,dora);path=Path(root)/f'{index}.safetensors';save_file(state,path)
                selections.append(({'weight':0. if dora else .7},path))
            apply_loras(pipe,selections)
            delta=state[f'unet.{target}.lora_B.weight']@state[f'unet.{target}.lora_A.weight']
            expected=torch.nn.functional.linear(x,base+.7*delta,bias)
            torch.testing.assert_close(pipe.unet.get_submodule(target)(x),expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(pipe.unet.active_adapters(),['lora_1'])

    def test_invalid_magnitude_and_metadata_fail_before_injection(self):
        pipe=self.pipeline();target='time_embedding.linear_1';state=self.state(pipe.unet.get_submodule(target),target)
        key=f'unet.{target}.lora_magnitude_vector'
        for value in (torch.ones(1),torch.full_like(state[key],float('nan')),torch.ones_like(state[key],dtype=torch.int8)):
            with self.assertRaises(ValueError):validate_lora(pipe,None,raw=dict(state,**{key:value}))
        with self.assertRaisesRegex(ValueError,'A/B'):validate_lora(pipe,None,raw={key:state[key]})
        with tempfile.TemporaryDirectory() as root:
            state.pop(key);path=Path(root)/'missing.safetensors'
            save_file(state,path,metadata={'lora_adapter_metadata':json.dumps({'unet.use_dora':True})})
            with self.assertRaisesRegex(ValueError,'requires magnitude'):apply_loras(pipe,[({'weight':1.},path)])
            self.assertFalse(hasattr(pipe.unet.get_submodule(target),'base_layer'))
