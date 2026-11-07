import tempfile
import unittest
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from safetensors.torch import save_file
from backend.app.pipeline_cache import PipelineCache


class SDLoraReuseTests(unittest.TestCase):
    def test_disabled_dora_replaced_by_plain_lora_updates_both_clip_encoders_and_unet(self):
        config=CLIPTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,projection_dim=32)
        pipe=StableDiffusionXLPipeline(
            unet=UNet2DConditionModel(block_out_channels=(32,),down_block_types=('DownBlock2D',),up_block_types=('UpBlock2D',),cross_attention_dim=32,layers_per_block=1),
            vae=AutoencoderKL(block_out_channels=(32,)),scheduler=DDIMScheduler(),
            text_encoder=CLIPTextModel(config),text_encoder_2=CLIPTextModelWithProjection(config),tokenizer=None,tokenizer_2=None)
        targets={'unet':'time_embedding.linear_1'}
        for component in ('text_encoder','text_encoder_2'):
            targets[component]=next(name for name,_ in getattr(pipe,component).named_modules() if name.endswith('self_attn.q_proj'))
        inputs={};original={};baseline={};plain={};dora={}
        for component,target in targets.items():
            layer=getattr(pipe,component).get_submodule(target)
            inputs[component]=torch.randn(2,layer.in_features)
            original[component]=layer.weight.detach().clone()
            baseline[component]=layer(inputs[component]).detach().clone()
            a=torch.full((2,layer.in_features),.1);b=torch.full((layer.out_features,2),.1)
            prefix=component+'.'+target
            plain[prefix+'.lora_A.weight']=a;plain[prefix+'.lora_B.weight']=b
            if component=='unet':
                dora={prefix+'.lora_A.weight':a,prefix+'.lora_B.weight':b,
                      prefix+'.lora_magnitude_vector':layer.weight.detach().norm(dim=1)*1.2}
        cache=PipelineCache();cache.pipe=pipe;cache.suspended=True
        plan=dict(runner='sdxl_txt2img',model_path='base',settings={'prediction_type':'epsilon'},loras=[])
        cache.key=cache.identity(plan,(768,1024))
        with tempfile.TemporaryDirectory() as folder:
            dora_path=Path(folder)/'dora.safetensors';plain_path=Path(folder)/'plain.safetensors'
            save_file(dora,dora_path);save_file(plain,plain_path)
            for label,path,strength in [('disabled-dora',dora_path,0.),('plain',plain_path,.7),('removed',None,0.)]:
                choices=[({'weight':strength},path)] if path else []
                key=cache.identity(plan|{'loras':[{'path':str(path)}] if path else []},(768,1024))
                self.assertIs(cache.get(key,'same prompt',replace_loras=choices),pipe)
                for component,target in targets.items():
                    layer=getattr(pipe,component).get_submodule(target)
                    base=layer.get_base_layer() if hasattr(layer,'get_base_layer') else layer
                    self.assertTrue(torch.equal(base.weight,original[component]))
                    expected=baseline[component]
                    if label=='plain':
                        prefix=component+'.'+target
                        expected=expected+.7*torch.nn.functional.linear(torch.nn.functional.linear(inputs[component],plain[prefix+'.lora_A.weight']),plain[prefix+'.lora_B.weight'])
                    torch.testing.assert_close(layer(inputs[component]),expected,rtol=1e-5,atol=1e-6,msg=f'{label}/{component}')
