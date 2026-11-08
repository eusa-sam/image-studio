import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import torch
from safetensors.torch import save_file
from backend.app.inference.conditioning.lora import validate_lora, apply_loras, set_lora_weights
from backend.app.models import Settings, preflight, MODEL_ROOT


class LoraTests(unittest.TestCase):
    def test_invalid_numeric_weights_are_rejected_before_adapter_injection(self):
        layer=torch.nn.Linear(3,4)
        pipe=SimpleNamespace(unet=SimpleNamespace(config={},get_submodule=lambda name:layer),
            lora_state_dict=lambda state,**kw:(state,None),load_lora_weights=Mock(),set_adapters=Mock())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'invalid.safetensors'
            for value in (float('nan'),float('inf'),float('-inf')):
                state={'unet.layer.lora_A.weight':torch.full((2,3),value),'unet.layer.lora_B.weight':torch.ones(4,2)}
                save_file(state,path)
                with self.assertRaisesRegex(ValueError,'nonfinite'):apply_loras(pipe,[({'weight':1},path)])
                pipe.load_lora_weights.assert_not_called()
            state={'unet.layer.lora_A.weight':torch.ones(2,3,dtype=torch.int8),'unet.layer.lora_B.weight':torch.ones(4,2)}
            with self.assertRaisesRegex(ValueError,'Invalid'):validate_lora(pipe,None,raw=state)
            state={'unet.layer.lora_A.weight':torch.empty(0,3),'unet.layer.lora_B.weight':torch.empty(4,0)}
            with self.assertRaisesRegex(ValueError,'shape'):validate_lora(pipe,None,raw=state)

    def test_alpha_is_checked_and_conversion_cannot_consume_the_validated_dictionary(self):
        layer=torch.nn.Linear(3,4)
        def convert(state,**kwargs):
            state.pop('layer.alpha');return state,None
        pipe=SimpleNamespace(unet=SimpleNamespace(config={},get_submodule=lambda name:layer),lora_state_dict=convert)
        state={'unet.layer.lora_A.weight':torch.ones(2,3),'unet.layer.lora_B.weight':torch.ones(4,2)}
        for alpha in (torch.tensor(float('nan')),torch.tensor(float('inf')),torch.ones(2)):
            with self.assertRaisesRegex(ValueError,'alpha'):validate_lora(pipe,None,raw=dict(state,**{'layer.alpha':alpha}))
        state['layer.alpha']=torch.tensor(2)
        self.assertEqual(validate_lora(pipe,None,raw=state),1)
        self.assertIn('layer.alpha',state)

    def test_spatial_kohya_conv_adapter_matches_unfused_reference(self):
        from diffusers import StableDiffusionPipeline,UNet2DConditionModel,AutoencoderKL,DDIMScheduler
        unet=UNet2DConditionModel(block_out_channels=(32,),down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',),cross_attention_dim=32,layers_per_block=1,sample_size=64)
        pipe=StableDiffusionPipeline(unet=unet,vae=AutoencoderKL(block_out_channels=(32,)),
            scheduler=DDIMScheduler(),text_encoder=None,tokenizer=None,safety_checker=None,
            feature_extractor=None,requires_safety_checker=False)
        base=unet.conv_in
        weight=base.weight.detach().clone();bias=base.bias.detach().clone()
        generator=torch.Generator().manual_seed(29)
        inputs=torch.randn(1,4,8,8,generator=generator)
        a=torch.randn(2,4,3,3,generator=generator)*.1
        b=torch.randn(32,2,1,1,generator=generator)*.1
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'conv.safetensors'
            save_file({'lora_unet_conv_in.lora_down.weight':a,'lora_unet_conv_in.lora_up.weight':b,
                       'lora_unet_conv_in.alpha':torch.tensor(2.)},path)
            self.assertEqual(validate_lora(pipe,path),1)
            second=Path(folder)/'second.safetensors'
            save_file({'lora_unet_conv_in.lora_down.weight':a,'lora_unet_conv_in.lora_up.weight':-b,
                       'lora_unet_conv_in.alpha':torch.tensor(2.)},second)
            selections=[({'weight':1},path),({'weight':.2},second)];apply_loras(pipe,selections)
            original=torch.nn.functional.conv2d(inputs,weight,bias,padding=1)
            delta=torch.nn.functional.conv2d(torch.nn.functional.conv2d(inputs,a,padding=1),b)
            for scale in (1,.4,0,1):
                selections[0][0]['weight']=scale;set_lora_weights(pipe,selections)
                torch.testing.assert_close(unet.conv_in(inputs),original+(scale-.2)*delta,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(unet.conv_in.base_layer.weight,weight,rtol=0,atol=0)

    def test_conv_adapter_rejects_unrepresentable_layer_settings(self):
        for kwargs in ({'groups':2},{'dilation':2},{'padding_mode':'reflect'}):
            layer=torch.nn.Conv2d(4,4,3,**kwargs)
            pipe=SimpleNamespace(unet=SimpleNamespace(config={},get_submodule=lambda name:layer),lora_state_dict=lambda state,**kw:(state,None))
            state={'unet.layer.lora_A.weight':torch.ones(2,layer.weight.shape[1],3,3),'unet.layer.lora_B.weight':torch.ones(4,2,1,1)}
            with self.assertRaisesRegex(ValueError,'Conv2d'):validate_lora(pipe,None,raw=state)

    def test_real_two_adapter_strength_updates_preserve_base_and_do_not_accumulate(self):
        from diffusers import StableDiffusionPipeline,UNet2DConditionModel,AutoencoderKL,DDIMScheduler
        unet=UNet2DConditionModel(block_out_channels=(32,),down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',),cross_attention_dim=32,layers_per_block=1,sample_size=64)
        pipe=StableDiffusionPipeline(unet=unet,vae=AutoencoderKL(block_out_channels=(32,)),
            scheduler=DDIMScheduler(),text_encoder=None,tokenizer=None,safety_checker=None,
            feature_extractor=None,requires_safety_checker=False)
        target='mid_block.attentions.0.transformer_blocks.0.attn1.to_q'
        base=unet.get_submodule(target).weight.detach().clone()
        generator=torch.Generator().manual_seed(17)
        inputs=torch.randn(2,base.shape[1],generator=generator)
        deltas=[]
        with tempfile.TemporaryDirectory() as folder:
            selections=[]
            for index in range(2):
                a=torch.randn(2,base.shape[1],generator=generator)*.1
                b=torch.randn(base.shape[0],2,generator=generator)*.1
                path=Path(folder)/f'{index}.safetensors'
                save_file({'unet.'+target+'.lora_A.weight':a,'unet.'+target+'.lora_B.weight':b},path)
                selections.append(({'weight':0},path));deltas.append(b@a)
            apply_loras(pipe,selections)
            for weights in ((0,.7),(1,.3),(0,0),(1,.3)):
                for (selection,_),weight in zip(selections,weights):selection['weight']=weight
                set_lora_weights(pipe,selections)
                actual=unet.get_submodule(target)(inputs)
                expected=torch.nn.functional.linear(inputs,base+sum(weight*delta for weight,delta in zip(weights,deltas)))
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(unet.get_submodule(target).base_layer.weight,base,rtol=0,atol=0)

    def test_anima_original_keys_and_conditioner_targets(self):
        from diffusers.loaders import AnimaLoraLoaderMixin
        layer=torch.nn.Linear(3,4)
        component=SimpleNamespace(get_submodule=lambda name:layer)
        pipe=SimpleNamespace(transformer=component,text_conditioner=component,lora_state_dict=AnimaLoraLoaderMixin.lora_state_dict)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'anima.safetensors'
            state={}
            for target in ('diffusion_model.blocks.0.self_attn.q_proj','diffusion_model.llm_adapter.blocks.0.cross_attn.k_proj'):
                state[target+'.lora_A.weight']=torch.ones(2,3)
                state[target+'.lora_B.weight']=torch.ones(4,2)
            save_file(state,path);self.assertEqual(validate_lora(pipe,path),2)
            state['text_encoder.bad.lora_A.weight']=torch.ones(2,3);save_file(state,path)
            with self.assertRaisesRegex(ValueError,'구성요소'):validate_lora(pipe,path)

    def test_legacy_text_model_prefix_matches_flattened_clip(self):
        encoder=torch.nn.Module();encoder.add_module('projection',torch.nn.Linear(3,4))
        pipe=SimpleNamespace(unet=SimpleNamespace(config={}),text_encoder=encoder,lora_state_dict=lambda state,**kw:(state,None))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'clip.safetensors'
            save_file({'text_encoder.text_model.projection.lora_A.weight':torch.ones(2,3),
                       'text_encoder.text_model.projection.lora_B.weight':torch.ones(4,2)},path)
            self.assertEqual(validate_lora(pipe,path),1)

    def test_pointwise_conv_lora_checks_rank_channels_and_kernel(self):
        layer=torch.nn.Conv2d(3,4,1)
        pipe=SimpleNamespace(unet=SimpleNamespace(config={},get_submodule=lambda name:layer),lora_state_dict=lambda state,**kw:(state,None))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'conv.safetensors'
            state={'unet.layer.lora_A.weight':torch.ones(2,3,1,1),'unet.layer.lora_B.weight':torch.ones(4,2,1,1)}
            save_file(state,path);self.assertEqual(validate_lora(pipe,path),1)
            state['unet.layer.lora_B.weight']=torch.ones(5,2,1,1);save_file(state,path)
            with self.assertRaisesRegex(ValueError,'shape'):validate_lora(pipe,path)
            state['unet.layer.lora_A.weight']=torch.ones(2,3,3,3);save_file(state,path)
            with self.assertRaisesRegex(ValueError,'1x1'):validate_lora(pipe,path)

    def test_all_targets_and_shapes_checked_before_loading(self):
        layer=torch.nn.Linear(3,4)
        pipe=SimpleNamespace(unet=SimpleNamespace(config={},get_submodule=lambda name:layer),lora_state_dict=lambda state,**kw:(state,None))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'lora.safetensors'
            state={'unet.layer.lora_A.weight':torch.ones(2,3),'unet.layer.lora_B.weight':torch.ones(4,2)}
            save_file(state,path)
            self.assertEqual(validate_lora(pipe,path),1)
            old={'unet.layer.lora.down.weight':state['unet.layer.lora_A.weight'],
                 'unet.layer.lora.up.weight':state['unet.layer.lora_B.weight']}
            save_file(old,path)
            self.assertEqual(validate_lora(pipe,path),1)
            state['unet.layer.lora_A.weight']=torch.ones(2,5);save_file(state,path)
            with self.assertRaisesRegex(ValueError,'shape'):
                validate_lora(pipe,path)
            del state['unet.layer.lora_A.weight'];save_file(state,path)
            with self.assertRaisesRegex(ValueError,'A/B'):
                validate_lora(pipe,path)

    def test_multiple_weights_passed_without_fusion(self):
        pipe=Mock(unet=SimpleNamespace(peft_config={}))
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in ('a.safetensors','b.safetensors')]
            for path in paths:save_file({'fixture':torch.ones(1)},path)
            with patch('backend.app.inference.conditioning.lora.validate_lora') as validate,patch('safetensors.torch.load_file',return_value={}) as read:
                apply_loras(pipe,[({'weight':0.0},paths[0]),({'weight':0.7},paths[1])])
        self.assertEqual(read.call_count,2)
        self.assertEqual(validate.call_count,2)
        pipe.set_adapters.assert_called_once_with(['lora_0','lora_1'],adapter_weights=[0.0,0.7])
        pipe.fuse_lora.assert_not_called()

    def test_preflight_rejects_unknown_duplicate_and_out_of_range(self):
        models=[dict(id='base',available=True,path='base',fingerprint={}),dict(id='lora',component='lora',path='lora',fingerprint={},profile={})]
        with patch('backend.app.models.scan',return_value=models), patch('pathlib.Path.is_file',return_value=True):
            with self.assertRaisesRegex(ValueError,'프로필'):
                preflight({'model':'base','loras':[{'model':'sd1'}]})
            with self.assertRaisesRegex(ValueError,'중복'):
                preflight({'model':'base','loras':[{'model':'lora'},{'model':'lora'}]})
            plan=preflight({'model':'base','loras':[{'model':'lora','weight':0.5}]})
            self.assertEqual(plan['loras'][0]['weight'],0.5)
        with self.assertRaises(ValueError):
            Settings(model='base',loras=[{'model':'lora','weight':3.0}])
