import unittest
import torch
from backend.app.inference.models.krea2_lora import convert_lora


class KreaLoraTests(unittest.TestCase):
    def test_unit_scaling_keeps_strength_disable_multi_adapter_and_gradient_fallbacks(self):
        from unittest.mock import Mock
        from peft import LoraConfig,get_peft_model
        from backend.app.inference.models.krea2_lora import skip_unit_scaling
        config=LoraConfig(target_modules=['0'],r=2,lora_alpha=2)
        model=get_peft_model(torch.nn.Sequential(torch.nn.Linear(4,4)),config)
        model.add_adapter('other',config)
        layer=model.base_model.model[0]
        for name,value in model.named_parameters():
            if 'lora_' in name:value.data.fill_(.25)
        old=layer.forward;spy=Mock(wraps=old);layer.forward=spy
        skip_unit_scaling(model)
        x=torch.randn(3,4)
        with torch.inference_mode():
            torch.testing.assert_close(layer(x),old(x),rtol=0,atol=0);spy.assert_not_called()
            layer.scaling['default']=0
            torch.testing.assert_close(layer(x),layer.base_layer(x),rtol=0,atol=0);spy.assert_not_called()
            layer.scaling['default']=.3
            torch.testing.assert_close(layer(x),old(x),rtol=0,atol=0);self.assertEqual(spy.call_count,1)
            layer.enable_adapters(False)
            torch.testing.assert_close(layer(x),old(x),rtol=0,atol=0);self.assertEqual(spy.call_count,2)
            layer.enable_adapters(True);layer.set_adapter(['default','other'])
            torch.testing.assert_close(layer(x),old(x),rtol=0,atol=0);self.assertEqual(spy.call_count,3)
        layer.set_adapter('default');layer.scaling['default']=1
        layer(x);self.assertEqual(spy.call_count,4)

    def test_kohya_alpha_and_underscored_modules(self):
        for source,target in [('blocks_0_attn_wq','transformer_blocks.0.attn.to_q'),
                              ('txtfusion_layerwise_blocks_0_mlp_up','text_fusion.layerwise_blocks.0.ff.up'),
                              ('txtfusion_refiner_blocks_1_attn_wo','text_fusion.refiner_blocks.1.attn.to_out.0'),
                              ('txtfusion_projector','text_fusion.projector')]:
            prefix='lora_unet_'+source
            down=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
            up=torch.tensor([[1.,2.],[3.,4.]])
            state={prefix+'.lora_down.weight':down,prefix+'.lora_up.weight':up,prefix+'.alpha':torch.tensor(1.)}
            result=convert_lora(state)
            a=result['transformer.'+target+'.lora_A.weight'];b=result['transformer.'+target+'.lora_B.weight']
            torch.testing.assert_close(b@a,(up@down)*.5)
            torch.testing.assert_close(state[prefix+'.lora_up.weight'],up)

    def test_invalid_pairs_unknown_targets_and_nonfinite_are_rejected(self):
        prefix='lora_unet_blocks_0_attn_wq'
        valid={prefix+'.lora_down.weight':torch.ones(2,3),prefix+'.lora_up.weight':torch.ones(4,2)}
        for invalid in ({prefix+'.alpha':torch.tensor(float('nan'))},
                        {prefix+'.alpha':torch.tensor(-1.)},
                        {prefix+'.alpha':torch.ones(2)},
                        {prefix+'.lora_up.weight':torch.ones(4,3)},
                        {'bad.weight':torch.ones(1)}, {'dora_scale':torch.ones(1)}):
            with self.subTest(invalid=list(invalid)),self.assertRaises(ValueError):convert_lora(valid|invalid)
        with self.assertRaises(ValueError):convert_lora({prefix+'.lora_down.weight':torch.ones(2,3)})
        with self.assertRaises(ValueError):convert_lora({key.replace('attn_wq','unknown'):value for key,value in valid.items()})
