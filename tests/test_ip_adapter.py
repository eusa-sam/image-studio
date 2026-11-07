import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
from backend.app.ip_adapter import load_vision


class VisionTests(unittest.TestCase):
    def test_full_loading_optional_derived_ids_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=CLIPVisionConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,image_size=16,patch_size=8,projection_dim=4)
            config.to_json_file(root/'config.json');model=CLIPVisionModelWithProjection(config)
            state={k:v.detach().clone() for k,v in model.state_dict().items()};position='vision_model.embeddings.position_ids'
            for ids in (False,True):
                data=dict(state)
                if ids:data[position]=torch.arange(5).view(1,-1)
                save_file(data,root/'model.safetensors');loaded=load_vision(root/'model.safetensors',root/'config.json')
                self.assertEqual(set(loaded.state_dict()),set(state))
                for key,value in loaded.state_dict().items():torch.testing.assert_close(value,state[key].half())
                self.assertFalse(any(p.is_meta for p in list(loaded.parameters())+list(loaded.buffers())))
            key=next(iter(state))
            for changed in ({position:torch.zeros(1,5,dtype=torch.long)}, {key:torch.full_like(state[key],float('nan'))}, {key:torch.zeros(1)}, {'unexpected':torch.zeros(1)}):
                save_file(state|changed,root/'model.safetensors')
                with self.assertRaises(ValueError):load_vision(root/'model.safetensors',root/'config.json')


    def test_adapter_checks_all_targets_before_registering(self):
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from backend.app.ip_adapter import attach_adapter
        weights={'image_proj.proj.weight':torch.zeros(8192,1024,dtype=torch.float16),
                 'image_proj.proj.bias':torch.zeros(8192),'image_proj.norm.weight':torch.ones(2048),
                 'image_proj.norm.bias':torch.zeros(2048),'ip_adapter.1.to_k_ip.weight':torch.zeros(2,2048),
                 'ip_adapter.1.to_v_ip.weight':torch.zeros(2,2048)}
        attention=SimpleNamespace(to_k=SimpleNamespace(weight=torch.zeros(2,2048)),to_v=SimpleNamespace(weight=torch.zeros(2,2048)))
        pipe=SimpleNamespace(unet=SimpleNamespace(dtype=torch.float16,config=SimpleNamespace(cross_attention_dim=2048),
             attn_processors={'block.attn1.processor':None,'block.attn2.processor':None},get_submodule=lambda name:attention),
             register_modules=Mock(),load_ip_adapter=Mock(),set_ip_adapter_scale=Mock())
        with patch('backend.app.ip_adapter.load_vision',return_value=SimpleNamespace(config=SimpleNamespace(projection_dim=1024))),patch('safetensors.torch.load_file',return_value=weights):
            attach_adapter(pipe,'adapter','vision','config',.7)
            pipe.set_ip_adapter_scale.assert_called_once_with(.7)
            self.assertTrue(pipe.load_ip_adapter.call_args.kwargs['local_files_only'])
            pipe.register_modules.reset_mock()
            weights['ip_adapter.1.to_k_ip.weight']=torch.zeros(3,2048)
            with self.assertRaisesRegex(ValueError,'shape mismatch'):attach_adapter(pipe,'adapter','vision','config',.7)
            pipe.register_modules.assert_not_called()
            weights['ip_adapter.1.to_k_ip.weight']=torch.full((2,2048),1e10)
            with self.assertRaisesRegex(ValueError,'overflow'):attach_adapter(pipe,'adapter','vision','config',.7)
            for strength in (-1,float('nan'),True,3):
                with self.assertRaises(ValueError):attach_adapter(pipe,'adapter','vision','config',strength)
