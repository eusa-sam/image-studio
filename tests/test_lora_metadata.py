import json,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from diffusers import Flux2KleinPipeline,Flux2Transformer2DModel
from backend.app.inference.conditioning.lora import apply_loras

class MetadataTests(unittest.TestCase):
    def pipeline(self):
        transformer=Flux2Transformer2DModel(in_channels=8,num_layers=1,num_single_layers=1,
            attention_head_dim=16,num_attention_heads=2,joint_attention_dim=32,
            timestep_guidance_channels=32,axes_dims_rope=(4,4,4,4),guidance_embeds=False)
        return Flux2KleinPipeline(transformer=transformer,vae=None,scheduler=None,text_encoder=None,tokenizer=None)

    def test_rslora_metadata_matches_native_loader_and_preserves_base(self):
        pipe=self.pipeline();native=self.pipeline();native.transformer.load_state_dict(pipe.transformer.state_dict())
        base=pipe.transformer.x_embedder.weight.detach().clone()
        state={'transformer.x_embedder.lora_A.weight':torch.ones(2,8)*.1,'transformer.x_embedder.lora_B.weight':torch.ones(32,2)*.1}
        metadata={'transformer.r':2,'transformer.lora_alpha':8,'transformer.use_rslora':True,'transformer.target_modules':['x_embedder'],'transformer.init_lora_weights':True}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'style.safetensors';save_file(state,path,metadata={'lora_adapter_metadata':json.dumps(metadata)})
            apply_loras(pipe,[({'weight':.7},path)])
            native.load_lora_weights(str(path),adapter_name='reference',local_files_only=True,use_safetensors=True)
            native.set_adapters(['reference'],adapter_weights=[.7])
            x=torch.ones(3,8)
            torch.testing.assert_close(pipe.transformer.x_embedder(x),native.transformer.x_embedder(x),rtol=0,atol=0)
            expected=torch.nn.functional.linear(x,base+.7*(8/2**.5)*(state['transformer.x_embedder.lora_B.weight']@state['transformer.x_embedder.lora_A.weight']))
            torch.testing.assert_close(pipe.transformer.x_embedder(x),expected,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(pipe.transformer.x_embedder.base_layer.weight,base,rtol=0,atol=0)
            metadata['transformer.use_dora']=True
            save_file(state,path,metadata={'lora_adapter_metadata':json.dumps(metadata)})
            with self.assertRaisesRegex(ValueError,'Unsupported LoRA metadata'):apply_loras(self.pipeline(),[({'weight':1},path)])

    def test_changed_source_fails_after_injection(self):
        pipe=self.pipeline()
        state={'transformer.x_embedder.lora_A.weight':torch.ones(2,8),'transformer.x_embedder.lora_B.weight':torch.ones(32,2)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'style.safetensors';save_file(state,path)
            with patch.object(pipe,'load_lora_weights',side_effect=lambda *a,**kw:os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1000000000))):
                with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):apply_loras(pipe,[({'weight':1},path)])
