import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.inspect_models import inspect
from backend.app.models import preflight
from backend.app.inference.conditioning.prompt_tokens import count_tokens


class SD2Tests(unittest.TestCase):
    def test_openclip_dimensions_and_depth_inpaint_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'renamed.safetensors'
            for channels in (4,5,9):
                shapes={'cond_stage_model.model.token_embedding.weight':[49408,1024],
                        'model.diffusion_model.input_blocks.0.0.weight':[320,channels,3,3],
                        'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight':[320,1024]}
                header=json.dumps({k:dict(shape=v,dtype='F16',data_offsets=[0,0]) for k,v in shapes.items()}).encode()
                path.write_bytes(struct.pack('<Q',len(header))+header)
                self.assertEqual(inspect(path,Path(folder))['family']=='sd2_candidate',channels==4)

    def test_family_routing_and_incompatible_lora(self):
        rows=[dict(id='base',family='sd2_candidate',available=True,path='base',fingerprint={}),
              dict(id='lora',component='lora',profile={'family':'sd1'},path='lora',fingerprint={})]
        with patch('backend.app.models.scan',return_value=rows),patch('pathlib.Path.is_file',return_value=True):
            self.assertEqual(preflight({'model':'base','prediction_type':'v_prediction'})['runner'],'sd2_txt2img')
            with self.assertRaisesRegex(ValueError,'계열'):preflight({'model':'base','loras':[{'model':'lora'}]})

    def test_token_encoder_label(self):
        class Tokenizer:
            model_max_length=77
            def __call__(self,*args,**kwargs):return {'input_ids':[1,2]}
        self.assertEqual(count_tokens('hello','',encoders=[Tokenizer()],family='sd2')['encoders'][0]['encoder'],'OpenCLIP-H')
