from tokenizer_fixture import local_tokenizers
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.inspect_models import inspect
from backend.app.models import preflight
from backend.app.prompt_tokens import count_tokens


class SD1Tests(unittest.TestCase):
    def test_header_uses_dimensions_not_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'renamed.safetensors'
            shapes={'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight':[49408,768],
                    'model.diffusion_model.input_blocks.0.0.weight':[320,4,3,3],
                    'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight':[320,768]}
            def write():
                header=json.dumps({k:dict(shape=v,dtype='F16',data_offsets=[0,0]) for k,v in shapes.items()}).encode()
                path.write_bytes(struct.pack('<Q',len(header))+header)
            write();self.assertEqual(inspect(path,Path(folder))['family'],'sd1_candidate')
            shapes['model.diffusion_model.input_blocks.0.0.weight']=[320,9,3,3]
            write();self.assertNotEqual(inspect(path,Path(folder))['family'],'sd1_candidate')

    def test_cross_family_components_are_rejected(self):
        rows=[dict(id='base',family='sd1_candidate',available=True,path='base',fingerprint={}),
              dict(id='lora',component='lora',profile={'family':'sdxl'},path='lora',fingerprint={}),
              dict(id='vae',component='vae',profile={'family':'sdxl'},path='vae',fingerprint={})]
        with patch('backend.app.models.scan',return_value=rows),patch('pathlib.Path.is_file',return_value=True):
            self.assertEqual(preflight({'model':'base'})['runner'],'sd1_txt2img')
            for settings in ({'vae':'vae'},{'loras':[{'model':'lora'}]}):
                with self.assertRaisesRegex(ValueError,'계열'):
                    preflight({'model':'base',**settings})
            rows[1]['profile']['family']='sd1'
            self.assertEqual(len(preflight({'model':'base','loras':[{'model':'lora'}]})['loras']),1)

    @local_tokenizers()
    def test_sd1_counts_only_its_single_encoder(self):
        report=count_tokens('a woman','blur',family='sd1')
        self.assertEqual(len(report['encoders']),1)
        self.assertEqual(report['encoders'][0]['encoder'],'CLIP-L')
