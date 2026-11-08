from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from backend.app.inference.conditioning.text_encoder import replace_encoder
from backend.app.models import preflight


class EncoderTests(unittest.TestCase):
    def test_target_precision_overflow_is_rejected_before_any_weight_changes(self):
        encoder=torch.nn.Linear(2,3).half()
        original={key:value.clone() for key,value in encoder.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'clip.safetensors'
            save_file({'weight':torch.ones(3,2),'bias':torch.full((3,),1e30)},str(path))
            with self.assertRaisesRegex(ValueError,'precision'):replace_encoder(encoder,path)
            for key,value in encoder.state_dict().items():self.assertTrue(torch.equal(value,original[key]))
            save_file({'weight':torch.full((3,2),1.125),'bias':torch.full((3,),65504.)},str(path))
            replace_encoder(encoder,path)
            self.assertTrue(torch.equal(encoder.weight,torch.full((3,2),1.125,dtype=torch.float16)))
            self.assertTrue(torch.equal(encoder.bias,torch.full((3,),65504.,dtype=torch.float16)))

    def test_complete_shapes_dtype_and_legacy_prefix(self):
        encoder=torch.nn.Linear(2,3)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'clip.safetensors'
            save_file({'text_model.weight':torch.ones(3,2),'text_model.bias':torch.zeros(3)},str(path))
            replace_encoder(encoder,path);self.assertTrue(torch.equal(encoder.weight,torch.ones(3,2)))
            for weights in ({'weight':torch.ones(3,2)}, {'weight':torch.ones(2,3),'bias':torch.zeros(3)},
                            {'weight':torch.full((3,2),float('nan')),'bias':torch.zeros(3)}):
                save_file(weights,str(path))
                with self.assertRaises(ValueError):replace_encoder(encoder,path)
            self.assertTrue(torch.equal(encoder.weight,torch.ones(3,2)))

    def test_family_and_slot_preflight(self):
        rows=[dict(id='base',family='sdxl_candidate',available=True,path='base',fingerprint={}),
              dict(id='clip',component='text_encoder',profile={'family':'sdxl','slot':1},path='clip',fingerprint={})]
        with patch('backend.app.models.scan',return_value=rows),patch('pathlib.Path.is_file',return_value=True):
            self.assertIn('clip',preflight({'model':'base','clip':'clip'})['components'])
            with self.assertRaisesRegex(ValueError,'슬롯'):preflight({'model':'base','clip2':'clip'})
            rows[0]['family']='sd1_candidate'
            with self.assertRaisesRegex(ValueError,'계열'):preflight({'model':'base','clip':'clip'})
