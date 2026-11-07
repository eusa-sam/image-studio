import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from backend.app.models import scan, preflight


class InpaintModelTests(unittest.TestCase):
    def test_preparation_rejects_invalid_weights_and_preserves_existing_export(self):
        import torch
        from scripts import prepare_inpaint
        with tempfile.TemporaryDirectory() as directory, patch.object(prepare_inpaint,'ROOT',Path(directory)), patch.object(prepare_inpaint,'prepare'):
            source=Path(directory)/'models/checkpoint/sd-v1-5-inpainting.ckpt'
            source.parent.mkdir(parents=True);source.write_bytes(b'original preserved')
            target=source.with_suffix('.safetensors')
            weights=torch.zeros(320,9,3,3)
            with patch('torch.load',return_value={'state_dict':{'model.diffusion_model.input_blocks.0.0.weight':weights}}) as load:
                weights[0,0,0,0]=float('nan')
                with self.assertRaisesRegex(ValueError,'Non-finite'):prepare_inpaint.main()
                self.assertFalse(target.exists())
                weights[0,0,0,0]=0
                prepare_inpaint.main()
                load.assert_called_with(source,map_location='cpu',weights_only=True)
                original=target.read_bytes()
                prepare_inpaint.main()
                self.assertEqual(target.read_bytes(),original)
                target.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'differs'):prepare_inpaint.main()
                self.assertEqual(target.read_bytes(),b'changed')
                self.assertEqual(source.read_bytes(),b'original preserved')

    def test_nine_channels_require_mask_flow_and_separate_config(self):
        with tempfile.TemporaryDirectory() as directory, patch('backend.app.models.MODEL_ROOT',Path(directory)):
            root=Path(directory)
            path=root/'checkpoint/renamed.safetensors';path.parent.mkdir()
            shapes={'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight':[49408,768],
                    'model.diffusion_model.input_blocks.0.0.weight':[320,9,3,3],
                    'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight':[320,768]}
            def write():
                header=json.dumps({k:dict(dtype='F32',shape=v,data_offsets=[0,4]) for k,v in shapes.items()}).encode()
                path.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
            write()
            model=scan()[0];self.assertTrue(model['available']);self.assertEqual(model['family'],'sd1_inpaint_candidate')
            settings=dict(model=model['id'],source_job=str(uuid4()),mask_id=str(uuid4()))
            for flow in ('txt2img','img2img'):
                with self.assertRaisesRegex(ValueError,'9채널'):preflight(dict(settings,flow=flow))
            settings['flow']='inpaint'
            with self.assertRaisesRegex(ValueError,'마스크'):preflight(dict(settings,mask_id=None))
            with self.assertRaisesRegex(ValueError,'구성 누락'):preflight(settings)
            for name in ('model_index.json','scheduler/scheduler_config.json','unet/config.json','vae/config.json',
                         'text_encoder/config.json','tokenizer/vocab.json','tokenizer/merges.txt'):
                config=root/'configs/sd1-inpaint'/name;config.parent.mkdir(parents=True,exist_ok=True);config.write_text('{}')
            plan=preflight(settings)
            self.assertEqual(plan['runner'],'sd1_inpaint');self.assertEqual(plan['inpaint_channels'],9)
            shapes['model.diffusion_model.input_blocks.0.0.weight']=[320,5,3,3];write()
            self.assertFalse(scan()[0]['available'])
