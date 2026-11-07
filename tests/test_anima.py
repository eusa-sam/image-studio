import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from backend.app.anima import verify_bundle,split_single_state
from backend.app.models import preflight
from unittest.mock import patch


class AnimaTests(unittest.TestCase):
    def test_single_file_prefixes_and_duplicate_keys(self):
        transformer,conditioner=split_single_state({'model.diffusion_model.blocks.0.weight':1,'net.llm_adapter.embed.weight':2})
        self.assertEqual(transformer,{'net.blocks.0.weight':1})
        self.assertEqual(conditioner,{'embed.weight':2})
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            split_single_state({'model.diffusion_model.blocks.0.weight':1,'net.blocks.0.weight':2})
        with self.assertRaisesRegex(ValueError,'Missing'):split_single_state({'net.blocks.0.weight':1})
        with self.assertRaisesRegex(ValueError,'Unknown'):split_single_state({'unexpected':1})

    def test_flow_prediction_and_cross_family_lora(self):
        rows=[dict(id='anima',family='anima',available=True,path='bundle',fingerprint={}),
              dict(id='sd1',family='sd1_candidate',available=True,path='sd1',fingerprint={}),
              dict(id='lora',component='lora',profile={'family':'sd1'},path='lora',fingerprint={})]
        with patch('backend.app.models.scan',return_value=rows),patch('pathlib.Path.is_file',return_value=True):
            self.assertEqual(preflight({'model':'anima','prediction_type':'flow'})['runner'],'anima_txt2img')
            for settings in ({'model':'anima'},{'model':'sd1','prediction_type':'flow'}):
                with self.assertRaisesRegex(ValueError,'flow'):preflight(settings)
            with self.assertRaisesRegex(ValueError,'계열'):
                preflight({'model':'anima','prediction_type':'flow','loras':[{'model':'lora'}]})
            rows[0].update(anima={'variant':'turbo'},bundle_fingerprint={})
            with self.assertRaisesRegex(ValueError,'Guidance'):
                preflight({'model':'anima','prediction_type':'flow'})
            self.assertEqual(preflight({'model':'anima','prediction_type':'flow','guidance':1.0})['anima']['variant'],'turbo')

    def test_manifest_rejects_escape_duplicate_tamper_and_incomplete_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'bundle';folder.mkdir()
            (root/'outside').write_bytes(b'x');(folder/'inside').write_bytes(b'x')
            sha=hashlib.sha256(b'x').hexdigest()
            for records,error in [([{'path':'../outside','sha256':sha}],'path'),
                                   ([{'path':'inside','sha256':sha}]*2,'duplicate'),
                                   ([{'path':'inside','sha256':'0'*64}],'hash'),
                                   ([{'path':'inside','sha256':sha}],'Incomplete')]:
                (folder/'download_manifest.json').write_text(json.dumps({'files':records}))
                with self.assertRaisesRegex(ValueError,error):verify_bundle(folder)
