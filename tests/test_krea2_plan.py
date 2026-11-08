import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.models import preflight, scan
from backend.app.inference.models.krea2 import verify_config, count_prompt_tokens


class KreaPlanTests(unittest.TestCase):
    def test_int8_profile_discovery_still_requires_explicit_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'checkpoint').mkdir()
            path=root/'checkpoint/renamed.safetensors';path.write_bytes(b'fixture')
            with patch('backend.app.models.MODEL_ROOT',root),patch('backend.app.models.inspect',return_value={'family':'krea2','dtypes':{'I8':224}}):
                self.assertFalse(scan()[0]['available'])
                profile=dict(schema_version=1,variant='turbo',sha256='a'*64,text_encoder='encoder',vae='vae',source='fixture')
                path.with_name(path.name+'.krea2.json').write_text(json.dumps(profile))
                row=scan()[0]
                self.assertTrue(row['available']);self.assertEqual(row['krea2'],profile)

    def test_defaults_overrides_and_unsupported_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'configs/krea2';config.mkdir(parents=True)
            rows=[]
            for name in ('text_encoder/config.json','vae/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json'):
                target=config/name;target.parent.mkdir(exist_ok=True);target.write_bytes(b'{}')
                rows.append(dict(path=name,sha256=hashlib.sha256(b'{}').hexdigest()))
            (config/'manifest.json').write_text(json.dumps({'files':rows}))
            inventory=[dict(id='base',family='krea2',available=True,path='checkpoint/base',fingerprint={},
                krea2=dict(text_encoder='text_encoders/qwen',vae='vae/qwen',sha256='a'*64))]
            for id,kind,path in [('text','text_encoder','text_encoders/qwen'),('vae','vae','vae/qwen'),('other','text_encoder','text_encoders/other')]:
                inventory.append(dict(id=id,component=kind,path=path,fingerprint={},profile=dict(family='krea2',sha256='b'*64)))
            inventory.append(dict(id='detector',component='detector',path='detector/face',fingerprint={},profile=dict(family='yolov8',target='face',sha256='d'*64)))
            settings=dict(model='base',prediction_type='flow',guidance=0.)
            inventory.append(dict(id='lora',component='lora',path='lora/style',fingerprint={},profile=dict(family='krea2',sha256='c'*64)))
            with patch('backend.app.models.scan',return_value=inventory),patch('backend.app.models.MODEL_ROOT',root):
                plan=preflight(settings)
                self.assertEqual(plan['runner'],'krea2_txt2img')
                self.assertIn('face_detector',preflight(settings|{'face_detector':'detector'})['components'])
                with self.assertRaises(ValueError):preflight(settings|{'hand_detector':'detector'})
                with self.assertRaises(ValueError):preflight(settings|{'face_detector':'detector','detail_steps':4,'detail_strength':.1})
                editing=preflight(settings|{'flow':'img2img','source_job':'11111111-1111-4111-8111-111111111111'})
                self.assertEqual(editing['runner'],'krea2_img2img')
                masked=settings|{'flow':'inpaint','source_job':'11111111-1111-4111-8111-111111111111','mask_id':'22222222-2222-4222-8222-222222222222'}
                self.assertEqual(preflight(masked)['runner'],'krea2_inpaint')
                for mask in (None,'invalid'):
                    with self.assertRaises(ValueError):preflight(masked|{'mask_id':mask})
                self.assertEqual(plan['components']['clip']['path'],'text_encoders/qwen')
                self.assertEqual(preflight(settings|{'clip':'other'})['components']['clip']['path'],'text_encoders/other')
                self.assertEqual(preflight(settings|{'loras':[{'model':'lora','weight':.5}]})['loras'][0]['weight'],.5)
                with self.assertRaisesRegex(ValueError,'중복'):
                    preflight(settings|{'loras':[{'model':'lora'},{'model':'lora'}]})
                for change in ({'guidance':1.},{'prediction_type':'epsilon'},{'prompt_mode':'weighted'},
                               {'clip2':'text'},{'loras':[{'model':'text'}]},{'face_detector':'face'}, {'clip':'missing'}):
                    with self.subTest(change=change),self.assertRaises(ValueError):preflight(settings|change)
                (config/'vae/config.json').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'config changed'):preflight(settings)

    def test_manifest_cannot_omit_required_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifest.json').write_text('{"files":[]}')
            with self.assertRaisesRegex(ValueError,'incomplete'):verify_config(root)

    def test_turbo_tokens_account_for_template_and_ignore_negative(self):
        class Tokenizer:
            def __call__(self,text):return {'input_ids':list(range(34+508))}
        result=count_prompt_tokens('long prompt','unused',Tokenizer())
        self.assertTrue(result['truncated'])
        self.assertFalse(result['negative_used'])
        self.assertEqual(result['encoders'][0]['positive'],508)
        self.assertEqual(result['encoders'][0]['limit'],507)
