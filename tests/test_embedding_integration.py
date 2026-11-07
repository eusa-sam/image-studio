from tokenizer_fixture import local_tokenizers
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import torch
from safetensors.torch import save_file
from fastapi.testclient import TestClient
from backend.app.models import scan,preflight,Settings
from backend.app.main import create_app
from backend.app.pipeline_cache import PipelineCache
from backend.app.worker import verify_components


class EmbeddingIntegrationTests(unittest.TestCase):
    def fixture(self,root):
        path=root/'embeddings/lego.safetensors';path.parent.mkdir()
        save_file({'clip_l':torch.ones(2,768),'clip_g':torch.ones(2,1280)},str(path))
        profile=dict(schema_version=1,kind='embedding',family='sdxl',scale=1,source='fixture',tokens=['<s0>','<s1>'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_name(path.name+'.profile.json').write_text(json.dumps(profile))
        return path

    def test_scan_preflight_hash_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.fixture(root)
            with patch('backend.app.models.MODEL_ROOT',root), local_tokenizers():
                row=scan()[0];self.assertEqual(row['component'],'embedding')
                base=dict(id='base',available=True,family='sdxl_candidate',path='base.safetensors',fingerprint={})
                with patch('backend.app.models.scan',return_value=[base,row]),patch('pathlib.Path.is_file',return_value=True):
                    plan=preflight({'model':'base','embeddings':[row['id']]})
                    for family in ['anima','krea2','flux2','zimage']:
                        base['family']=family
                        with self.assertRaisesRegex(ValueError,'SD1/SD2/SDXL'):preflight({'model':'base','embeddings':[row['id']]})
                    base['family']='sdxl_candidate'
                    with self.assertRaisesRegex(ValueError,'중복'):preflight({'model':'base','embeddings':[row['id'],row['id']]})
                with patch('backend.app.worker.MODEL_ROOT',root):
                    resolved,_=verify_components(plan);self.assertEqual(resolved['embedding0'],path)
                key=PipelineCache.identity(plan,(768,1024))
                self.assertNotEqual(key,PipelineCache.identity(dict(plan,components={}),(768,1024)))
                plan['components']['embedding0']['profile']['tokens']=['<changed>','<s1>']
                self.assertNotEqual(key,PipelineCache.identity(plan,(768,1024)))
                path.write_bytes(path.read_bytes()+b'changed')
                self.assertNotIn('component',scan()[0])
        self.assertEqual(Settings(model='base').embeddings,[])

    def test_token_api_and_saved_snapshot_keep_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);models=root/'models';models.mkdir();self.fixture(models)
            with TestClient(create_app(root/'data'),base_url='http://127.0.0.1:8765') as client,patch('backend.app.models.MODEL_ROOT',models), local_tokenizers():
                row=scan()[0]
                for mode in ['native','weighted']:
                    result=client.post('/api/prompt-tokens',json={'family':'sdxl','embeddings':[row['id']],'positive':'<s0><s1>','negative':'<s1>','mode':mode})
                    self.assertEqual(result.status_code,200,result.text)
                    self.assertTrue(all(x['positive']==2 and x['negative']==1 for x in result.json()['encoders']))
                self.assertEqual(client.post('/api/prompt-tokens',json={'family':'krea2','embeddings':[row['id']]}).status_code,422)
                self.assertEqual(client.post('/api/prompt-tokens',json={'embeddings':['missing']}).status_code,422)
                session=client.post('/api/sessions',json={'title':'embedding'}).json()['ref'];identifier=str(uuid4())
                response=client.post('/api/prompts',json=dict(session=session,prompt_id=identifier,request_id=str(uuid4()),expected_revision=0,positive='<s0><s1>',negative='<s1>',settings={'model':'base','embeddings':[row['id']]},generate=False))
                self.assertEqual(response.status_code,200,response.text)
            with TestClient(create_app(root/'data'),base_url='http://127.0.0.1:8765') as client:
                stored=client.get('/api/prompts/'+identifier,params={'session':session}).json()
                self.assertEqual(stored['settings']['embeddings'],[row['id']])
                self.assertEqual(stored['positive'],'<s0><s1>')
