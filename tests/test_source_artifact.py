import io
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch
from PIL import Image
from backend.app.storage import SessionStore,digest,encode
from backend.app.models import Settings
from backend.app.jobs import execution_plan
from backend.app.inference.conditioning.ip_adapter import reference_image
from backend.app.masks import save_mask,read_mask


def png(size,color):
    out=io.BytesIO();Image.new('RGB',size,color).save(out,format='PNG');return out.getvalue()


class SourceArtifactTests(unittest.TestCase):
    def fixture(self,store,status='SUCCEEDED'):
        session=store.create_session('source')
        job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
        original=png((32,48),'blue');final=png((64,96),'red')
        artifacts={'original.png':original}
        if status=='SUCCEEDED':artifacts['image.png']=final
        store.transition(session,job,('AWAITING_VALIDATION',),status,artifacts=artifacts)
        return session,job,original,final

    def test_original_plan_and_generated_reference_preserve_file_identity(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session,job,original,final=self.fixture(store)
            for artifact,content,size in [('original.png',original,[32,48]),('image.png',final,[64,96])]:
                settings=Settings(model='test',source_job=job,source_artifact=artifact,reference_job=job,reference_artifact=artifact).model_dump()
                with patch('backend.app.jobs.preflight',return_value={'runner':'sdxl_img2img','settings':settings}):plan=execution_plan(store,session,settings)
                self.assertEqual(plan['source']['artifact'],artifact);self.assertEqual(plan['source']['sha256'],digest(content));self.assertEqual(plan['source']['size'],size)
                path,meta=reference_image(store,session,settings)
                self.assertEqual(path.read_bytes(),content);self.assertEqual(meta['kind'],'generated_output');self.assertEqual(meta['artifact'],artifact)
            with self.assertRaises(ValueError):Settings(model='test',source_artifact='../image.png')

    def test_partial_original_reference_is_allowed_but_missing_final_is_not(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session,job,original,_=self.fixture(store,'PARTIAL_FAILED')
            settings=dict(reference_job=job,reference_attachment=None,reference_artifact='original.png')
            path,meta=reference_image(store,session,settings);self.assertEqual(path.read_bytes(),original)
            with self.assertRaises((ValueError,FileNotFoundError)):reference_image(store,session,settings|{'reference_artifact':'image.png'})
            with self.assertRaises(FileNotFoundError):reference_image(store,store.create_session('other'),settings)

    def test_mask_tracks_artifact_dimensions_identity_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session,job,original,final=self.fixture(store)
            with self.assertRaisesRegex(ValueError,'dimensions'):save_mask(store,session,job,png((64,96),'white'),source_artifact='original.png')
            meta=save_mask(store,session,job,png((32,48),'white'),source_artifact='original.png')
            self.assertEqual(meta['source_sha256'],digest(original));self.assertEqual(meta['source_artifact'],'original.png')
            read_mask(store,session,meta['id'],job,'original.png')
            with self.assertRaisesRegex(ValueError,'source mismatch'):read_mask(store,session,meta['id'],job,'image.png')
            legacy=save_mask(store,session,job,png((64,96),'white'));legacy.pop('source_artifact',None)
            store._replace(f'sessions/{session}/masks/{legacy["id"]}/meta.json',encode(legacy))
            read_mask(store,session,legacy['id'],job,'image.png')
            with self.assertRaisesRegex(ValueError,'source mismatch'):read_mask(store,session,legacy['id'],job,'original.png')

    def test_api_mask_and_prompt_settings_survive_restart(self):
        import base64
        from fastapi.testclient import TestClient
        from backend.app.main import create_app
        with tempfile.TemporaryDirectory() as directory:
            with SessionStore(directory) as store:session,job,_,_=self.fixture(store)
            prompt=str(uuid4())
            with TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
                body=dict(session=session,source_job=job,source_artifact='original.png',png=base64.b64encode(png((32,48),'white')).decode())
                response=client.post('/api/masks',json=body);self.assertEqual(response.status_code,200,response.text)
                mask=response.json();self.assertEqual(mask['source_artifact'],'original.png')
                self.assertEqual(client.post('/api/masks',json=body|{'source_artifact':'../image.png'}).status_code,422)
                settings=dict(model='test',flow='inpaint',source_job=job,source_artifact='original.png',mask_id=mask['id'])
                response=client.post('/api/prompts',json=dict(session=session,prompt_id=prompt,request_id=str(uuid4()),expected_revision=0,positive='preserve',negative='',settings=settings))
                self.assertEqual(response.status_code,200,response.text)
            with TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
                self.assertEqual(client.get('/api/masks/'+mask['id'],params={'session':session}).status_code,200)
            with SessionStore(directory) as store:
                saved=store.read_prompt(session,prompt)
                self.assertEqual(saved['settings']['source_artifact'],'original.png')

