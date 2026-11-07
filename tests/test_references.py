import base64
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.chat import ChatInput,ChatService
from backend.app.references import save_reference,read_reference
from backend.app.storage import SessionStore


def picture(color,format='PNG'):
    data=io.BytesIO();Image.new('RGB',(64,96),color).save(data,format=format);return data.getvalue()


class ReferenceTests(unittest.TestCase):
    def test_missing_attachment_returns_stream_error_before_inference(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
                session=store.create_session('missing')
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='local',message='reference',positive='',negative='',settings={'model':'m'},attachments=[uuid4()])
                events=[event async for event in ChatService(store,SimpleNamespace()).stream(body)]
                self.assertEqual(events[-1]['type'],'error')
                self.assertEqual(store.chats(session),[])
        asyncio.run(run())

    def test_generated_and_user_images_have_distinct_sources_and_order(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('sources')
            original=picture('red','JPEG');reference=save_reference(store,session,'reference.jpg',original)
            folder=Path(directory)/'sessions'/session/'references'/reference['id']
            self.assertEqual((folder/'original.jpg').read_bytes(),original)
            job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='blue generated result',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
            store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':picture('blue')})
            body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='local',message='use reference color',positive='current prompt',negative='',settings={'model':'m'},attachments=[reference['id']])
            service=ChatService(store,SimpleNamespace());payload=service._payload(body)
            request=service._request(body,payload,{'capabilities':['vision']})
            system=request['messages'][0]['content'];images=request['messages'][-1]['images']
            self.assertIn('Image 1 is APP-GENERATED OUTPUT',system)
            self.assertIn('Image 2 is USER-UPLOADED REFERENCE 1',system)
            self.assertIn('2 images total; 1 user-uploaded references',system)
            self.assertEqual(len(images),2)
            for encoded,channel in zip(images,(2,0)):
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:self.assertGreater(image.getpixel((0,0))[channel],240)
            store.begin_chat(session,str(body.request_id),payload)
            turn=store.chats(session)[0]
            self.assertEqual(turn['user_references'][0]['id'],reference['id'])
            self.assertEqual(turn['image_context']['job_id'],job)
            persisted=(Path(directory)/'sessions'/session/'messages'/str(body.request_id)/'context/settings.json').read_text()
            self.assertNotIn(images[1],persisted)
            with self.assertRaisesRegex(ValueError,'vision'):service._request(body,payload,{'capabilities':['completion']})
            body.image_mode='none';request=service._request(body,service._payload(body),{'capabilities':['vision']})
            self.assertIn('Image 1 is USER-UPLOADED REFERENCE 1',request['messages'][0]['content'])
            body.attachments*=2
            with self.assertRaisesRegex(ValueError,'중복'):service._payload(body)
            store.finish_chat(session,str(body.request_id),proposal=dict(reply='Earlier reference was red.',update=False,positive=body.positive,negative='',memory=''))
            body.attachments=[]
            request=service._request(body,service._payload(body),{'capabilities':['vision']})
            self.assertIn('0 images total; 0 user-uploaded references',request['messages'][0]['content'])
            self.assertNotIn('images',request['messages'][-1])
            self.assertIn('Earlier reference was red.',str(request['messages'][1:-1]))
            self.assertIn('not current attachments',request['messages'][0]['content'])
            body.image_mode='latest'
            request=service._request(body,service._payload(body),{'capabilities':['vision']})
            self.assertIn('1 images total; 0 user-uploaded references',request['messages'][0]['content'])
            self.assertEqual(len(request['messages'][-1]['images']),1)
            (folder/'preview.png').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'해시'):read_reference(store,session,reference['id'])

    def test_upload_api_rejects_bad_files_and_cross_workspace_reads(self):
        with tempfile.TemporaryDirectory() as directory,TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
            a=client.post('/api/sessions',json={}).json()['ref'];b=client.post('/api/sessions',json={}).json()['ref']
            body=dict(session=a,filename='photo.png',data=base64.b64encode(picture('green')).decode())
            response=client.post('/api/references',json=body);self.assertEqual(response.status_code,200)
            reference=response.json()['id']
            self.assertEqual(client.get('/api/references/'+reference,params={'session':a}).status_code,200)
            self.assertEqual(client.get('/api/references/'+reference,params={'session':b}).status_code,404)
            for data in ('not base64',base64.b64encode(b'not an image').decode()):
                self.assertEqual(client.post('/api/references',json=dict(body,data=data)).status_code,422)
