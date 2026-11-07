import base64
import io
import tempfile
import unittest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from backend.app.chat import ChatInput,ChatService
from backend.app.storage import SessionStore


class ChatContextTests(unittest.TestCase):
    def test_selected_image_reads_only_its_job_and_preserves_validation(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('selected')
            other=store.create_session('other')
            ids=[]
            image=io.BytesIO();Image.new('RGB',(16,16),'blue').save(image,format='PNG')
            for number in range(12):
                job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive=str(number),negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':image.getvalue()})
                ids.append(job)
            body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='local',message='describe',positive='',negative='',settings={'model':'test'},image_mode='selected',image_job=ids[0])
            service=ChatService(store,SimpleNamespace())
            scope='sessions/'+session+'/jobs'
            store.read_errors[scope]=[{'error':'unrelated job error'}]
            with patch.object(store,'_read_json',wraps=store._read_json) as read:
                payload=service._payload(body)
            self.assertEqual(store.read_errors[scope],[{'error':'unrelated job error'}])
            states=[call.args[0] for call in read.call_args_list if call.args[0].endswith('/state.json')]
            self.assertEqual(len(states),2)  # Selected metadata and artifact integrity gate.
            self.assertTrue(all(ids[0] in path for path in states))
            self.assertEqual(payload['image_context']['job_id'],ids[0])
            body.session=other
            with self.assertRaisesRegex(ValueError,'Workspace'):service._payload(body)
            body.session=session
            store.transition(session,ids[0],('SUCCEEDED',),'FAILED')
            with self.assertRaisesRegex(ValueError,'Workspace'):service._payload(body)
            with self.assertRaises(ValueError):store.jobs(session,job_id='../outside')

    def test_user_output_and_context_settings_are_not_overridden_by_reasoning(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('limits');service=ChatService(store,SimpleNamespace())
            base=dict(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='local',
                message='scene',positive='',negative='',settings={'model':'test'},image_mode='none',think=True)
            info={'capabilities':['thinking'],'thinking':{'values':[False,True]}}
            for limit in (None,-1,32768,131072):
                body=ChatInput(**base,num_predict=limit,num_ctx=65536)
                options=service._request(body,service._payload(body),info)['options']
                if limit is None:self.assertNotIn('num_predict',options)
                else:self.assertEqual(options['num_predict'],limit)
                self.assertEqual(options['num_ctx'],65536)
            body=ChatInput(**base)
            self.assertNotIn('num_ctx',service._request(body,service._payload(body),info)['options'])
            for invalid in (0,-2,1.5,True):
                with self.assertRaises(ValueError):ChatInput(**base,num_predict=invalid)

    def test_model_specific_thinking_and_previous_image_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('context')
            job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='original blue coat',negative='blur',settings={},request_id=str(uuid4()),generate=True)['job_id']
            image=io.BytesIO();Image.new('RGB',(64,96),'blue').save(image,format='PNG')
            store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':image.getvalue()})
            body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='local',message='make it red',positive='current green coat',negative='',settings={'model':'test'},think='xhigh')
            service=ChatService(store,SimpleNamespace())
            payload=service._payload(body)
            info={'capabilities':['completion','vision','thinking'],'thinking':{'values':[False,'low','xhigh'],'default':'low'}}
            request=service._request(body,payload,info)
            self.assertEqual(request['think'],'xhigh')
            self.assertIn('current green coat',request['messages'][0]['content'])
            self.assertIn('original blue coat',request['messages'][0]['content'])
            with Image.open(io.BytesIO(base64.b64decode(request['messages'][-1]['images'][0]))) as result:self.assertEqual(result.size,(64,96))
            self.assertEqual(payload['image_context']['job_id'],job)
            with self.assertRaisesRegex(ValueError,'reasoning'):service._request(body,payload,{'thinking':{'values':[False,True]}})
            body.think=False
            request=service._request(body,payload,{'thinking':{'values':[False]},'capabilities':['completion']})
            self.assertNotIn('images',request['messages'][-1]);self.assertIn('no pixels',request['messages'][0]['content'])
            body.image_mode='none';self.assertNotIn('image_context',service._payload(body))
            body.image_mode='selected';body.image_job=uuid4()
            with self.assertRaisesRegex(ValueError,'Workspace'):service._payload(body)
            body.image_job=job
            payload=service._payload(body)
            store.artifact(session,job,'image.png').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'변경|hash mismatch'):service._request(body,payload,info)
