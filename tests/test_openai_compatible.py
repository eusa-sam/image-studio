import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from backend.app.openai_compatible import OpenAICompatible
from backend.app.ollama import OllamaError
from backend.app.main import create_app
from backend.app.jobs import JobQueue
from backend.app.chat import ChatService,ChatInput
from backend.app.storage import SessionStore


class CompatibleTests(unittest.TestCase):
    def test_format_images_and_standard_options(self):
        client=OpenAICompatible('https://example.test/v1','test-key')
        request=dict(model='model',messages=[dict(role='system',content='rules'),dict(role='user',content='look',images=['encoded'])],
                     format={'type':'object'},options=dict(temperature=.5,top_k=30,num_ctx=4096,num_predict=123,num_gpu=-1),keep_alive=0)
        body=client.body(request)
        self.assertEqual(body['max_tokens'],123)
        self.assertEqual(body['temperature'],.5)
        self.assertEqual(body['messages'][1]['content'][1]['image_url']['url'],'data:image/png;base64,encoded')
        self.assertNotIn('options',body);self.assertNotIn('keep_alive',body)
        client.token_field='max_completion_tokens';client.response_format='text'
        self.assertEqual(client.body(request)['max_completion_tokens'],123)
        self.assertNotIn('response_format',client.body(request))
        client.response_format='json_schema'
        self.assertEqual(client.body(request)['response_format']['json_schema']['schema'],request['format'])

    def test_fragmented_sse_reasoning_usage_and_errors(self):
        class Fragments(httpx.AsyncByteStream):
            async def __aiter__(self):
                data=(b': heartbeat\r\n\r\ndata: {"choices":[{"delta":{"reasoning_content":"thought"},"finish_reason":null}]}\r\n\r\n'
                      b'data: {"choices":[]}\n\ndata: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                for offset in range(0,len(data),7):yield data[offset:offset+7]
        async def run():
            requests=[]
            def handle(request):
                requests.append(request);return httpx.Response(200,stream=Fragments())
            transport=httpx.MockTransport(handle)
            client=OpenAICompatible('https://example.test/v1','test-key')
            request=dict(model='m',messages=[dict(role='system',content='schema')],format={},options={},stream=True)
            http=httpx.AsyncClient(transport=transport)
            with patch('backend.app.openai_compatible.httpx.AsyncClient',return_value=http):
                chunks=[chunk async for chunk in client.stream_chat(request)]
            self.assertEqual([c['message']['content'] for c in chunks],['','answer'])
            self.assertEqual(chunks[0]['message']['thinking'],'thought')
            self.assertTrue(chunks[-1]['done'])
            self.assertEqual(requests[0].headers['Authorization'],'Bearer test-key')
            self.assertEqual(str(requests[0].url),'https://example.test/v1/chat/completions')
        asyncio.run(run())
        for code in (401,429,500):
            client=OpenAICompatible('https://example.test/v1','never-return-this')
            http=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(code,json={'error':'never-return-this'})))
            with patch('backend.app.openai_compatible.httpx.Client',return_value=http),self.assertRaises(OllamaError) as error:
                client.models()
            self.assertNotIn('never-return-this',str(error.exception))
            self.assertIn(str(code),str(error.exception))

    def test_remote_chat_preserves_image_residency_and_validates_proposal(self):
        with tempfile.TemporaryDirectory(dir='data') as directory,SessionStore(directory) as store:
            session=store.create_session('remote')
            queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:self.fail('Remote request unloaded image model'))
            service=ChatService(store,queue);service.compatible=OpenAICompatible('https://example.test/v1')
            body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='custom',
                           message='hello',positive='keep',negative='also keep',settings={'model':'image'},image_mode='none')
            content=json.dumps(dict(reply='hello',update=False,positive='',negative=''))
            response={'choices':[{'message':{'content':content},'finish_reason':'stop'}]}
            http=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=response)))
            with patch('backend.app.openai_compatible.httpx.Client',return_value=http),patch('backend.app.models.scan',return_value=[]):
                turn=service.send(body)
            self.assertEqual(turn['status'],'SUCCEEDED');self.assertEqual(turn['positive'],'keep')
            self.assertFalse(queue.gpu.locked())

    def test_settings_key_roundtrip_redaction_and_backup_exclusion(self):
        with tempfile.TemporaryDirectory(dir='data') as directory,patch.object(JobQueue,'_loop'):
            root=Path(directory).resolve();(root/'models').mkdir()
            with TestClient(create_app(root/'store',port=18765),base_url='http://127.0.0.1:18765') as client:
                settings=client.get('/api/settings').json()
                settings.update(model_path=str(root/'models'),output_path=str(root/'outputs'),llm_provider='openai_compatible',openai_base_url='https://example.test/v1',openai_api_key='test-secret')
                result=client.post('/api/settings',json=settings)
                self.assertEqual(result.status_code,200,result.text)
                self.assertNotIn('test-secret',result.text)
                current=result.json();self.assertTrue(current['openai_key_registered'])
                self.assertNotIn('test-secret',(root/'store/settings/application.json').read_text())
                self.assertEqual(client.app.state.chat.client.api_key,'test-secret')
                current=client.post('/api/settings',json=current).json()
                self.assertTrue(current['openai_key_registered'])
                replace=client.app.state.store._replace
                def deny_settings(path,content):
                    if path=='settings/application.json':raise OSError('disk full')
                    return replace(path,content)
                before=list((root/'store/credentials').glob('*.json'))
                with patch.object(client.app.state.store,'_replace',side_effect=deny_settings):
                    failed=client.post('/api/settings',json={**current,'openai_api_key':'failed-new-key'})
                self.assertEqual(failed.status_code,422)
                self.assertEqual(client.get('/api/settings').json(),current)
                self.assertEqual(client.app.state.chat.client.api_key,'test-secret')
                self.assertEqual(list((root/'store/credentials').glob('*.json')),before)
                bad=client.post('/api/settings',json={**current,'openai_api_key':'secret\ninvalid'})
                self.assertEqual(bad.status_code,422);self.assertNotIn('secret',bad.text)
                result=client.post('/api/settings',json={**current,'openai_base_url':'https://different.test/v1'})
                self.assertFalse(result.json()['openai_key_registered'])
                current=result.json()
                result=client.post('/api/settings',json={**current,'openai_api_key':'new-secret'})
                current=result.json()
                client.post('/api/backups',json={})
                for path in (root/'store/backups').rglob('*'):
                    if path.is_file():self.assertNotIn(b'new-secret',path.read_bytes())
                result=client.post('/api/settings',json={**current,'openai_api_key':''})
                self.assertFalse(result.json()['openai_key_registered'])
