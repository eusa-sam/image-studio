import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.jobs import JobQueue
from backend.app.ollama import Ollama, OllamaError
from backend.app.chat import ChatService


class FakeOllama:
    def __init__(self):
        self.calls=[]
        self.released=[]
        self.invalid=False

    def local_model(self, name):
        return {'name':name, 'digest':'test-digest'}, {'capabilities':['completion']}

    def running(self):
        return []

    def image_ready(self, selected_model=None):
        return None

    def request(self, path, body, timeout):
        self.calls.append(body)
        return {'done':True, 'message':{'content':'invalid' if self.invalid else json.dumps(
            {'reply':'수정했습니다.', 'update':True, 'positive':'a blue teapot', 'negative':'blur'})}}

    def release(self, name):
        self.released.append(name)


class ChatTests(unittest.TestCase):
    def test_request_schema_matches_edit_operation_boundaries(self):
        from backend.app.chat import ChatInput, resolve_suggestion
        body=ChatInput(**self.body)
        with patch('backend.app.models.scan',return_value=[]):
            request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        variants=request['format']['$defs']['PromptEdit']['anyOf']
        self.assertEqual(len(variants),3)
        for variant,operation in zip(variants,('replace','append','prepend')):
            self.assertEqual(variant['properties']['action'],{'type':'string','const':operation})
            self.assertIn('action',variant['required'])
            self.assertFalse(variant['additionalProperties'])
            self.assertEqual(sorted(variant['properties']),['action','field','source','text'])
            old=variant['properties']['source']
            self.assertEqual(old.get('minLength',0),1 if operation=='replace' else 0)
            self.assertEqual(old['maxLength'],16000 if operation=='replace' else 0)
            invalid_old='' if operation=='replace' else 'not empty'
            content=json.dumps(dict(reply='edit',update=True,positive='',negative='',
                edits=[dict(field='positive',op=operation,old=invalid_old,new='text')]))
            with self.assertRaises(ValueError):resolve_suggestion(content,body)

    def test_sync_request_reuses_validated_history_only_within_that_request(self):
        store=self.client.app.state.store
        with patch.object(store,'chats',wraps=store.chats) as history:
            first=self.client.post('/api/chat',json=self.body)
            self.assertEqual(first.json()['status'],'SUCCEEDED')
            self.assertEqual(history.call_count,1)
            second=self.client.post('/api/chat',json=dict(self.body,request_id=str(uuid4())))
            self.assertEqual(second.json()['status'],'SUCCEEDED')
            self.assertEqual(history.call_count,2)
            self.assertIn(self.body['message'],[message['content'] for message in self.fake.calls[-1]['messages'][1:-1]])

    def test_custom_sampling_options_reach_request_and_cpu_wins_gpu_override(self):
        from backend.app.chat import ChatInput
        options={'temperature':.8,'top_p':.9,'top_k':24,'min_p':.05,'seed':123,'num_gpu':12,'stop':['END']}
        body=ChatInput(**dict(self.body,device='cpu',options=options,num_predict=-1,num_ctx=16384))
        with patch('backend.app.models.scan',return_value=[]):
            request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        self.assertEqual(request['options'],dict(temperature=.8,repeat_penalty=1.15,repeat_last_n=256,**{k:v for k,v in options.items() if k!='temperature'},num_predict=-1,num_ctx=16384)|{'num_gpu':0})
        for invalid in ({'format':'json'},{'temperature':float('inf')},{'top_p':1.1},{'top_k':1.5},{'stop':[123]}):
            with self.assertRaises(ValueError):ChatInput(**dict(self.body,options=invalid))

    def test_recent_history_preserves_long_tail_constraints_and_user_context_setting(self):
        from backend.app.chat import ChatInput
        store=self.client.app.state.store
        expected=[]
        for index in range(9):
            message=f'{index}: '+('장면 설명을 유지해 주세요. '*90)+'마지막 조건: 초록색 코트는 절대 사용하지 마세요.'
            reply=('확인한 장면 정보입니다. '*90)+'마지막 답변: 보라색 코트를 유지하겠습니다.'
            payload=dict(self.body,request_id=str(uuid4()),message=message)
            store.begin_chat(self.session,payload['request_id'],payload)
            store.finish_chat(self.session,payload['request_id'],proposal=dict(reply=reply,update=False,positive='',negative='',memory=''))
            expected.extend([{'role':'user','content':message},{'role':'assistant','content':reply}])
        body=ChatInput(**dict(self.body,num_ctx=32768,num_predict=-1))
        with patch('backend.app.models.scan',return_value=[]):
            request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        self.assertEqual(request['messages'][1:-1],expected[-16:])
        self.assertEqual(request['messages'][-1]['content'],body.message)
        self.assertEqual(request['options']['num_ctx'],32768)
        self.assertEqual(request['options']['num_predict'],-1)

    def test_history_page_is_workspace_scoped_and_validates_limits(self):
        for _ in range(3):
            body=dict(self.body,request_id=str(uuid4()))
            self.assertEqual(self.client.post('/api/chat',json=body).status_code,200)
        response=self.client.get('/api/chat/page',params=dict(session=self.session,limit=2))
        self.assertEqual(response.status_code,200)
        latest=response.json();self.assertEqual(len(latest['items']),2);self.assertTrue(latest['has_more'])
        older=self.client.get('/api/chat/page',params=dict(session=self.session,limit=2,before=latest['before'])).json()
        self.assertEqual(len(older['items']),1);self.assertFalse(older['has_more'])
        self.assertEqual(len(self.client.get('/api/chat',params=dict(session=self.session)).json()),3)
        for limit in (0,201):
            self.assertEqual(self.client.get('/api/chat/page',params=dict(session=self.session,limit=limit)).status_code,422)
        other=self.client.post('/api/sessions',json={}).json()['ref']
        self.assertEqual(self.client.get('/api/chat/page',params=dict(session=other,before=latest['before'])).status_code,422)

    def test_klein_context_preserves_document_and_explains_reference_selection(self):
        from backend.app.chat import ChatInput
        body=ChatInput(**self.body)
        with patch('backend.app.models.scan',return_value=[dict(id='image-test',family='flux2',name='Klein')]):
            request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        system=request['messages'][0]['content']
        self.assertIn('Negative prompts are not used',system)
        self.assertIn('reference selection is separate from chat attachments',system)
        self.assertIn('red teapot',system)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.loop=patch.object(JobQueue,'_loop'); self.loop.start()
        self.client=TestClient(create_app(self.temp.name),base_url='http://127.0.0.1:8765')
        self.client.__enter__()
        self.fake=FakeOllama(); self.client.app.state.chat.ollama=self.fake
        self.session=self.client.post('/api/sessions',json={}).json()['ref']
        self.body=dict(session=self.session, request_id=str(uuid4()), prompt_id=str(uuid4()), expected_revision=0,
            model='local:test', device='cpu', message='파란 주전자로 바꿔줘', positive='red teapot', negative='',settings={'model':'image-test'})

    def tearDown(self):
        self.client.__exit__(None,None,None); self.loop.stop(); self.temp.cleanup()

    def test_chat_persists_deduplicates_and_never_generates_by_itself(self):
        result=self.client.post('/api/chat',json=self.body)
        self.assertEqual(result.status_code,200)
        turn=result.json(); self.assertEqual(turn['status'],'SUCCEEDED')
        self.assertEqual(turn['positive'],'a blue teapot')
        self.assertEqual(turn['context']['positive'],'red teapot')
        self.assertEqual(self.client.post('/api/chat',json=self.body).json(),turn)
        self.assertEqual(len(self.fake.calls),1)
        self.assertEqual(self.fake.calls[0]['options']['num_gpu'],0)
        self.assertEqual(self.fake.calls[0]['keep_alive'],0)
        self.assertEqual(self.fake.released,['local:test'])
        self.assertEqual(self.client.get('/api/jobs',params={'session':self.session}).json(),[])
        folder=Path(self.temp.name)/'sessions'/self.session/'messages'/self.body['request_id']
        self.assertEqual((folder/'user.txt').read_text(encoding='utf-8'),self.body['message'])
        self.assertTrue((folder/'assistant.txt').exists())
        self.body['message']='다른 요청'
        self.assertEqual(self.client.post('/api/chat',json=self.body).status_code,409)

    def test_memory_rejects_invented_and_assistant_only_attributes(self):
        from backend.app.chat import grounded_memory,ChatInput
        self.assertEqual(grounded_memory('emerald coat\n보라색 코트를 입어\nElegant yet approachable',['보라색 코트를 입어']), '보라색 코트를 입어')
        self.assertEqual(grounded_memory('초록색',['초록색이 아니야']), '')
        body=ChatInput(**self.body)
        self.assertEqual(self.client.app.state.chat._ground_memory(body,'invented mood'),'')
        self.assertEqual(self.client.app.state.chat._ground_memory(body,''),'')

    def test_memory_omission_preserves_preferences_and_only_explicit_clear_resets(self):
        from backend.app.chat import ChatInput
        store=self.client.app.state.store;service=self.client.app.state.chat
        preference='I prefer watercolor and one adult woman.'
        payload=dict(self.body,request_id=str(uuid4()),message=preference)
        store.begin_chat(self.session,payload['request_id'],payload)
        store.finish_chat(self.session,payload['request_id'],proposal=dict(reply='noted',update=False,positive='',negative='',memory=preference))
        body=ChatInput(**dict(self.body,message='What did I ask earlier?'))
        for candidate,action in [('', 'auto'),('', 'replace'),('invented preference','replace'),(body.message,'keep')]:
            self.assertEqual(service._ground_memory(body,candidate,action),preference)
        self.assertEqual(service._ground_memory(body,body.message,'replace'),body.message)
        self.assertEqual(service._ground_memory(body,'','clear'),'')
        self.assertEqual(service._ground_memory(body,preference,'clear'),'')
        for action,expected in [('auto',preference),('clear',''),('auto','')]:
            proposal=dict(reply='okay',update=False,positive='',negative='',memory='',memory_action=action)
            self.fake.request=lambda *args,**kwargs:{'done':True,'message':{'content':json.dumps(proposal)}}
            response=self.client.post('/api/chat',json=dict(self.body,request_id=str(uuid4()),message='Forget all stored preferences.' if action=='clear' else 'Hello.'))
            turn=response.json();self.assertEqual(turn['status'],'SUCCEEDED')
            self.assertEqual(turn['memory'],expected)
            self.assertNotIn('memory_action',turn)

    def test_memory_source_ids_copy_complete_user_lines_and_reject_unknown_sources(self):
        from backend.app.chat import ChatInput,memory_sources
        store=self.client.app.state.store;service=self.client.app.state.chat
        preference='Use watercolor, never photorealism.'
        payload=dict(self.body,request_id=str(uuid4()),message=preference)
        store.begin_chat(self.session,payload['request_id'],payload)
        store.finish_chat(self.session,payload['request_id'],proposal=dict(reply='okay',update=False,positive='',negative='',memory=preference))
        body=ChatInput(**dict(self.body,message='The coat must not be green.\nKeep a silver star pin.'))
        request=service._request(body,service._payload(body),{})
        self.assertEqual(request['format']['properties']['memory_sources']['items']['enum'],['saved_0','current_0','current_1'])
        selected=service._ground_memory(body,'paraphrased incorrectly','replace',['saved_0','current_0'])
        self.assertEqual(selected,preference+'\nThe coat must not be green.')
        self.assertEqual(service._ground_memory(body,'','keep',['current_0']),selected)
        with self.assertRaisesRegex(ValueError,'출처'):
            service._ground_memory(body,'','replace',['other_workspace_0'])
        self.assertEqual(service._ground_memory(body,'','clear',['saved_0']),'')
        self.assertEqual(memory_sources('', 'x'*1601),{})
        self.assertEqual(len(memory_sources('', '\n'.join(str(i) for i in range(80)))),32)

    def test_memory_survives_recent_window_and_is_hash_checked(self):
        from backend.app.chat import ChatInput
        store=self.client.app.state.store
        for number in range(19):
            payload=dict(self.body,request_id=str(uuid4()),message='Mira has a silver star pin' if number==0 else f'turn {number}')
            store.begin_chat(self.session,payload['request_id'],payload)
            proposal=dict(reply='okay',update=False,positive='',negative='')
            if number==0:proposal['memory']='Mira has a silver star pin'
            store.finish_chat(self.session,payload['request_id'],proposal=proposal)
        body=ChatInput(**self.body)
        request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        self.assertIn('silver star pin',request['messages'][0]['content'])
        self.assertNotIn('turn 0',[m['content'] for m in request['messages'][1:]])
        self.assertEqual(len(request['messages']),18)
        payload=dict(self.body,request_id=str(uuid4()),message='forget the old character')
        store.begin_chat(self.session,payload['request_id'],payload)
        store.finish_chat(self.session,payload['request_id'],proposal=dict(reply='forgotten',update=False,positive='',negative='',memory=''))
        request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        self.assertNotIn('silver star pin',request['messages'][0]['content'])
        other=store.create_session('other')
        body=ChatInput(**dict(self.body,session=other))
        request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),{})
        self.assertNotIn('silver star pin',request['messages'][0]['content'])
        turn=store.chats(self.session)[0]
        path=Path(self.temp.name)/'sessions'/self.session/'messages'/turn['id']/'memory.txt'
        path.write_text('tampered',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'hash'):store.chats(self.session)

    def test_invalid_model_response_is_saved_without_prompt_mutation(self):
        self.fake.invalid=True
        turn=self.client.post('/api/chat',json=self.body).json()
        self.assertEqual(turn['status'],'FAILED')
        self.assertTrue(turn['error'])
        self.assertEqual(self.fake.released,['local:test'])
        self.assertEqual(self.client.get('/api/prompts',params={'session':self.session}).json(),[])

    def test_image_lock_blocks_chat_and_manual_edit_causes_cas_conflict(self):
        queue=self.client.app.state.queue
        queue.gpu.acquire()
        try:
            self.assertEqual(self.client.post('/api/chat',json=self.body).status_code,409)
            self.assertEqual(self.fake.calls,[])
        finally:
            queue.gpu.release()
        turn=self.client.post('/api/chat',json=self.body).json()
        context=turn['context']
        manual=dict(context,request_id=str(uuid4()),positive='user edit',generate=False)
        self.assertEqual(self.client.post('/api/prompts',json=manual).status_code,200)
        proposal=dict(context,request_id=turn['id'],positive=turn['positive'],negative=turn['negative'],generate=False)
        self.assertEqual(self.client.post('/api/prompts',json=proposal).status_code,409)

    def test_pending_recovery_and_tamper_detection(self):
        from backend.app.chat import ChatInput
        store=self.client.app.state.store
        payload=ChatInput.model_validate(self.body).model_dump(mode='json')
        store.begin_chat(self.session,self.body['request_id'],payload)
        ChatService(store,self.client.app.state.queue,self.fake)
        self.assertEqual(store.chats(self.session)[0]['status'],'FAILED')
        path=Path(self.temp.name)/'sessions'/self.session/'messages'/self.body['request_id']/'user.txt'
        path.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            store.chats(self.session)

    def test_local_model_gate_rejects_remote_alias_and_embeddings(self):
        ollama=Ollama()
        for info in ({'remote_host':'https://ollama.com','capabilities':['completion']}, {'capabilities':['embedding']}):
            with patch.object(ollama,'models',return_value=[{'name':'test'}]), patch.object(ollama,'request',return_value=info):
                with self.assertRaises(ValueError):
                    ollama.local_model('test')
        with patch.object(ollama,'running',return_value=[{'name':'external','size_vram':100}]):
            with self.assertRaises(OllamaError):
                ollama.image_ready()
            ollama.image_ready(selected_model='external')

    def test_thinking_controls_preserve_levels_and_offer_boolean_fallback(self):
        ollama=Ollama()
        for advertised in (None, {}, {'values':[]}, {'values':[False,'low','medium'],'default':'medium'}, {'values':[False],'default':False}):
            info={'capabilities':['completion','thinking']}
            if advertised is not None:info['thinking']=advertised
            with patch.object(ollama,'models',return_value=[{'name':'test'}]),patch.object(ollama,'request',return_value=info):
                _,actual=ollama.local_model('test')
            expected=advertised if advertised and advertised.get('values') else dict(advertised or {},values=[True,False])
            self.assertEqual(actual['thinking'],expected)
            from backend.app.chat import ChatInput
            if expected['values']==[True,False]:
                for value in (True,False):
                    body=ChatInput(**dict(self.body,think=value))
                    with patch('backend.app.models.scan',return_value=[]):
                        request=self.client.app.state.chat._request(body,body.model_dump(mode='json'),actual)
                    self.assertIs(request['think'],value)
                body=ChatInput(**dict(self.body,think='xhigh'))
                with patch('backend.app.models.scan',return_value=[]), self.assertRaises(ValueError):
                    self.client.app.state.chat._request(body,body.model_dump(mode='json'),actual)
        with patch.object(ollama,'models',return_value=[{'name':'test'}]),patch.object(ollama,'request',return_value={'capabilities':['completion']}):
            self.assertNotIn('thinking',ollama.local_model('test')[1])

    def test_apply_and_generate_reuses_existing_job_contract(self):
        turn=self.client.post('/api/chat',json=self.body).json()
        proposal=dict(turn['context'],request_id=turn['id'],positive=turn['positive'],negative=turn['negative'],generate=True)
        with patch('backend.app.main.execution_plan',return_value={'runner':'test'}):
            first=self.client.post('/api/prompts',json=proposal)
            self.assertEqual(first.status_code,200)
            self.assertEqual(self.client.post('/api/prompts',json=proposal).json(),first.json())
        jobs=self.client.get('/api/jobs',params={'session':self.session}).json()
        self.assertEqual(len(jobs),1)
        self.assertEqual(jobs[0]['status'],'QUEUED')
