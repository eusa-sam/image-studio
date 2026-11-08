import asyncio
import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch

from backend.app.chat import ChatService, ChatInput, partial_reply
from backend.app.storage import SessionStore
from backend.app.ollama import OllamaError


class StreamingModel:
    def __init__(self):
        self.released=[]
        self.wait=False
        self.started=asyncio.Event()
        self.requests=[]

    def local_model(self,name): return {'digest':'test'},{'capabilities':['completion']}
    def image_ready(self,**kwargs): pass
    def release(self,name): self.released.append(name)
    async def stream_chat(self,body):
        self.requests.append(body)
        yield {'message':{'thinking':'Compare the '},'done':False}
        yield {'message':{'thinking':'user constraints.'},'done':False}
        content=json.dumps(dict(reply='좋아요! 함께 만들어 봐요.',update=False,positive='must not apply',negative=''),ensure_ascii=False)
        for part in (content[:20],content[20:]):
            yield {'message':{'content':part},'done':False}
            self.started.set()
            if self.wait: await asyncio.Event().wait()
        yield {'done':True,'done_reason':'stop'}


class StreamTests(unittest.TestCase):
    def test_gpu_choice_warns_on_cpu_placement_without_losing_response(self):
        class CPUModel(StreamingModel):
            async def stream_chat(self,body):
                yield {'message':{},'_gpu_bytes':0}
                async for chunk in super().stream_chat(body):yield chunk
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('placement')
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None)
                model=CPUModel();service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='hello',positive='',negative='',settings={'model':'image'})
                events=[event async for event in service.stream(body)]
                self.assertEqual(len([event for event in events if event.get('code')=='cpu_fallback']),1)
                self.assertEqual(events[-1]['turn']['status'],'SUCCEEDED')
                self.assertEqual(model.released,['test'])
                self.assertFalse(queue.gpu.locked())
        asyncio.run(run())

    def test_completion_write_failure_reports_error_and_preserves_recovery(self):
        async def run(prepared):
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('save failure')
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None)
                model=StreamingModel();service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='preserve request',positive='original',negative='keep',settings={'model':'image'})
                replace=store._replace;complete=store.finish_chat
                def fail(path,content):
                    if path.endswith('/assistant.txt' if prepared else '/payload/0'):raise OSError('disk full simulation')
                    return replace(path,content)
                def finish(*args,**kwargs):
                    with patch.object(store,'_replace',side_effect=fail):return complete(*args,**kwargs)
                with patch.object(store,'finish_chat',side_effect=finish):
                    events=[event async for event in service.stream(body)]
                self.assertEqual(events[-1]['type'],'error');self.assertIn('저장',events[-1]['text'])
                self.assertIn('disk full simulation',events[-1]['text'])
                self.assertFalse(any(event['type']=='done' for event in events))
                self.assertFalse(queue.gpu.locked());self.assertFalse(service.active);self.assertEqual(model.released,['test'])
                store.recover()
                service=ChatService(store,queue,model)
                saved=store.chats(session)[0]
                self.assertEqual(saved['message'],body.message);self.assertEqual(saved['context']['positive'],'original')
                self.assertEqual(saved['status'],'SUCCEEDED' if prepared else 'FAILED')
                if prepared:
                    self.assertEqual(saved['reply'],'좋아요! 함께 만들어 봐요.')
                    duplicate=[event async for event in service.stream(body)]
                    self.assertEqual(duplicate[-1]['turn']['id'],saved['id']);self.assertEqual(len(model.requests),1)
                retried=[event async for event in service.stream(body.model_copy(update={'request_id':uuid4()}))]
                self.assertEqual(retried[-1]['turn']['status'],'SUCCEEDED')
        for prepared in (False,True):
            with self.subTest(prepared=prepared):asyncio.run(run(prepared))

    def test_cancel_during_initial_image_validation_does_not_create_or_run_chat(self):
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('validate input')
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None)
                model=StreamingModel();service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='hello',positive='original',negative='',settings={'model':'image'})
                entered=threading.Event();finish=threading.Event();timed_out=threading.Event()
                original=service._payload
                def validate(body):
                    entered.set()
                    if not finish.wait(2):timed_out.set()
                    return original(body)
                async def collect():return [event async for event in service.stream(body)]
                with patch.object(service,'_payload',side_effect=validate):
                    task=asyncio.create_task(collect())
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait,3))
                        self.assertFalse(timed_out.is_set(),'Input validation blocked the event loop')
                        self.assertFalse(service.cancel('other',str(body.request_id))['cancelled'])
                        self.assertTrue(service.cancel(session,str(body.request_id))['cancelled'])
                        await asyncio.sleep(.02)
                        self.assertTrue(queue.gpu.locked());self.assertFalse(task.done())
                    finally:
                        finish.set();events=await task
                self.assertEqual(events[-1]['type'],'error');self.assertIn('중단',events[-1]['text'])
                self.assertEqual(store.chats(session),[]);self.assertEqual(model.requests,[])
                self.assertFalse(queue.gpu.locked());self.assertFalse(service.active)
                retry=[event async for event in service.stream(body.model_copy(update={'request_id':uuid4()}))]
                self.assertEqual(retry[-1]['turn']['status'],'SUCCEEDED')
        asyncio.run(run())

    def test_history_preparation_keeps_loop_responsive_and_cancel_waits_for_reader(self):
        async def run(method):
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('history preparation')
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None)
                model=StreamingModel();service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='hello',positive='original',negative='',settings={'model':'image'})
                entered=threading.Event();finish=threading.Event();timed_out=threading.Event()
                original=getattr(service,method)
                def prepare(*args,**kwargs):
                    entered.set()
                    if not finish.wait(2):timed_out.set()
                    return original(*args,**kwargs)
                async def collect():return [event async for event in service.stream(body)]
                with patch.object(service,method,side_effect=prepare):
                    task=asyncio.create_task(collect())
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait,3))
                        self.assertFalse(timed_out.is_set(),'History preparation blocked the event loop')
                        self.assertTrue(service.cancel(session,str(body.request_id))['cancelled'])
                        await asyncio.sleep(.02)
                        self.assertTrue(queue.gpu.locked());self.assertFalse(task.done())
                    finally:
                        finish.set();events=await task
                self.assertEqual(events[-1]['turn']['status'],'CANCELLED')
                self.assertEqual(model.requests,[]);self.assertFalse(queue.gpu.locked());self.assertFalse(service.active)
                retry=[event async for event in service.stream(body.model_copy(update={'request_id':uuid4()}))]
                self.assertEqual(retry[-1]['turn']['status'],'SUCCEEDED')
        for method in ('_history','_request'):
            with self.subTest(method=method):asyncio.run(run(method))

    def test_cancel_during_unload_waits_and_finalizes_chat_before_accepting_retry(self):
        async def run(fail_release):
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                started=threading.Event();finish=threading.Event()
                class SlowRelease(StreamingModel):
                    def release(self,name):
                        started.set()
                        if not finish.wait(5):raise RuntimeError('test unload timeout')
                        super().release(name)
                        if fail_release:raise OllamaError('unload failed')
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None)
                model=SlowRelease();service=ChatService(store,queue,model)
                session=store.create_session('cancel unload')
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='hello',positive='original',negative='keep',settings={'model':'image'})
                async def collect():return [event async for event in service.stream(body)]
                task=asyncio.create_task(collect())
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait,2))
                    self.assertTrue(service.cancel(session,str(body.request_id))['cancelled'])
                    await asyncio.sleep(0.02)
                    self.assertTrue(queue.gpu.locked(),'GPU lease released while Ollama unload still runs')
                    self.assertFalse(task.done())
                    self.assertEqual(store.chats(session)[0]['status'],'PENDING')
                finally:
                    finish.set()
                    outcome=await asyncio.gather(task,return_exceptions=True)
                self.assertNotIsInstance(outcome[0],BaseException)
                turn=outcome[0][-1]['turn']
                self.assertEqual(turn['status'],'CANCELLED');self.assertTrue(turn['reply'])
                self.assertEqual(turn['reasoning'],'Compare the user constraints.')
                if fail_release:self.assertIn('unload failed',turn['error'])
                self.assertEqual(store.chats(session)[0]['status'],'CANCELLED')
                self.assertEqual(model.released,['test']);self.assertFalse(queue.gpu.locked());self.assertFalse(service.active)
                service.ollama=StreamingModel()
                retry=[event async for event in service.stream(body.model_copy(update={'request_id':uuid4()}))]
                self.assertEqual(retry[-1]['turn']['status'],'SUCCEEDED')
        for fail_release in (False,True):
            with self.subTest(fail_release=fail_release):asyncio.run(run(fail_release))

    def test_cancel_during_image_suspend_keeps_gpu_lease_until_thread_finishes(self):
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                started=threading.Event();finish=threading.Event();finished=threading.Event()
                def suspend():
                    started.set()
                    try:
                        if not finish.wait(5):raise RuntimeError('test suspend timeout')
                    finally:finished.set()
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=suspend)
                model=StreamingModel();service=ChatService(store,queue,model)
                session=store.create_session('cancel preparation')
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='hello',positive='original',negative='',settings={'model':'image'})
                async def collect():return [event async for event in service.stream(body)]
                task=asyncio.create_task(collect())
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait,2))
                    self.assertTrue(service.cancel(session,str(body.request_id))['cancelled'])
                    await asyncio.sleep(0.02)
                    self.assertTrue(queue.gpu.locked(),'GPU lease released while suspend thread still runs')
                    self.assertFalse(task.done())
                    rejected=[event async for event in service.stream(body.model_copy(update={'request_id':uuid4()}))]
                    self.assertEqual(rejected[-1]['type'],'error')
                finally:
                    finish.set();events=await task
                self.assertTrue(finished.is_set());self.assertFalse(queue.gpu.locked())
                self.assertEqual(events[-1]['turn']['status'],'CANCELLED')
                self.assertEqual(model.requests,[]);self.assertFalse(service.active)
                queue.suspend_image_model=lambda:None
                retry=[event async for event in service.stream(body.model_copy(update={'request_id':uuid4()}))]
                self.assertEqual(retry[-1]['turn']['status'],'SUCCEEDED')
        asyncio.run(run())

    def test_length_failure_returns_saved_done_and_releases_gpu_for_retry(self):
        class LimitedModel(StreamingModel):
            async def stream_chat(self,body):
                self.requests.append(body)
                yield {'message':{'content':'{"reply":"partial answer'},'done':False}
                yield {'done':True,'done_reason':'length'}
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('retry')
                queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None)
                model=LimitedModel();service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,
                    model='test',message='keep this request',positive='original',negative='preserve',
                    settings={'model':'image'},num_predict=5,num_ctx=4096)
                events=[event async for event in service.stream(body)]
                self.assertEqual(events[-1]['type'],'done')
                turn=events[-1]['turn'];self.assertEqual(turn['status'],'FAILED')
                self.assertEqual(turn['message'],body.message)
                self.assertIn('5',turn['error']);self.assertIn('4096',turn['error'])
                self.assertEqual(store.chats(session)[0]['status'],'FAILED')
                self.assertEqual(model.released,['test']);self.assertFalse(queue.gpu.locked())
                service.ollama=StreamingModel()
                retry=body.model_copy(update={'request_id':uuid4(),'num_predict':None})
                retried=[event async for event in service.stream(retry)]
                self.assertEqual(retried[-1]['turn']['status'],'SUCCEEDED')
                self.assertFalse(queue.gpu.locked());self.assertEqual(len(store.chats(session)),2)
        asyncio.run(run())

    def test_disconnected_consumer_closes_generation_and_preserves_partial(self):
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('chat');queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None);model=StreamingModel()
                service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='test',message='hello',positive='',negative='',settings={'model':'image'})
                stream=service.stream(body)
                self.assertEqual((await anext(stream))['type'],'status')
                self.assertEqual((await anext(stream))['type'],'reasoning')
                self.assertEqual((await anext(stream))['type'],'reasoning')
                self.assertEqual((await anext(stream))['type'],'reply')
                await stream.aclose()
                self.assertEqual(store.chats(session)[0]['status'],'CANCELLED')
                self.assertEqual(model.released,['test']);self.assertFalse(queue.gpu.locked())
        asyncio.run(run())

    def test_partial_reply_handles_escape_boundaries_without_leaking_json(self):
        content=json.dumps({'reply':'안녕 "친구"\n🙂','positive':'secret'},ensure_ascii=True)
        results=[partial_reply(content[:i]) for i in range(len(content)+1)]
        self.assertEqual(results[-1],'안녕 "친구"\n🙂')
        self.assertTrue(all('secret' not in text for text in results))
        self.assertTrue(all(not any(0xD800<=ord(c)<=0xDFFF for c in text) for text in results))
        self.assertEqual(partial_reply('{"update":false,"reply":"later"}'),'later')

    def test_partial_reply_skips_complete_top_level_values_without_exposing_nested_text(self):
        reply='답변 "quoted"\n🙂'
        prefixes=[{'update':False}, {'edits':[{'new':'secret', 'reply':'nested secret'}]},
                  {'positive':'literal {"reply":"prompt secret"}', 'memory':None},
                  {'score':-1.2e5,'empty':{},'list':[True,False,None,12]}]
        for prefix in prefixes:
            for ascii_only in (False,True):
                content=json.dumps(dict(prefix,reply=reply,negative='hidden'),ensure_ascii=ascii_only)
                results=[partial_reply(content[:i]) for i in range(len(content)+1)]
                self.assertEqual(results[-1],reply)
                self.assertTrue(all(reply.startswith(value) for value in results),results)
        for content in ['[{"reply":"secret"}]','{"edits":[{"reply":"secret"}',
                        '{"x":1 "reply":"secret"}', '{"x":false,,"reply":"secret"}',
                        '{"reply":{"reply":"secret"}}']:
            self.assertEqual(partial_reply(content),'')
        self.assertEqual(partial_reply('{"x":null,"repl\\u0079":"visible"}'),'visible')

    def test_stream_persists_final_and_keeps_workspace_conversation_context(self):
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('chat');queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None);model=StreamingModel()
                service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='test',message='hello',positive='original',negative='',settings={'model':'image'})
                with patch.object(store,'chats',wraps=store.chats) as history:
                    events=[event async for event in service.stream(body)]
                    self.assertEqual(history.call_count,1)
                self.assertTrue(any(event['type']=='reply' for event in events))
                turn=events[-1]['turn'];self.assertEqual(turn['status'],'SUCCEEDED');self.assertEqual(turn['positive'],'original')
                self.assertEqual(''.join(e['text'] for e in events if e['type']=='reasoning'),'Compare the user constraints.')
                self.assertEqual(turn['reasoning'],'Compare the user constraints.')
                self.assertEqual(store.chats(session)[0]['reasoning'],turn['reasoning'])
                duplicate=[event async for event in service.stream(body)]
                self.assertEqual(duplicate[-1]['turn'],turn);self.assertEqual(len(model.requests),1)
                body=body.model_copy(update={'request_id':uuid4(),'prompt_id':uuid4(),'message':'continue'})
                _=[event async for event in service.stream(body)]
                self.assertIn('hello',[message['content'] for message in model.requests[-1]['messages']])
                self.assertNotIn(turn['reasoning'],[message['content'] for message in model.requests[-1]['messages']])
                self.assertEqual(model.released,['test','test']);self.assertFalse(queue.gpu.locked())
                self.assertEqual(model.requests[-1]['format']['properties']['negative']['maxLength'],4000)
                self.assertGreater(model.requests[-1]['options']['repeat_penalty'],1)
                path=store._path('sessions/'+session+'/messages/'+turn['id']+'/reasoning.txt')
                path.write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    store._chat(session,turn['id'])
        asyncio.run(run())

    def test_cancel_saves_partial_releases_gpu_and_rejects_wrong_workspace(self):
        async def run():
            with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
                session=store.create_session('chat');queue=SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None);model=StreamingModel();model.wait=True
                service=ChatService(store,queue,model)
                body=ChatInput(session=session,request_id=uuid4(),prompt_id=uuid4(),expected_revision=0,model='test',message='hello',positive='',negative='',settings={'model':'image'})
                async def collect():return [event async for event in service.stream(body)]
                task=asyncio.create_task(collect());await model.started.wait()
                self.assertFalse(service.cancel('other',str(body.request_id))['cancelled'])
                # A concurrent rejected request must not erase the active request's cancellation handle.
                rejected=[event async for event in service.stream(body)]
                self.assertEqual(rejected[-1]['type'],'error')
                self.assertTrue(service.cancel(session,str(body.request_id))['cancelled'])
                events=await task;turn=events[-1]['turn']
                self.assertEqual(turn['status'],'CANCELLED');self.assertTrue(turn['reply'])
                self.assertEqual(turn['reasoning'],'Compare the user constraints.')
                self.assertEqual(model.released,['test']);self.assertFalse(queue.gpu.locked());self.assertFalse(service.active)
                self.assertEqual(store.chats(session)[0]['status'],'CANCELLED')
        asyncio.run(run())
