import json,tempfile,unittest
from pathlib import Path
from uuid import uuid4
from backend.app.storage import SessionStore
from backend.app.chat_index import page

class ChatApplicationTests(unittest.TestCase):
    def test_application_survives_later_edits_and_is_only_read_for_ui(self):
        for generate in (False,True):
            with self.subTest(generate=generate),tempfile.TemporaryDirectory() as directory:
                with SessionStore(directory) as store:
                    session=store.create_session('apply');request=str(uuid4());prompt=str(uuid4())
                    payload=dict(session=session,prompt_id=prompt,expected_revision=0,model='test',positive='original',negative='keep',settings={},message='edit')
                    store.begin_chat(session,request,payload)
                    store.finish_chat(session,request,proposal=dict(reply='edited',update=True,positive='revised',negative='keep'))
                    self.assertNotIn('application',page(store,session)['items'][0])
                    receipt=store.save_prompt(session,prompt,expected_revision=0,positive='revised',negative='keep',settings={'prompt_mode':'weighted'},request_id=request,generate=generate)
                    expected={key:receipt[key] for key in ('prompt_id','revision','job_id')}
                    self.assertNotIn('application',store.chats(session)[0])
                    self.assertEqual(page(store,session)['items'][0]['application'],expected)
                    store.save_prompt(session,prompt,expected_revision=1,positive='manual revision',negative='keep',settings={},request_id=str(uuid4()))
                with SessionStore(directory) as store:
                    self.assertEqual(store.chats(session,include_application=True)[0]['application'],expected)
                    self.assertEqual(page(store,session)['items'][0]['application'],expected)
                    version=Path(directory)/'sessions'/session/'prompts'/prompt/'revisions/000001/positive.txt'
                    version.write_text('damaged',encoding='utf-8')
                    row=page(store,session)['items'][0]
                    self.assertNotIn('application',row);self.assertIn('hash',row['application_error'])
                    self.assertEqual(row['reply'],'edited')
                    self.assertNotIn('application_error',store.chats(session)[0])

    def test_unrelated_request_receipt_does_not_mark_proposal_applied(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('apply');request=str(uuid4());prompt=str(uuid4())
            payload=dict(session=session,prompt_id=prompt,expected_revision=0,model='test',positive='original',negative='',settings={},message='edit')
            store.begin_chat(session,request,payload)
            store.finish_chat(session,request,proposal=dict(reply='edited',update=True,positive='revised',negative=''))
            store.save_prompt(session,prompt,expected_revision=0,positive='different',negative='',settings={},request_id=request)
            row=page(store,session)['items'][0]
            self.assertNotIn('application',row);self.assertIn('mismatch',row['application_error'])

    def test_http_history_and_page_return_verified_application(self):
        from fastapi.testclient import TestClient
        from backend.app.main import create_app
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
                store=client.app.state.store
                session=store.create_session('applied');request=str(uuid4());prompt=str(uuid4())
                payload=dict(session=session,prompt_id=prompt,expected_revision=0,model='test',positive='original',negative='',settings={},message='edit')
                store.begin_chat(session,request,payload)
                store.finish_chat(session,request,proposal=dict(reply='edited',update=True,positive='revised',negative=''))
                store.save_prompt(session,prompt,expected_revision=0,positive='revised',negative='',settings={},request_id=request)
            with TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
                legacy=client.get('/api/chat',params={'session':session});paged=client.get('/api/chat/page',params={'session':session})
                self.assertEqual(legacy.status_code,200);self.assertEqual(paged.status_code,200)
                self.assertEqual(legacy.json()[0]['application'],{'prompt_id':prompt,'revision':1,'job_id':None})
                self.assertEqual(paged.json()['items'],legacy.json())
