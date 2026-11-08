import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from backend.app.storage import SessionStore, ConflictError, digest
from backend.app.conversations import create, read, listing, update, default_id
from backend.app.chat import ChatService, ChatInput
from backend.app.chat_index import page


class ConversationTests(unittest.TestCase):
    def test_legacy_files_isolation_pages_memory_archive_and_restart(self):
        with tempfile.TemporaryDirectory(dir='data') as directory:
            with SessionStore(directory) as store:
                session=store.create_session('workspace')
                other=store.create_session('other')
                body=dict(session=session,request_id=str(uuid4()),prompt_id=str(uuid4()),expected_revision=0,
                          model='test',message='legacy preference',positive='original',negative='',settings={'model':'image'})
                legacy=body['request_id'];payload=dict(body)
                store.begin_chat(session,legacy,payload)
                store.finish_chat(session,legacy,proposal=dict(reply='saved',update=False,positive='original',negative='',memory='legacy preference'))
                files={p:digest(p.read_bytes()) for p in (Path(directory)/'sessions'/session/'messages'/legacy).rglob('*') if p.is_file()}
                row=create(store,session)
                self.assertNotEqual(default_id(session),row['id'])
                with self.assertRaises(FileNotFoundError):read(store,other,row['id'])
                new=dict(body,request_id=str(uuid4()),conversation_id=row['id'],message='new preference')
                store.begin_chat(session,new['request_id'],new)
                with self.assertRaises(ConflictError):update(store,session,row['id'],2,'busy',True)
                store.finish_chat(session,new['request_id'],proposal=dict(reply='new',update=False,positive='original',negative='',memory='new preference'))
                service=ChatService(store,SimpleNamespace(gpu=threading.Lock()))
                self.assertEqual([t['memory'] for t in service._history(ChatInput(**new))],['new preference'])
                self.assertEqual([t['memory'] for t in service._history(ChatInput(**body))],['legacy preference'])
                self.assertEqual([t['id'] for t in page(store,session,conversation_id=row['id'])['items']],[new['request_id']])
                with self.assertRaises(ValueError):page(store,session,conversation_id=row['id'],before=legacy)
                updated=read(store,session,row['id'])
                self.assertEqual(updated['title'],'new preference')
                archived=update(store,session,row['id'],updated['revision'],'custom title',True)
                self.assertEqual(len(listing(store,session)),1)
                self.assertEqual(len(listing(store,session,True)),2)
                with self.assertRaises(ConflictError):store.begin_chat(session,str(uuid4()),new)
                restored=update(store,session,row['id'],archived['revision'],'custom title',False)
                store.begin_chat(session,str(uuid4()),dict(new,message='different'))
                self.assertEqual(read(store,session,row['id'])['title'],'custom title')
                self.assertEqual(files,{p:digest(p.read_bytes()) for p in files})
                # Original payload hashes and idempotent old requests remain valid.
                self.assertFalse(store.begin_chat(session,legacy,payload)[1])
            with SessionStore(directory) as store:
                self.assertEqual(read(store,session,row['id'])['revision'],restored['revision'])
                self.assertEqual(len(store.chats(session)),3)

    def test_default_chat_can_be_named_and_archived_without_rewriting_messages(self):
        with tempfile.TemporaryDirectory(dir='data') as directory,SessionStore(directory) as store:
            session=store.create_session('workspace');row=read(store,session)
            saved=update(store,session,row['id'],0,'name',True)
            self.assertEqual(listing(store,session),[])
            with self.assertRaises(ConflictError):update(store,session,row['id'],0,'stale',False)
            self.assertEqual(update(store,session,row['id'],saved['revision'],'name',False)['title'],'name')
