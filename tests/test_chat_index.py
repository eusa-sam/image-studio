import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from backend.app.chat_index import page
from backend.app.storage import SessionStore,digest


class ChatIndexTests(unittest.TestCase):
    def test_pages_read_only_selected_bodies_and_recover_corrupt_index(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('history');request=str(uuid4())
            payload=dict(session=session,prompt_id=str(uuid4()),expected_revision=0,model='test',
                         message='original message',positive='scene',negative='',settings={})
            store.begin_chat(session,request,payload)
            source=Path(directory)/'sessions'/session/'messages'/request
            ids=[request]
            for number in range(202):
                identifier=str(uuid4());target=source.with_name(identifier);shutil.copytree(source,target)
                meta=json.loads((target/'meta.json').read_text(encoding='utf-8'))
                meta.update(id=identifier,created_at=f'2026-01-01T00:{number//60:02d}:{number%60:02d}+00:00')
                (target/'meta.json').write_text(json.dumps(meta),encoding='utf-8');ids.append(identifier)
            originals={p:digest(p.read_bytes()) for p in source.parent.rglob('*') if p.is_file()}
            with patch.object(store,'_chat',wraps=store._chat) as read:
                latest=page(store,session,limit=2)
                self.assertEqual(read.call_count,2)
            self.assertEqual(len(latest['items']),2);self.assertTrue(latest['has_more'])
            index=Path(directory)/'indexes/messages'/session
            self.assertEqual(len(list(index.glob('page-*.json'))),2)
            # Cached summaries must not contain prompt or chat bodies.
            for p in index.glob('page-*.json'):
                self.assertNotIn('original message',p.read_text(encoding='utf-8'))
            older=page(store,session,limit=2,before=latest['before'])
            self.assertFalse({x['id'] for x in older['items']} & {x['id'] for x in latest['items']})
            for path in (index/'page-000001.json',index/'head.json'):
                path.write_text('broken',encoding='utf-8')
                self.assertEqual(page(store,session,limit=2),latest)
            self.assertEqual(originals,{p:digest(p.read_bytes()) for p in originals})
            with self.assertRaises(ValueError):page(store,session,before=str(uuid4()))
            with self.assertRaises(ValueError):page(store,session,limit=0)
            # Selected body corruption is never returned from the summary cache.
            bad=source.parent/latest['items'][-1]['id']/'user.txt';bad.write_text('tampered',encoding='utf-8')
            self.assertEqual(len(page(store,session,limit=2)['items']),1)
            self.assertTrue(store.read_errors['sessions/'+session+'/messages'])
            # Failed index publication still returns healthy originals.
            (index/'head.json').unlink()
            with patch.object(store,'_replace',side_effect=OSError('disk full')):
                result=page(store,session,limit=2)
            self.assertEqual(len(result['items']),1)
            self.assertIn('indexes/messages/'+session,store.read_errors)

    def test_new_messages_and_status_changes_invalidate_summary(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('history')
            payload=dict(session=session,prompt_id=str(uuid4()),expected_revision=0,model='test',
                         message='one',positive='scene',negative='',settings={})
            first=str(uuid4());store.begin_chat(session,first,payload)
            self.assertEqual(page(store,session)['items'][0]['status'],'PENDING')
            store.finish_chat(session,first,error='interrupted')
            self.assertEqual(page(store,session)['items'][0]['status'],'FAILED')
            second=str(uuid4());store.begin_chat(session,second,payload)
            result=page(store,session,limit=1)
            self.assertEqual(result['items'][0]['id'],second)
            self.assertEqual(page(store,session,before=result['before'])['items'][0]['id'],first)
