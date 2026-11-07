import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.storage import SessionStore,digest
from backend.app.prompt_index import summaries

class PromptIndexTests(unittest.TestCase):
    def test_cached_previews_rebuild_without_changing_originals(self):
        with tempfile.TemporaryDirectory() as root,SessionStore(root) as store:
            session=store.create_session('prompts');key=str(uuid4())
            def save(revision,text):
                return store.save_prompt(session,key,expected_revision=revision,positive=text,negative='negative stays private to document',settings={'steps':9},request_id=str(uuid4()),generate=False)
            save(0,'a'*100)
            with patch.object(store,'read_prompt',wraps=store.read_prompt) as read:
                rows=summaries(store,session);self.assertEqual(read.call_count,1)
                self.assertEqual(summaries(store,session),rows);self.assertEqual(read.call_count,1)
            self.assertEqual(rows,[dict(id=key,revision=1,title='a'*60)])
            source=Path(root)/'sessions'/session/'prompts'/key
            originals={p:digest(p.read_bytes()) for p in source.rglob('*') if p.is_file()}
            cache=Path(root)/'indexes/prompts'/session/(key+'.json')
            self.assertNotIn('negative stays',cache.read_text(encoding='utf8'))
            cache.write_text('broken',encoding='utf8')
            self.assertEqual(summaries(store,session),rows)
            self.assertEqual(originals,{p:digest(p.read_bytes()) for p in originals})
            cache.unlink()
            with patch.object(store,'_replace',side_effect=OSError('disk full')):
                self.assertEqual(summaries(store,session),rows)
            self.assertIn('indexes/prompts/'+session,store.read_errors)
            save(1,'new title');self.assertEqual(summaries(store,session)[0]['title'],'new title')
            self.assertNotIn('indexes/prompts/'+session,store.read_errors)
            (source/'revisions/000002/positive.txt').write_text('tampered body',encoding='utf8')
            self.assertEqual(summaries(store,session),[])
            self.assertTrue(store.read_errors['sessions/'+session+'/prompts'])

    def test_api_separates_previews_and_verified_document(self):
        with tempfile.TemporaryDirectory() as root,TestClient(create_app(root),base_url='http://127.0.0.1:8765') as client:
            store=client.app.state.store;session=store.create_session('API');other=store.create_session('other');key=str(uuid4())
            store.save_prompt(session,key,expected_revision=0,positive='scene',negative='blur',settings={'steps':9},request_id=str(uuid4()),generate=False)
            response=client.get('/api/prompts/summaries',params={'session':session});self.assertEqual(response.status_code,200)
            self.assertEqual(response.json(),[dict(id=key,revision=1,title='scene')])
            response=client.get('/api/prompts/'+key,params={'session':session});self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['negative'],'blur');self.assertEqual(response.json()['settings'],{'steps':9})
            self.assertNotEqual(client.get('/api/prompts/'+key,params={'session':other}).status_code,200)
            source=Path(root)/'sessions'/session/'prompts'/key/'revisions/000001/negative.txt'
            source.write_text('changed',encoding='utf8')
            # A preview is not a document health certificate; selection verifies all files.
            self.assertEqual(client.get('/api/prompts/summaries',params={'session':session}).json(),[dict(id=key,revision=1,title='scene')])
            self.assertNotEqual(client.get('/api/prompts/'+key,params={'session':session}).status_code,200)
