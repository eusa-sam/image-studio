import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.storage import SessionStore


class CorruptRecordTests(unittest.TestCase):
    def test_server_opens_good_records_without_rewriting_corrupt_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with SessionStore(root) as store:
                good=store.create_session('good');bad=store.create_session('bad')
                prompt=str(uuid4())
                store.save_prompt(good,prompt,expected_revision=0,positive='kept',negative='',settings={},request_id=str(uuid4()))
            base=root/'sessions'/good
            paths={root/'sessions'/bad/'session.json':b'broken json',
                   base/'prompts'/str(uuid4())/'head.json':b'{"schema_version":99}',
                   base/'messages'/str(uuid4())/'meta.json':b'[]',
                   base/'jobs'/str(uuid4())/'state.json':b'{"schema_version":1,"id":"wrong"}'}
            for path,content in paths.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            missing=base/'jobs'/str(uuid4())/'state.json';missing.parent.mkdir()
            with TestClient(create_app(root),base_url='http://127.0.0.1:8765') as client:
                self.assertEqual([r['ref'] for r in client.get('/api/sessions').json()],[good])
                self.assertEqual(client.get('/api/prompts',params={'session':good}).json()[0]['positive'],'kept')
                self.assertEqual(client.get('/api/jobs',params={'session':good}).json(),[])
                self.assertEqual(client.get('/api/chat',params={'session':good}).json(),[])
                errors=client.get('/api/storage-errors').json()
                self.assertEqual(len(errors),5)
                self.assertTrue(any(row['path']==missing.relative_to(root).as_posix() for row in errors))
                self.assertIsNone(client.app.state.queue.failure)
                with self.assertRaises(ValueError):client.app.state.store.sessions()
                for path,content in paths.items():self.assertEqual(path.read_bytes(),content)
                self.assertFalse(missing.exists())

    def test_tampered_pending_snapshot_is_not_queued_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with SessionStore(root) as store:
                session=store.create_session('test')
                job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='original',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                store.transition(session,job,('AWAITING_VALIDATION',),'QUEUED')
            base=root/'sessions'/session/'jobs'/job
            snapshot=base/'snapshot/positive.txt';snapshot.write_bytes(b'tampered')
            state=(base/'state.json').read_bytes()
            with TestClient(create_app(root),base_url='http://127.0.0.1:8765') as client:
                self.assertEqual(client.app.state.queue.pending,[])
                self.assertTrue(any('hash mismatch' in r['error'] for r in client.get('/api/storage-errors').json()))
                self.assertEqual((base/'state.json').read_bytes(),state)
                self.assertEqual(snapshot.read_bytes(),b'tampered')
