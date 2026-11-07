import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from filelock import Timeout
from backend.app.storage import SessionStore, ConflictError, RecoveryRequired


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, store, session, prompt, **overrides):
        args = dict(expected_revision=0, positive='한글\nred teapot', negative='', settings={'seed': 42}, request_id=str(uuid4()), generate=True)
        args.update(overrides)
        return store.save_prompt(session, prompt, **args)

    def test_separation_revisions_snapshot_and_duplicate(self):
        with SessionStore(self.root) as store:
            a, b = store.create_session('A'), store.create_session('B')
            prompt, request = str(uuid4()), str(uuid4())
            first = self.save(store, a, prompt, request_id=request)
            self.assertEqual(first, self.save(store, a, prompt, request_id=request))
            with self.assertRaises(ConflictError):
                self.save(store, a, prompt, request_id=request, positive='different')
            with self.assertRaises(ConflictError):
                self.save(store, a, prompt)
            self.save(store, a, prompt, expected_revision=1, positive='new', generate=False)
            self.save(store, b, prompt, positive='other session')
            self.assertEqual(store.read_prompt(a, prompt)['positive'], 'new')
            self.assertEqual(store.read_prompt(b, prompt)['positive'], 'other session')
            snapshot = self.root / 'sessions' / a / 'jobs' / first['job_id'] / 'snapshot/positive.txt'
            self.assertEqual(snapshot.read_text(encoding='utf-8'), '한글\nred teapot')
            self.assertEqual(len(list((self.root/'sessions'/a/'jobs').iterdir())), 1)
            state = json.loads((snapshot.parent.parent/'state.json').read_text())
            self.assertEqual(state['status'], 'AWAITING_VALIDATION')

    def test_job_times_keep_original_start_and_finish_on_artifact_followup(self):
        with SessionStore(self.root) as store:
            session=store.create_session('timing')
            job=self.save(store,session,str(uuid4()))['job_id']
            running=store.transition(session,job,('AWAITING_VALIDATION',),'RUNNING')
            self.assertEqual(running['started_at'],running['updated_at'])
            ended=store.transition(session,job,('RUNNING',),'CANCELLED')
            self.assertEqual(ended['started_at'],running['started_at'])
            self.assertEqual(ended['finished_at'],ended['updated_at'])
            followup=store.transition(session,job,('CANCELLED',),'CANCELLED',details={'has_original':True})
            self.assertEqual(followup['finished_at'],ended['finished_at'])
            self.assertEqual(store.job(session,job)['state']['started_at'],running['started_at'])

    def test_writer_lock_and_release(self):
        with SessionStore(self.root):
            with self.assertRaises(Timeout):
                SessionStore(self.root)
            code = 'from backend.app.storage import SessionStore; import sys; SessionStore(sys.argv[1])'
            result = subprocess.run([sys.executable, '-c', code, str(self.root)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Timeout', result.stderr)
        with SessionStore(self.root):
            pass

    def test_process_death_recovers_ready_commit(self):
        with SessionStore(self.root) as store:
            session = store.create_session('crash')
        prompt, request = str(uuid4()), str(uuid4())
        code = '''
import os,sys
from backend.app.storage import SessionStore
s=SessionStore(sys.argv[1]); original=s._replace
def crash(path, content):
    original(path, content)
    if '/revisions/' in path and path.endswith('positive.txt'): os._exit(17)
s._replace=crash
s.save_prompt(sys.argv[2],sys.argv[3],expected_revision=0,positive='saved',negative='',settings={},request_id=sys.argv[4],generate=True)
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.root), session, prompt, request], capture_output=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        with SessionStore(self.root) as store:
            self.assertEqual(store.read_prompt(session, prompt)['positive'], 'saved')
            result = store.save_prompt(session, prompt, expected_revision=0, positive='saved', negative='', settings={}, request_id=request, generate=True)
            self.assertEqual(len(list((self.root/'sessions'/session/'jobs').iterdir())), 1)
            self.assertIsNotNone(result['job_id'])

    def test_failed_write_blocks_reads_until_recovery(self):
        with SessionStore(self.root) as store:
            session = store.create_session('error'); prompt = str(uuid4())
            original = store._replace
            def fail(path, content):
                if path.startswith('sessions/'):
                    raise OSError('disk full simulation')
                original(path, content)
            with patch.object(store, '_replace', side_effect=fail):
                with self.assertRaises(OSError):
                    self.save(store, session, prompt)
            with self.assertRaises(RecoveryRequired):
                store.read_prompt(session, prompt)
            store.recover()
            self.assertEqual(store.read_prompt(session, prompt)['revision'], 1)

    def test_unready_preparation_is_not_applied(self):
        with SessionStore(self.root) as store:
            session = store.create_session('unready'); prompt = str(uuid4())
            original = store._replace
            def fail(path, content):
                if path.endswith('/READY.json'):
                    raise OSError('not ready')
                original(path, content)
            with patch.object(store, '_replace', side_effect=fail):
                with self.assertRaises(OSError):
                    self.save(store, session, prompt)
            store.recover()
            self.assertFalse((self.root/'sessions'/session/'prompts').exists())

    def test_paths_schema_and_corruption(self):
        with SessionStore(self.root) as store:
            for path in ('../outside', 'C:/outside', 'sessions/../../outside', 'a\\..\\b'):
                with self.assertRaises(ValueError):
                    store._path(path)
            with self.assertRaises(ValueError):
                store.read_prompt('../bad', str(uuid4()))
            session = store.create_session('hash'); prompt = str(uuid4())
            self.save(store, session, prompt)
            text = self.root/'sessions'/session/'prompts'/prompt/'revisions/000001/positive.txt'
            text.write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                store.read_prompt(session, prompt)

    def test_recovery_preserves_external_edit(self):
        with SessionStore(self.root) as store:
            session = store.create_session('conflict'); prompt = str(uuid4())
            self.save(store, session, prompt)
            original = store._replace
            def fail(path, content):
                if path.startswith('sessions/'):
                    raise OSError('interrupted')
                original(path, content)
            with patch.object(store, '_replace', side_effect=fail):
                with self.assertRaises(OSError):
                    self.save(store, session, prompt, expected_revision=1)
            head = self.root/'sessions'/session/'prompts'/prompt/'head.json'
            head.write_text('external edit')
            with self.assertRaises(RecoveryRequired):
                store.recover()
            self.assertEqual(head.read_text(), 'external edit')
            self.assertFalse((head.parent/'revisions/000002').exists())

    def test_corrupt_payload_stops_recovery(self):
        with SessionStore(self.root) as store:
            session = store.create_session('payload'); prompt = str(uuid4())
            original = store._replace
            def fail(path, content):
                if path.startswith('sessions/'):
                    raise OSError('interrupted')
                original(path, content)
            with patch.object(store, '_replace', side_effect=fail):
                with self.assertRaises(OSError):
                    self.save(store, session, prompt)
            pending = [p.parent for p in (self.root/'transactions').glob('*/*/READY.json') if not (p.parent/'COMMITTED.json').exists()]
            (pending[0]/'payload/0').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RecoveryRequired, 'Corrupt'):
                store.recover()
            self.assertFalse((self.root/'sessions'/session/'prompts').exists())


if __name__ == '__main__':
    unittest.main()
