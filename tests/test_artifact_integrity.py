import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from backend.app.storage import SessionStore, ConflictError, digest, encode


class ArtifactIntegrityTests(unittest.TestCase):
    def test_interrupted_artifact_commit_recovers_bytes_and_hashes_together(self):
        with tempfile.TemporaryDirectory() as directory:
            with SessionStore(directory) as store:
                session=store.create_session('interrupted')
                receipt=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',
                    negative='',settings={},request_id=str(uuid4()),generate=True)
                job=receipt['job_id']
                replace=store._replace
                def fail_state(relative,content):
                    if relative.endswith('/state.json'):
                        raise OSError('simulated disk write failure')
                    return replace(relative,content)
                with patch.object(store,'_replace',side_effect=fail_state):
                    with self.assertRaises(OSError):
                        store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':b'complete'})
            with SessionStore(directory) as recovered:
                self.assertEqual(recovered.artifact(session,job,'image.png').read_bytes(),b'complete')
                self.assertEqual(recovered.job(session,job)['state']['artifact_hashes']['image.png'],digest(b'complete'))

    def test_commit_hash_read_tampering_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store:
            session = store.create_session('integrity')
            receipt = store.save_prompt(session, str(uuid4()), expected_revision=0,
                positive='scene', negative='', settings={}, request_id=str(uuid4()), generate=True)
            job = receipt['job_id']
            state = store.transition(session, job, ('AWAITING_VALIDATION',), 'SUCCEEDED',
                artifacts={'image.png': b'image', 'report.json': b'{}'})
            self.assertEqual(state['artifact_hashes'], {'image.png': digest(b'image'), 'report.json': digest(b'{}')})
            image = store.artifact(session, job, 'image.png')
            self.assertEqual(image.read_bytes(), b'image')
            before = (image.parent.parent/'state.json').read_bytes()
            with self.assertRaises(ConflictError):
                store.transition(session, job, ('SUCCEEDED',), 'SUCCEEDED', artifacts={'image.png': b'replacement'})
            self.assertEqual(image.read_bytes(), b'image')
            self.assertEqual((image.parent.parent/'state.json').read_bytes(), before)
            image.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                store.artifact(session, job, 'image.png')
            self.assertEqual(store.artifact(session, job, 'report.json').read_bytes(), b'{}')
            state['artifact_hashes'].pop('report.json')
            (image.parent.parent/'state.json').write_bytes(encode(state))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                store.artifact(session, job, 'report.json')

    def test_legacy_results_stay_readable_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store:
            session = store.create_session('legacy')
            job = str(uuid4())
            base = Path(directory)/'sessions'/session/'jobs'/job
            (base/'artifacts').mkdir(parents=True)
            (base/'artifacts/image.png').write_bytes(b'legacy')
            state = encode(dict(schema_version=1, id=job, status='SUCCEEDED'))
            (base/'state.json').write_bytes(state)
            self.assertEqual(store.artifact(session, job, 'image.png').read_bytes(), b'legacy')
            self.assertEqual((base/'state.json').read_bytes(), state)
