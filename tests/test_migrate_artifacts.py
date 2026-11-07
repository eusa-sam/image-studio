import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from backend.app.migrate_artifacts import migrate
from backend.app.storage import SessionStore, encode


def legacy(store, session):
    job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='',
        settings={},request_id=str(uuid4()),generate=True)['job_id']
    state=store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':b'original'})
    state.pop('artifact_hashes')
    relative=f'sessions/{session}/jobs/{job}/state.json'
    store._commit({relative:encode(state)})
    return job,relative


class ArtifactMigrationTests(unittest.TestCase):
    def test_unknown_version_is_preserved_while_known_records_migrate(self):
        with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
            session=store.create_session('versions')
            _,unknown=legacy(store,session)
            _,known=legacy(store,session)
            state=store._read_json(unknown)
            state['schema_version']=99
            original=encode(state)
            store._replace(unknown,original)
            result=migrate(store,apply=True)
            self.assertEqual(result['migrated'],[known])
            self.assertEqual(result['errors'],[dict(path=unknown,error='Unsupported schema version')])
            self.assertEqual(store._path(unknown).read_bytes(),original)

    def test_progress_write_failure_resumes_remaining_jobs_without_double_revision(self):
        with tempfile.TemporaryDirectory() as root:
            with SessionStore(root) as store:
                session=store.create_session('resume')
                records=sorted(legacy(store,session)[1] for _ in range(3))
                originals={path:store._read_json(path)['revision'] for path in records}
                replace=store._replace
                writes=0
                def fail(path,content):
                    nonlocal writes
                    if path.startswith('migrations/artifact-hashes/'):
                        writes+=1
                        if writes==2:raise OSError('progress disk failure')
                    return replace(path,content)
                with patch.object(store,'_replace',side_effect=fail):
                    with self.assertRaisesRegex(OSError,'progress disk failure'):
                        migrate(store,apply=True)
                self.assertIn('artifact_hashes',store._read_json(records[0]))
                self.assertNotIn('artifact_hashes',store._read_json(records[1]))
            with SessionStore(root) as store:
                result=migrate(store,apply=True)
                self.assertEqual(result['migrated'],records[1:])
                self.assertEqual(result['skipped'],records[:1])
                for path in records:
                    self.assertEqual(store._read_json(path)['revision'],originals[path]+1)
                    self.assertEqual((store._path(path).parent/'artifacts/image.png').read_bytes(),b'original')
                self.assertEqual(migrate(store,apply=True)['eligible'],[])

    def test_dry_run_backup_migration_and_idempotence(self):
        with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
            session=store.create_session('legacy');job,relative=legacy(store,session)
            original=store._path(relative).read_bytes()
            report=migrate(store)
            self.assertEqual(report['eligible'],[relative]);self.assertEqual(report['errors'],[])
            self.assertEqual(store._path(relative).read_bytes(),original)
            result=migrate(store,apply=True)
            self.assertEqual(result['migrated'],[relative])
            backup=store._path('backups/'+result['backup']['id']+'/files/'+relative)
            self.assertEqual(backup.read_bytes(),original)
            self.assertEqual(store.artifact(session,job,'image.png').read_bytes(),b'original')
            self.assertEqual(migrate(store,apply=True)['eligible'],[])

    def test_changed_or_missing_evidence_is_not_blessed(self):
        with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
            session=store.create_session('corrupt');job,relative=legacy(store,session)
            image=store._path(relative).parent/'artifacts/image.png';image.write_bytes(b'changed')
            original=store._path(relative).read_bytes()
            result=migrate(store,apply=True)
            self.assertEqual(len(result['errors']),1);self.assertEqual(result['migrated'],[])
            self.assertEqual(store._path(relative).read_bytes(),original)
            image.write_bytes(b'original')
            with patch('backend.app.migrate_artifacts.checksum',return_value='bad'):
                with self.assertRaisesRegex(ValueError,'Corrupt artifact transaction'):
                    migrate(store,apply=True)

    def test_interrupted_state_commit_recovers_and_resume_skips_completed_job(self):
        with tempfile.TemporaryDirectory() as root:
            with SessionStore(root) as store:
                session=store.create_session('interrupt');job,relative=legacy(store,session)
                replace=store._replace
                def fail(path,content):
                    if path==relative:raise OSError('disk failure')
                    return replace(path,content)
                with patch.object(store,'_replace',side_effect=fail):
                    with self.assertRaises(OSError):migrate(store,apply=True)
                self.assertTrue(list(Path(root).glob('backups/*/manifest.json')))
            with SessionStore(root) as store:
                self.assertEqual(store.artifact(session,job,'image.png').read_bytes(),b'original')
                self.assertEqual(migrate(store,apply=True)['eligible'],[])
