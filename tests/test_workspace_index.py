import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from backend.app.storage import SessionStore,encode,digest
from backend.app.workspace_index import workspaces


class WorkspaceIndexTests(unittest.TestCase):
    def seed(self,store,count):
        for number in range(count):
            session=f'2026-{10+number%2:02d}/'+str(uuid4())
            store._replace(f'sessions/{session}/session.json',encode(dict(schema_version=1,id=session.split('/')[1],title=f'Workspace {number}',revision=1,created_at='2026-11-07T00:00:00Z')))

    def test_shards_cache_corruption_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as folder,SessionStore(folder) as store:
            self.seed(store,402)
            originals={p:digest(p.read_bytes()) for p in Path(folder).glob('sessions/*/*/session.json')}
            expected=store.sessions();self.assertEqual(workspaces(store),expected)
            head=store._read_json('indexes/workspaces/head.json')
            self.assertEqual(len(head['pages']),4)
            with patch.object(store,'sessions',side_effect=AssertionError('Cache must avoid source JSON parsing')):
                self.assertEqual(workspaces(store),expected)
            for mode in ('missing','corrupt','header'):
                page=store._path(head['pages'][0]['path'])
                if mode=='missing':page.unlink()
                elif mode=='corrupt':page.write_text('broken',encoding='utf-8')
                else:
                    broken=dict(head,pages=[]);store._replace('indexes/workspaces/head.json',encode(broken))
                self.assertEqual(workspaces(store),expected)
            self.assertEqual(originals,{p:digest(p.read_bytes()) for p in originals})

    def test_rename_archive_missing_metadata_and_failed_index_write(self):
        with tempfile.TemporaryDirectory() as folder,SessionStore(folder) as store:
            session=store.create_session('one');workspaces(store)
            store.update_session(session,expected_revision=1,title='renamed',archived=True)
            self.assertEqual(workspaces(store)[0]['title'],'renamed')
            self.assertTrue(workspaces(store)[0]['archived'])
            missing='2026-11/'+str(uuid4());store._path('sessions/'+missing).mkdir(parents=True)
            self.assertEqual(len(workspaces(store)),1)
            self.assertEqual(len(store.read_errors['sessions']),1)
            workspaces(store);self.assertEqual(len(store.read_errors['sessions']),1)
            store._path('indexes/workspaces/head.json').unlink()
            with patch.object(store,'_replace',side_effect=OSError('disk full')):
                self.assertEqual(workspaces(store)[0]['ref'],session)
                self.assertEqual(len(store.read_errors['workspace_index']),1)
            self.assertEqual(workspaces(store)[0]['ref'],session)
            self.assertNotIn('workspace_index',store.read_errors)
            store._path(f'sessions/{session}/session.json').write_text('broken',encoding='utf-8')
            self.assertEqual(workspaces(store),[])
            self.assertEqual(len(store.read_errors['sessions']),2)
