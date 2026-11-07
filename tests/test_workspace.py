import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.storage import digest


class WorkspaceTests(unittest.TestCase):
    def test_archive_restore_conflict_and_prompt_preservation(self):
        with tempfile.TemporaryDirectory() as directory,TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
            ref=client.post('/api/sessions',json={'title':'original'}).json()['ref']
            prompt=str(uuid4())
            self.assertEqual(client.post('/api/prompts',json=dict(session=ref,prompt_id=prompt,request_id=str(uuid4()),expected_revision=0,positive='do not change',settings={'model':'missing'})).status_code,200)
            folder=Path(directory)/'sessions'/ref
            files={p:digest(p.read_bytes()) for p in folder.rglob('*') if p.is_file() and p.name!='session.json'}
            body=dict(session=ref,expected_revision=1,title='renamed',archived=True)
            response=client.post('/api/sessions/update',json=body)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['revision'],2)
            self.assertEqual(client.get('/api/sessions').json(),[])
            rows=client.get('/api/sessions?include_archived=true').json()
            self.assertEqual(rows[0]['ref'],ref);self.assertEqual(rows[0]['title'],'renamed')
            self.assertEqual(client.post('/api/sessions/update',json=body).status_code,409)
            self.assertEqual(client.post('/api/sessions/update',json=dict(body,expected_revision=2,title=' ')).status_code,422)
            self.assertEqual(client.post('/api/sessions/update',json=dict(body,expected_revision=2,archived=False)).status_code,200)
            self.assertEqual(len(client.get('/api/sessions').json()),1)
            self.assertEqual(files,{p:digest(p.read_bytes()) for p in files})
            self.assertEqual(client.get('/api/prompts',params={'session':ref}).json()[0]['positive'],'do not change')
