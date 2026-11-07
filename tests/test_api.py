import json
import tempfile
import struct
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.jobs import JobQueue
from backend.app.models import Settings, scan
from backend.app.storage import SessionStore


class ApiTests(unittest.TestCase):
    def test_frontend_assets_have_module_mime_and_do_not_cache_old_code(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
            page=client.get('/')
            self.assertIn('id="startup-status"',page.text)
            self.assertIn('/boot.js?v=20261107-ui32',page.text)
            for name in ('boot.js','app.js','navigation.mjs','chat.mjs','mask.mjs','pipeline.mjs','runtime.mjs','reference.mjs'):
                response=client.get('/'+name+'?v=20261106-ui2')
                self.assertEqual(response.status_code,200)
                self.assertIn('javascript',response.headers['content-type'])
                self.assertEqual(response.headers['cache-control'],'no-store')
            self.assertEqual(page.headers['cache-control'],'no-store')

    def test_prompt_conflicts_validation_and_origin(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(directory), base_url='http://127.0.0.1:8765') as client:
            session = client.post('/api/sessions', json={'title': 'test'}).json()['ref']
            body = dict(session=session, prompt_id=str(uuid4()), request_id=str(uuid4()),
                        expected_revision=0, positive='teapot', negative='', settings={'model': 'missing'})
            response = client.post('/api/prompts', json=body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.post('/api/prompts', json=body).json(), response.json())
            body['request_id'] = str(uuid4())
            self.assertEqual(client.post('/api/prompts', json=body).status_code, 409)
            body.update(generate=True, expected_revision=1)
            self.assertEqual(client.post('/api/prompts', json=body).status_code, 422)
            self.assertEqual(client.get('/api/jobs', params={'session':session}).json(), [])
            self.assertEqual(client.post('/api/sessions', json={}, headers={'Origin':'https://evil.example'}).status_code, 403)
            self.assertEqual(client.get('/api/sessions', headers={'Host':'evil.example'}).status_code, 403)
            self.assertEqual(client.post('/api/sessions', content='{}', headers={'Content-Type':'text/plain'}).status_code, 415)
            self.assertEqual(client.get('/api/prompts', params={'session':'../../models'}).status_code, 422)

    def test_duplicate_job_cancel_and_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(JobQueue, '_loop'):
            plan = {'schema_version':1, 'runner':'test'}
            with TestClient(create_app(directory), base_url='http://127.0.0.1:8765') as client, patch('backend.app.main.execution_plan', return_value=plan):
                session = client.post('/api/sessions', json={}).json()['ref']
                body = dict(session=session, prompt_id=str(uuid4()), request_id=str(uuid4()), expected_revision=0,
                            positive='test', settings={'model':'test'}, generate=True)
                receipt = client.post('/api/prompts', json=body).json()
                self.assertEqual(client.post('/api/prompts', json=body).json(), receipt)
                jobs = client.get('/api/jobs', params={'session':session}).json()
                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0]['status'], 'QUEUED')
                cancelled = client.post(f'/api/jobs/{receipt["job_id"]}/cancel', params={'session':session}, json={})
                self.assertEqual(cancelled.json()['status'], 'CANCELLED')
                body.update(request_id=str(uuid4()), expected_revision=1)
                next_job = client.post('/api/prompts', json=body).json()['job_id']
                client.app.state.store.transition(session, next_job, ('QUEUED',), 'RUNNING')
            with TestClient(create_app(directory), base_url='http://127.0.0.1:8765') as client:
                jobs = client.get('/api/jobs', params={'session':session}).json()
                self.assertEqual(next(j for j in jobs if j['id'] == next_job)['status'], 'INTERRUPTED')
                self.assertEqual(next(j for j in jobs if j['id'] == receipt['job_id'])['status'], 'CANCELLED')

    def test_snapshot_tamper_and_artifact_transaction(self):
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store:
            session = store.create_session('test')
            receipt = store.save_prompt(session, str(uuid4()), expected_revision=0, positive='original', negative='',
                settings={}, request_id=str(uuid4()), generate=True)
            job = receipt['job_id']
            store.transition(session, job, ('AWAITING_VALIDATION',), 'RUNNING')
            store.transition(session, job, ('RUNNING',), 'SUCCEEDED', artifacts={'report.json':b'{}', 'image.png':b'image'})
            self.assertEqual(store.artifact(session, job, 'image.png').read_bytes(), b'image')
            with self.assertRaises(ValueError):
                store.artifact(session, job, '../../session.json')
            snapshot = Path(directory) / 'sessions' / session / 'jobs' / job / 'snapshot/positive.txt'
            snapshot.write_text('edited', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                store.job(session, job)

    def test_settings_reject_unsupported_and_out_of_bounds(self):
        for extra in ({'width':513}, {'steps':0}, {'seed':-1}, {'lora':'anything'}, {'prediction_type':'unknown'}):
            with self.assertRaises(ValueError):
                Settings.model_validate(dict(model='test', **extra))

    def test_scan_uses_structure_and_keeps_broken_files_visible(self):
        with tempfile.TemporaryDirectory() as directory, patch('backend.app.models.MODEL_ROOT', Path(directory)):
            folder = Path(directory) / 'checkpoint'
            folder.mkdir()
            shapes={'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight':[49408,768],
                    'conditioner.embedders.1.model.token_embedding.weight':[49408,1280],
                    'model.diffusion_model.input_blocks.0.0.weight':[320,4,3,3],
                    'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight':[640,2048]}
            header = json.dumps({k:{'dtype':'F32','shape':v,'data_offsets':[0,4]} for k,v in shapes.items()}).encode()
            (folder / 'renamed.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
            (folder / 'broken.safetensors').write_bytes(b'broken')
            (folder / 'unknown.gguf').write_bytes(b'unknown')
            models = {m['name']:m for m in scan()}
            self.assertTrue(models['renamed.safetensors']['available'])
            self.assertFalse(models['broken.safetensors']['available'])
            self.assertFalse(models['unknown.gguf']['available'])
            shapes['model.diffusion_model.input_blocks.0.0.weight']=[320,9,3,3]
            header=json.dumps({k:{'dtype':'F32','shape':v,'data_offsets':[0,4]} for k,v in shapes.items()}).encode()
            (folder/'renamed.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
            self.assertFalse(next(m for m in scan() if m['name']=='renamed.safetensors')['available'])

    def test_worker_parent_watch_does_not_block_import_and_exits_on_eof(self):
        code = ('import threading,time; from backend.app.worker import parent_watch; '
                'threading.Thread(target=parent_watch,daemon=True).start(); '
                'import torch; print("READY",flush=True); time.sleep(30)')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output.txt'
            with output.open('wb') as log:
                process = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline and 'READY' not in output.read_text(errors='replace'):
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertIn('READY', output.read_text(errors='replace'))
                process.stdin.close()
                self.assertEqual(process.wait(timeout=10), 20)
            finally:
                if not process.stdin.closed:
                    process.stdin.close()
                process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
