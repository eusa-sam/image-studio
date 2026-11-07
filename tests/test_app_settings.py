import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.app_settings import AppSettings,export_images
from backend.app.storage import SessionStore,digest
from backend.app import models,prompt_tokens
from backend.app.ollama import Ollama


class AppSettingsTests(unittest.TestCase):
    def test_failed_settings_publish_preserves_runtime_disk_and_retry(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'models';model.mkdir();data=root/'data'
            with TestClient(create_app(data),base_url='http://127.0.0.1:8765',raise_server_exceptions=False) as client:
                original=client.get('/api/settings').json()
                saved=client.post('/api/settings',json={**original,'model_path':str(model),'output_path':str(root/'first')}).json()
                path=data/'settings/application.json';before=path.read_bytes()
                replacement={**saved,'output_path':str(root/'second'),'ollama_endpoint':'/custom/api'}
                replace=os.replace
                def deny_settings(source,target):
                    if Path(target)==path:raise PermissionError('settings file locked')
                    return replace(source,target)
                with patch('backend.app.storage.os.replace',side_effect=deny_settings):
                    failed=client.post('/api/settings',json=replacement)
                self.assertEqual(failed.status_code,422,failed.text)
                self.assertIn('설정 파일',failed.json()['detail'])
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(client.get('/api/settings').json(),saved)
                self.assertEqual(client.app.state.queue.output_path,Path(saved['output_path']))
                self.assertEqual(client.app.state.chat.ollama.endpoint,saved['ollama_endpoint'])
                self.assertFalse(list(path.parent.glob('*.tmp')))
                self.assertTrue(client.app.state.queue.gpu.acquire(blocking=False));client.app.state.queue.gpu.release()
                retry=client.post('/api/settings',json=replacement)
                self.assertEqual(retry.status_code,200,retry.text)
                self.assertEqual(retry.json()['revision'],saved['revision']+1)
            with TestClient(create_app(data),base_url='http://127.0.0.1:8765') as client:
                self.assertEqual(client.get('/api/settings').json(),retry.json())

    def test_retry_export_api_recovers_publish_crash_and_preserves_modified_files(self):
        from backend.app.jobs import JobQueue
        with tempfile.TemporaryDirectory() as directory,patch.object(JobQueue,'_loop'):
            with TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client:
                store=client.app.state.store;queue=client.app.state.queue
                session=store.create_session('retry')
                job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',details={'output_status':'FAILED'},artifacts={'image.png':b'original'})
                url=f'/api/jobs/{job}/export?session={session}'
                queue.gpu.acquire()
                try:self.assertEqual(client.post(url,json={}).status_code,409)
                finally:queue.gpu.release()
                with patch('backend.app.app_settings.export_images',side_effect=OSError('disk full')):
                    self.assertEqual(client.post(url,json={}).status_code,200)
                self.assertEqual(store.job(session,job)['state']['output_status'],'FAILED')
                generated=store.job(session,job)['state']['finished_at']
                self.assertEqual(client.post(url,json={}).status_code,200)
                state=store.job(session,job)['state'];self.assertEqual(state['output_status'],'SUCCEEDED')
                self.assertEqual(state['finished_at'],generated)
                self.assertGreaterEqual(state['output_finished_at'],state['output_retry_started_at'])
                self.assertEqual(client.post(url,json={}).status_code,409)
                destination=Path(state['output_path']);before=(destination/'image.png').stat().st_mtime_ns
                store.transition(session,job,('SUCCEEDED',),'SUCCEEDED',details={'output_status':'INTERRUPTED'})
                self.assertEqual(client.post(url,json={}).status_code,200)
                self.assertEqual(store.job(session,job)['state']['output_status'],'SUCCEEDED')
                self.assertEqual((destination/'image.png').stat().st_mtime_ns,before)
                (destination/'image.png').write_bytes(b'user changed')
                store.transition(session,job,('SUCCEEDED',),'SUCCEEDED',details={'output_status':'INTERRUPTED'})
                self.assertEqual(client.post(url,json={}).status_code,200)
                self.assertEqual(store.job(session,job)['state']['output_status'],'FAILED')
                self.assertEqual((destination/'image.png').read_bytes(),b'user changed')
                self.assertEqual(store.artifact(session,job,'image.png').read_bytes(),b'original')
                self.assertFalse(queue.pending)

    def test_output_completion_and_crash_state_preserve_generated_original(self):
        from backend.app.jobs import JobQueue
        with tempfile.TemporaryDirectory() as directory,SessionStore(Path(directory)/'data') as store,patch.object(JobQueue,'_loop'):
            queue=JobQueue(store)
            try:
                session=store.create_session('export timing')
                def new_job():
                    job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                    store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',details={'output_status':'PENDING'},artifacts={'image.png':b'image'})
                    return job
                completed=new_job();generated=store.job(session,completed)['state']['finished_at']
                queue._export_result(session,completed)
                state=store.job(session,completed)['state']
                self.assertEqual(state['output_status'],'SUCCEEDED')
                self.assertEqual(state['finished_at'],generated)
                self.assertGreaterEqual(state['output_finished_at'],generated)
                failed=new_job()
                with patch('backend.app.app_settings.export_images',side_effect=OSError('disk full')):queue._export_result(session,failed)
                self.assertEqual(store.job(session,failed)['state']['output_status'],'FAILED')
                interrupted=new_job()
            finally:queue.close()
            restarted=JobQueue(store)
            try:
                state=store.job(session,interrupted)['state']
                self.assertEqual(state['status'],'SUCCEEDED')
                self.assertEqual(state['output_status'],'INTERRUPTED')
                self.assertIn('output_error',state)
                self.assertEqual(store.artifact(session,interrupted,'image.png').read_bytes(),b'image')
            finally:restarted.close()

    def test_save_apply_conflict_busy_and_restart(self):
        before=models.MODEL_ROOT
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'custom models';model.mkdir();data=root/'data'
            with TestClient(create_app(data),base_url='http://127.0.0.1:8765') as client:
                original=client.get('/api/settings').json()
                body=dict(original,model_path=str(model),output_path=str(root/'images'),ollama_url='http://127.0.0.1:12555',ollama_endpoint='/ollama/api')
                saved=client.post('/api/settings',json=body);self.assertEqual(saved.status_code,200,saved.text)
                self.assertEqual(models.MODEL_ROOT,model);self.assertEqual(prompt_tokens.MODEL_ROOT,model)
                self.assertEqual(client.get('/api/models').json(),[])
                self.assertEqual(client.app.state.chat.ollama.base_url,body['ollama_url'])
                self.assertIs(client.app.state.queue.ollama,client.app.state.chat.ollama)
                self.assertEqual(client.post('/api/settings',json=body).status_code,409)
                queue=client.app.state.queue;queue.gpu.acquire()
                try:self.assertEqual(client.post('/api/settings',json=saved.json()).status_code,409)
                finally:queue.gpu.release()
                self.assertEqual(client.get('/api/settings').json(),saved.json())
            self.assertEqual(models.MODEL_ROOT,before)
            with TestClient(create_app(data),base_url='http://127.0.0.1:8765') as client:
                self.assertEqual(client.get('/api/settings').json(),saved.json())
                self.assertEqual(models.MODEL_ROOT,model)
        self.assertEqual(models.MODEL_ROOT,before)

    def test_connection_url_validation_and_custom_request_address(self):
        base=dict(model_path=str(Path.cwd()),output_path=str(Path.cwd()/'outputs'))
        for url in ('http://user:secret@localhost:123','https://example.com','file:///tmp','http://0.0.0.0','http://127.0.0.1/path'):
            with self.assertRaises(ValueError):AppSettings(**base,ollama_url=url)
        for endpoint in ('/api/../chat','//evil.com','/api?x=1'):
            with self.assertRaises(ValueError):AppSettings(**base,ollama_endpoint=endpoint)
        self.assertEqual(AppSettings(**base,ollama_url='http://192.168.1.2:11434/').ollama_url,'http://192.168.1.2:11434')
        with patch('backend.app.ollama.local_listener',return_value=True),patch('urllib.request.build_opener') as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value=b'{"version":"test"}'
            self.assertEqual(Ollama('http://127.0.0.1:12555','/custom/api').request('version')['version'],'test')
            self.assertEqual(factory.return_value.open.call_args.args[0].full_url,'http://127.0.0.1:12555/custom/api/version')

    def test_export_keeps_history_and_never_overwrites_destination(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(Path(directory)/'data') as store:
            session=store.create_session('output');job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='blur',settings={'seed':42},request_id=str(uuid4()),generate=True)['job_id']
            store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':b'final','original.png':b'original'})
            destination=Path(export_images(store,session,job,Path(directory)/'exports'))
            manifest=json.loads((destination/'manifest.json').read_text(encoding='utf8'))
            for name,sha in manifest['hashes'].items():self.assertEqual(digest((destination/name).read_bytes()),sha)
            self.assertEqual((destination/'positive.txt').read_text(),'scene')
            self.assertEqual(store.artifact(session,job,'image.png').read_bytes(),b'final')
            with self.assertRaises(ValueError):export_images(store,session,job,Path(directory)/'exports')
            self.assertEqual((destination/'image.png').read_bytes(),b'final')

    def test_export_streams_and_manifest_failure_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(Path(directory)/'data') as store:
            session=store.create_session('output');job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='blur',settings={},request_id=str(uuid4()),generate=True)['job_id']
            payload=b'pixels'*(400*1024)
            store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':payload})
            root=Path(directory)/'exports';destination=root/session/job
            original_read=Path.read_bytes
            def bounded_read(path):
                if path.name=='image.png':raise AssertionError('Export must stream images')
                return original_read(path)
            with patch.object(Path,'read_bytes',bounded_read),patch('backend.app.app_settings.os.fsync',side_effect=[None,None,None,None,OSError('disk unavailable')]):
                with self.assertRaisesRegex(OSError,'disk unavailable'):export_images(store,session,job,root)
            self.assertFalse(destination.exists())
            partials=list(destination.parent.glob('*.partial'))
            self.assertEqual(len(partials),1)
            self.assertEqual(store.job(session,job)['state']['status'],'SUCCEEDED')
            self.assertEqual(store.artifact(session,job,'image.png').read_bytes(),payload)
            with patch.object(Path,'read_bytes',bounded_read):export_images(store,session,job,root)
            self.assertEqual((destination/'image.png').read_bytes(),payload)
            self.assertTrue(partials[0].exists())
            manifest=json.loads((destination/'manifest.json').read_text())
            for name,sha in manifest['hashes'].items():self.assertEqual(digest((destination/name).read_bytes()),sha)
