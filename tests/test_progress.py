import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app.progress import Progress,read_progress
from backend.app.main import create_app
from backend.app.jobs import JobQueue


class ProgressTests(unittest.TestCase):
    def test_phase_times_accumulate_reentry_without_resetting_each_step(self):
        with tempfile.TemporaryDirectory() as folder,patch('backend.app.progress.time.monotonic') as clock:
            clock.return_value=0
            progress=Progress(folder)
            progress.update('load')
            clock.return_value=3
            progress.update('generate',step=0)
            clock.return_value=5
            progress.update('generate',step=1)
            self.assertEqual(progress.timings(),{'load':3,'generate':2})
            clock.return_value=7
            progress.update('detail',region=1)
            clock.return_value=11
            progress.update('generate',step=0)
            clock.return_value=13
            self.assertEqual(progress.timings(),{'load':3,'generate':6,'detail':4})
            with self.assertRaises(ValueError):progress.update('invalid')
            self.assertEqual(progress.timings(),{'load':3,'generate':6,'detail':4})

    def test_status_is_atomic_bounded_and_never_overrides_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            job={'id':str(uuid4()),'status':'RUNNING'}
            folder=Path(directory)/'runtime/jobs'/job['id'];folder.mkdir(parents=True)
            progress=Progress(folder)
            progress.policy={'mode':'partial','free_vram_bytes':12*1024**3,'secret':'not exposed'}
            progress.update('generate',step=2,total=8)
            row=read_progress(directory,job)
            self.assertEqual(row['step'],2);self.assertNotIn('secret',row['memory_policy'])
            with patch('backend.app.progress.os.replace',side_effect=PermissionError),patch('builtins.print'):
                progress.update('save')
            self.assertEqual(read_progress(directory,job)['stage'],'generate')
            progress.update('detail',region=1,regions=3,target='face')
            self.assertNotIn('step',read_progress(directory,job))
            for state in ('CANCELLED','SUCCEEDED','FAILED','INTERRUPTED','QUEUED'):
                self.assertIsNone(read_progress(directory,{**job,'status':state}))
            for content in ('{',json.dumps({'stage':['load']}),' '*8193):
                progress.path.write_text(content,encoding='utf-8')
                self.assertIsNone(read_progress(directory,job))

    def test_jobs_api_exposes_only_running_job_progress(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(JobQueue,'_loop'):
            with TestClient(create_app(directory),base_url='http://127.0.0.1:8765') as client,patch('backend.app.main.execution_plan',return_value={'runner':'test'}):
                session=client.post('/api/sessions',json={}).json()['ref']
                receipt=client.post('/api/prompts',json=dict(session=session,prompt_id=str(uuid4()),request_id=str(uuid4()),
                    expected_revision=0,positive='test',settings={'model':'test'},generate=True)).json()
                job_id=receipt['job_id'];store=client.app.state.store
                store.transition(session,job_id,('QUEUED',),'RUNNING')
                folder=Path(directory)/'runtime/jobs'/job_id;folder.mkdir(parents=True)
                Progress(folder).update('upscale',step=2,total=4)
                jobs=lambda:client.get('/api/jobs',params={'session':session}).json()
                self.assertEqual(jobs()[0]['progress'],{'stage':'upscale','step':2,'total':4})
                self.assertNotIn('progress',store.job(session,job_id)['state'])
                client.post('/api/jobs/'+job_id+'/cancel',params={'session':session},json={})
                self.assertNotIn('progress',jobs()[0])
