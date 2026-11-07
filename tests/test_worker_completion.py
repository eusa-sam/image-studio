import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from uuid import uuid4

from PIL import Image
from backend.app.jobs import JobQueue
from backend.app.storage import SessionStore


class WorkerCompletionTests(unittest.TestCase):
    def test_failed_worker_with_corrupt_report_preserves_original_in_workspace(self):
        for failure in ('worker_exit','completion','final_png','bad_original'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store,patch.object(JobQueue,'_loop'):
                queue=JobQueue(store)
                session=store.create_session('preserve original')
                job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                plan={'runner':'sdxl_txt2img','settings':{}}
                store.transition(session,job,('AWAITING_VALIDATION',),'QUEUED',details={'plan':plan})
                work=Path(directory)/'runtime/jobs'/job;work.mkdir(parents=True)
                Image.new('RGB',(8,8),'red').save(work/'original.png')
                original=(work/'original.png').read_bytes()
                (work/'report.json').write_text('[]',encoding='utf-8')
                process=Mock();process.stdin=io.BytesIO();process.poll.return_value=1;process.returncode=1
                if failure in ('completion','bad_original'):(work/'worker-done.json').write_text('{}',encoding='utf-8')
                elif failure=='final_png':
                    (work/'worker-done.json').write_text('{"code":0}',encoding='utf-8')
                    (work/'image.png').write_bytes(b'partial final PNG')
                if failure=='bad_original':(work/'original.png').write_bytes(b'partial original PNG')
                try:
                    with patch('backend.app.jobs.preflight',return_value=plan),patch('backend.app.jobs.Ollama.image_ready'),patch('backend.app.jobs.subprocess.Popen',return_value=process):
                        queue._execute(session,job)
                    state=store.job(session,job)['state']
                    if failure=='bad_original':
                        self.assertEqual(state['status'],'FAILED')
                        self.assertFalse(state['has_original'])
                        self.assertIn('원본 복구 실패',state['error'])
                    else:
                        self.assertEqual(state['status'],'PARTIAL_FAILED')
                        self.assertTrue(state['has_original'])
                        self.assertEqual(store.artifact(session,job,'original.png').read_bytes(),original)
                        self.assertIn(b'report_error',store.artifact(session,job,'report.json').read_bytes())
                    self.assertEqual((work/'report.json').read_text(encoding='utf-8'),'[]')
                    self.assertIsNone(queue.current)
                    self.assertIsNone(queue.failure)
                finally:queue.close()

    def test_invalid_completion_fails_one_job_releases_worker_and_allows_next_job(self):
        for content in ('{broken','null','[]','{}','{"code":true}','{"code":"0"}','{"code":0,"cpu_cached":"false"}'):
            with self.subTest(content=content),tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store,patch.object(JobQueue,'_loop'):
                queue=JobQueue(store)
                session=store.create_session('completion recovery')
                plan={'runner':'sdxl_txt2img','settings':{}}
                jobs=[]
                for _ in range(2):
                    job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                    store.transition(session,job,('AWAITING_VALIDATION',),'QUEUED',details={'plan':plan})
                    jobs.append(job)
                processes=[]
                def start(*args,**kwargs):
                    index=len(processes)
                    process=Mock();process.stdin=io.BytesIO();process.poll.return_value=None
                    def wait(**kwargs):
                        process.poll.return_value=1
                        return 1
                    process.wait.side_effect=wait
                    processes.append(process)
                    work=Path(directory)/'runtime/jobs'/jobs[index]
                    (work/'worker-done.json').write_text(content if index==0 else '{"code":0}',encoding='utf-8')
                    if index:
                        Image.new('RGB',(8,8),'blue').save(work/'image.png')
                        (work/'report.json').write_text('{}',encoding='utf-8')
                    return process
                try:
                    with patch('backend.app.jobs.preflight',return_value=plan),patch('backend.app.jobs.Ollama.image_ready'),patch('backend.app.jobs.subprocess.Popen',side_effect=start):
                        queue._execute(session,jobs[0])
                        first=store.job(session,jobs[0])['state']
                        self.assertEqual(first['status'],'FAILED')
                        self.assertIn('완료 보고서',first['error'])
                        self.assertIsNone(queue.process)
                        self.assertEqual(queue.image_model_status()['state'],'released')
                        self.assertTrue(processes[0].stdin.closed)
                        self.assertIsNotNone(processes[0].poll())
                        queue._execute(session,jobs[1])
                        self.assertEqual(store.job(session,jobs[1])['state']['status'],'SUCCEEDED')
                        self.assertEqual(len(processes),2)
                        self.assertIsNone(queue.failure)
                finally:queue.close()
