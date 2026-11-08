import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from uuid import uuid4
from backend.app.storage import SessionStore
from backend.app.jobs import JobQueue


class WorkerStartupTests(unittest.TestCase):
    def test_early_exit_reports_owned_lifecycle_or_code_and_prefers_job_log(self):
        for job_log,lifecycle,expected in [(None,'ImportError: missing runtime','missing runtime'),('inference failed','old startup warning','inference failed'),(None,'','exit code 7'),('__directory__','startup failed','startup failed'),(None,'padding'*1000+'last error','last error')]:
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store,patch.object(JobQueue,'_loop'):
                session=store.create_session('worker startup')
                queue=JobQueue(store)
                try:
                    job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
                    plan=dict(runner='sdxl_txt2img',settings={})
                    store.transition(session,job,('AWAITING_VALIDATION',),'QUEUED',details={'plan':plan})
                    work=Path(directory)/'runtime/jobs'/job;work.mkdir(parents=True)
                    if job_log=='__directory__':(work/'worker.log').mkdir()
                    elif job_log is not None:(work/'worker.log').write_text(job_log,encoding='utf8')
                    process=Mock();process.poll.return_value=7;process.returncode=7;process.stdin=io.BytesIO()
                    def start(*args,**kwargs):
                        self.assertEqual(args[0][-1],str(store.root))
                        kwargs['stdout'].write(lifecycle.encode());kwargs['stdout'].flush();return process
                    with patch('backend.app.jobs.preflight',return_value=plan),patch('backend.app.jobs.Ollama.image_ready'),patch('backend.app.jobs.subprocess.Popen',side_effect=start):queue._execute(session,job)
                    state=store.job(session,job)['state']
                    self.assertEqual(state['status'],'FAILED');self.assertIn(expected,state['error'])
                    self.assertLessEqual(len(state['error']),4100);self.assertNotIn('No such file',state['error']);self.assertIsNone(queue.current)
                finally:queue.close()

    def test_custom_store_rejects_a_worker_outside_its_runtime(self):
        from backend.app.worker_service import serve
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'Invalid worker directory'):
                serve(root/'outside-worker',root/'store')
            self.assertFalse((root/'outside-worker').exists())

    def test_custom_store_rejects_requests_from_another_store(self):
        import json
        from backend.app.worker_service import serve
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=root/'store';service=store/'runtime/worker-test'
            request=root/'other/runtime/jobs/job/request.json'
            with patch('backend.app.offline.install_guard'),patch('backend.app.worker_service.threading.Thread'),patch('backend.app.worker_service.queue.Queue') as commands,patch('backend.app.inference.runtime.pipeline_cache.PipelineCache') as cache,patch('backend.app.worker.run') as run:
                commands.return_value.get.return_value=json.dumps(str(request))
                with self.assertRaisesRegex(ValueError,'Invalid worker request'):serve(service,store)
                cache.assert_called_once_with(store/'indexes/model-verification')
                run.assert_not_called()
