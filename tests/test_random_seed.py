import tempfile
import unittest
import json
import io
from pathlib import Path
from unittest.mock import patch, Mock
from uuid import uuid4

from backend.app.jobs import JobQueue
from backend.app.storage import SessionStore
from backend.app.models import Settings


class RandomSeedTests(unittest.TestCase):
    def test_worker_revalidation_uses_the_saved_random_seed(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store, patch.object(JobQueue,'_loop'):
            queue=JobQueue(store)
            try:
                session=store.create_session('worker seed')
                settings=Settings(model='test',random_seed=True).model_dump()
                receipt=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='',
                    settings=settings,request_id=str(uuid4()),generate=True)
                job=receipt['job_id']
                with patch('backend.app.jobs.secrets.randbits',return_value=987):queue.enqueue(session,job,{'settings':settings,'runner':'sdxl_txt2img'})
                work=Path(directory)/'runtime/jobs'/job;work.mkdir(parents=True)
                Image.new('RGB',(8,8),'red').save(work/'image.png');(work/'report.json').write_text('{}')
                (work/'worker-done.json').write_text(json.dumps({'code':0,'cpu_cached':False}))
                process=Mock();process.wait.return_value=0;process.stdin=io.BytesIO()
                with patch('backend.app.jobs.preflight',side_effect=lambda s:{'settings':s,'runner':'sdxl_txt2img'}),patch('backend.app.jobs.Ollama.image_ready'),patch('backend.app.jobs.subprocess.Popen',return_value=process):
                    queue._execute(session,job)
                self.assertEqual(store.job(session,job)['state']['status'],'SUCCEEDED')
                self.assertEqual(json.loads((work/'request.json').read_text())['plan']['settings']['seed'],987)
            finally:queue.close()

    def test_each_job_resolves_once_and_fixed_seed_stays_fixed(self):
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store:
            session=store.create_session('seeds')
            with patch.object(JobQueue,'_loop'), patch('backend.app.jobs.secrets.randbits',side_effect=[123,456]) as random:
                queue=JobQueue(store)
                try:
                    for enabled,expected in [(True,123),(True,456),(False,42)]:
                        settings=Settings(model='test',random_seed=enabled).model_dump()
                        receipt=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='scene',negative='',
                            settings=settings,request_id=str(uuid4()),generate=True)
                        plan={'settings':settings}
                        for _ in range(2):queue.enqueue(session,receipt['job_id'],plan)
                        saved=store.job(session,receipt['job_id'])
                        self.assertEqual(saved['state']['plan']['settings']['seed'],expected)
                        self.assertEqual(saved['settings']['seed'],42)
                        self.assertEqual(settings['seed'],42)
                    self.assertEqual(random.call_count,2)
                finally:queue.close()
            with patch.object(JobQueue,'_loop'), patch('backend.app.jobs.secrets.randbits') as random:
                restarted=JobQueue(store)
                self.assertEqual(len(restarted.pending),3)
                random.assert_not_called()
                restarted.close()
