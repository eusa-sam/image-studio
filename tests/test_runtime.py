import tempfile,unittest
import threading
import json
from pathlib import Path
from unittest.mock import Mock,patch
from fastapi.testclient import TestClient
from backend.app.jobs import JobQueue
from backend.app.main import create_app


class RuntimeTests(unittest.TestCase):
    def idle_queue(self):
        queue=JobQueue.__new__(JobQueue)
        queue.lock=threading.RLock();queue.gpu=threading.Lock()
        queue.idle_since=1;queue.idle_checked_at=0;queue.current=None;queue.persistent=True
        queue.service=None;queue.suspended=False
        queue.process=Mock();queue.process.poll.return_value=None;queue._terminate=Mock()
        return queue

    def test_suspend_waits_for_ack_and_keeps_cpu_cache_during_gpu_pressure(self):
        queue=self.idle_queue()
        with tempfile.TemporaryDirectory() as folder:
            queue.service=Path(folder)
            def acknowledge(data):
                command=json.loads(data)
                (queue.service/('suspend-'+command['token']+'.json')).write_text(json.dumps(dict(code=0,cached=True)))
            queue.process.stdin.write.side_effect=acknowledge
            queue.suspend_image_model();queue.suspend_image_model()
            self.assertEqual(queue.image_model_status(),{'state':'cpu'})
            queue.process.stdin.write.assert_called_once();queue._terminate.assert_not_called()
            with patch('backend.app.jobs.time.monotonic',return_value=600),patch('backend.app.memory_policy.driver_memory',return_value=(1024**3,16*1024**3)),patch('backend.app.memory_policy.available_ram',return_value=16*1024**3):
                queue.maintain_idle_model()
            queue._terminate.assert_not_called()
            queue.release_image_model();queue._terminate.assert_called_once()

    def test_suspend_failed_ack_or_transport_releases_process_before_return(self):
        for result in ({'code':1,'cached':False},{'code':0,'cached':'true'},None):
            queue=self.idle_queue()
            with tempfile.TemporaryDirectory() as folder:
                queue.service=Path(folder)
                def acknowledge(data):
                    if result is None:raise BrokenPipeError('closed')
                    command=json.loads(data)
                    (queue.service/('suspend-'+command['token']+'.json')).write_text(json.dumps(result))
                queue.process.stdin.write.side_effect=acknowledge
                queue.suspend_image_model()
                queue._terminate.assert_called_once();self.assertIsNone(queue.process)
        queue=self.idle_queue();queue.current=('session','job')
        with self.assertRaisesRegex(RuntimeError,'active'):queue.suspend_image_model()
        queue._terminate.assert_not_called()
        queue=self.idle_queue()
        with tempfile.TemporaryDirectory() as folder:
            queue.service=Path(folder)
            with patch('backend.app.jobs.time.monotonic',side_effect=[0,31]):queue.suspend_image_model()
            queue._terminate.assert_called_once();self.assertIsNone(queue.process)

    def test_elapsed_idle_time_alone_does_not_reload_model_and_checks_are_throttled(self):
        queue=self.idle_queue()
        with patch('backend.app.jobs.time.monotonic',return_value=600),patch('backend.app.memory_policy.driver_memory',return_value=(4*1024**3,16*1024**3)) as driver,patch('backend.app.memory_policy.available_ram',return_value=16*1024**3):
            queue.maintain_idle_model();queue.maintain_idle_model()
            driver.assert_called_once();queue._terminate.assert_not_called()
            self.assertEqual(queue.image_model_status(),{'state':'warm'})

    def test_malformed_suspend_ack_releases_worker_and_gpu_lock(self):
        for content in ('null','[]','"bad"','{}','{broken','{"code":false,"cached":true}','{"code":0.0,"cached":true}'):
            with self.subTest(content=content),tempfile.TemporaryDirectory() as folder:
                queue=self.idle_queue();queue.service=Path(folder)
                def acknowledge(data):
                    command=json.loads(data)
                    (queue.service/('suspend-'+command['token']+'.json')).write_text(content,encoding='utf-8')
                queue.process.stdin.write.side_effect=acknowledge
                with patch('backend.app.jobs.time.monotonic',return_value=600),patch('backend.app.memory_policy.driver_memory',return_value=(1024**3,16*1024**3)),patch('backend.app.memory_policy.available_ram',return_value=16*1024**3):
                    queue.maintain_idle_model()
                queue._terminate.assert_called_once()
                self.assertEqual(queue.image_model_status(),{'state':'released'})
                self.assertFalse(queue.suspended)
                self.assertTrue(queue.gpu.acquire(blocking=False));queue.gpu.release()

    def test_memory_pressure_and_unknown_measurement_release_only_idle_worker(self):
        for free,ram,error in [(1024**3,16*1024**3,None),(4*1024**3,7*1024**3,None),(0,0,OSError('driver unavailable'))]:
            queue=self.idle_queue()
            with patch('backend.app.jobs.time.monotonic',return_value=600),patch('backend.app.memory_policy.driver_memory',return_value=(free,16*1024**3),side_effect=error),patch('backend.app.memory_policy.available_ram',return_value=ram):
                queue.maintain_idle_model()
            queue._terminate.assert_called_once();self.assertIsNone(queue.process)
            self.assertTrue(queue.gpu.acquire(blocking=False));queue.gpu.release()

    def test_busy_gpu_and_running_jobs_never_get_evicted(self):
        for running in (False,True):
            queue=self.idle_queue()
            if running:queue.current=('workspace','job')
            else:queue.gpu.acquire()
            with patch('backend.app.jobs.time.monotonic',return_value=600),patch('backend.app.memory_policy.driver_memory') as driver:
                queue.maintain_idle_model();driver.assert_not_called();queue._terminate.assert_not_called()
            if not running:queue.gpu.release()

    def test_release_requires_idle_gpu_and_clears_warm_process(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(JobQueue,'_loop'),TestClient(create_app(folder),base_url='http://127.0.0.1:8765') as client:
            queue=client.app.state.queue
            self.assertEqual(client.get('/api/image-runtime').json(),{'state':'released'})
            queue.process=Mock();queue.process.poll.return_value=None;queue.persistent=True
            self.assertEqual(client.get('/api/image-runtime').json(),{'state':'warm'})
            with patch.object(queue,'_terminate') as terminate:
                queue.gpu.acquire()
                try:self.assertEqual(client.post('/api/image-runtime/release',json={}).status_code,409)
                finally:queue.gpu.release()
                terminate.assert_not_called()
                self.assertEqual(client.post('/api/image-runtime/release',json={}).json(),{'state':'released'})
                terminate.assert_called_once()
                self.assertIsNone(queue.process)
