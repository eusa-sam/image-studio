import json
from pathlib import Path
import socket
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

from scripts.server import ROOT, current, start, status, stop, write
from backend.app.storage import SessionStore


class ServerControlTests(unittest.TestCase):
    def test_state_reader_and_writer_share_file_lock(self):
        from concurrent.futures import ThreadPoolExecutor
        from filelock import FileLock
        from scripts.server import read
        import threading
        with tempfile.TemporaryDirectory(dir=ROOT/'data/runtime') as directory:
            path=Path(directory)/'state.json';write(path,{'state':'starting'})
            for operation in (lambda:read(path),lambda:write(path,{'state':'running'})):
                entered=threading.Event()
                def run():
                    entered.set();return operation()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    with FileLock(str(path)+'.lock',timeout=0):
                        future=pool.submit(run)
                        self.assertTrue(entered.wait(2))
                        with self.assertRaises(TimeoutError):future.result(timeout=.1)
                    future.result(timeout=2)
            self.assertEqual(read(path),{'state':'running'})

    def test_real_start_duplicate_stop_and_store_release(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/runtime') as directory:
            root=Path(directory);runtime=root/'server';data=root/'data'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            try:
                first=start(runtime,data,port)
                self.assertEqual(first['state'],'running')
                self.assertEqual(start(runtime,data,port)['pid'],first['pid'])
                request=urllib.request.Request(f'http://127.0.0.1:{port}/api/settings',headers={'Host':'127.0.0.1:8765'})
                with urllib.request.urlopen(request,timeout=5) as response:
                    self.assertEqual(response.status,200)
                    self.assertIn('model_path',json.load(response))
                ended=stop(runtime)
                self.assertEqual(ended['state'],'stopped')
                self.assertEqual(stop(runtime)['state'],'stopped')
                with socket.socket() as sock:self.assertNotEqual(sock.connect_ex(('127.0.0.1',port)),0)
                with SessionStore(data) as store:self.assertIsNotNone(store)
            finally:
                stop(runtime)

    def test_other_listener_is_not_stopped_and_invalid_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/runtime') as directory:
            runtime=Path(directory)/'server'
            self.assertEqual(stop(runtime),{'state':'unmanaged'})
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
                with patch('scripts.server.subprocess.Popen') as launch:
                    with self.assertRaisesRegex(RuntimeError,'Port already in use'):start(runtime,Path(directory)/'data',port)
                    launch.assert_not_called()
                with socket.socket() as probe:self.assertEqual(probe.connect_ex(('127.0.0.1',port)),0)
            write(runtime/'current.json',{'run_id':'../../outside'})
            with self.assertRaises(ValueError):current(runtime)

    def test_stale_run_does_not_claim_to_be_running(self):
        from uuid import uuid4
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/runtime') as directory:
            runtime=Path(directory);run=runtime/str(uuid4());run.mkdir()
            write(runtime/'current.json',{'run_id':run.name})
            write(run/'state.json',{'state':'running','pid':123})
            self.assertEqual(status(runtime)['state'],'not_running')
            self.assertEqual(stop(runtime)['state'],'not_running')
            self.assertFalse((run/'stop').exists())
