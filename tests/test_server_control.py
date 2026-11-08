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
    def test_native_port_defaults_file_environment_and_container(self):
        from scripts.server import default_port
        with tempfile.TemporaryDirectory() as directory, patch('scripts.server.ROOT', Path(directory)), patch.dict('os.environ', {}, clear=True):
            path=Path(directory)/'.env'
            self.assertEqual(default_port(),8765)
            path.write_text('# Local settings\nOTHER=value\nIMAGE_STUDIO_PORT="18765" # development\n',encoding='utf-8')
            self.assertEqual(default_port(),18765)
            with patch.dict('os.environ',{'IMAGE_STUDIO_PORT':'23456'}):
                self.assertEqual(default_port(),23456)
            for invalid in ('0','65536','abc','18.765',''):
                with patch.dict('os.environ',{'IMAGE_STUDIO_PORT':invalid}):
                    with self.assertRaises(ValueError):default_port()
            with patch.dict('os.environ',{'IMAGE_STUDIO_CONTAINER':'1'}):
                self.assertEqual(default_port(),8765)

    def test_port_specific_host_and_origin(self):
        from fastapi.testclient import TestClient
        from backend.app.main import create_app
        for port in (8765,18765):
            client=TestClient(create_app(port=port),base_url=f'http://127.0.0.1:{port}')
            self.assertEqual(client.get('/',headers={'Origin':f'http://localhost:{port}'}).status_code,200)
            other=18765 if port==8765 else 8765
            for headers in ({'Host':f'localhost:{other}'},{'Origin':f'http://localhost:{other}'},
                            {'Host':f'evil.example:{port}'},{'Origin':f'https://localhost:{port}'}):
                self.assertEqual(client.get('/',headers=headers).status_code,403)
            client.close()

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
                request=urllib.request.Request(f'http://127.0.0.1:{port}/api/settings',headers={'Origin':f'http://127.0.0.1:{port}'})
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
