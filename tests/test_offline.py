import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OfflineTests(unittest.TestCase):
    def test_dns_tcp_and_udp_are_blocked_in_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            log=Path(temporary)/'audit.jsonl'
            code='''
import socket,sys
from pathlib import Path
from backend.app.offline import install_guard
install_guard(Path(sys.argv[1]))
import os
expected=Path.cwd()/'data/cache'
for variable,name in [('CUDA_CACHE_PATH','cuda'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton')]:
 assert Path(os.environ[variable])==expected/name
for action in [lambda:socket.getaddrinfo('example.com',443), lambda:socket.socket().connect(('127.0.0.1',9)),lambda:socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(b'x',('127.0.0.1',9))]:
 try:action()
 except PermissionError:pass
 else:raise AssertionError('Network call was not blocked')
'''
            result=subprocess.run([sys.executable,'-c',code,str(log)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['guard_installed','socket.getaddrinfo','socket.connect','socket.sendto'])
