"""Deny Python socket egress in image workers; not an OS network sandbox."""
import json
import sys
import os
from pathlib import Path


def install_guard(log):
    cache = Path(__file__).resolve().parents[2] / 'data/cache'
    for variable, name in [('CUDA_CACHE_PATH','cuda'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton')]:
        target = cache / name
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    def record(event):
        with log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event)+'\n')
    record(dict(event='guard_installed',scope='python_socket_audit',policy='deny_all'))
    blocked={'socket.connect','socket.sendto','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr'}
    def audit(event,args):
        if event in blocked:
            record(dict(event=event,blocked=True))
            raise PermissionError('Image worker network access is disabled: '+event)
    sys.addaudithook(audit)
