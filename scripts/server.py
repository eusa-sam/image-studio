"""Owned local server lifecycle: uv run --no-sync python -m scripts.server start|status|stop."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from uuid import UUID, uuid4

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[1]


def default_port():
    if os.environ.get('IMAGE_STUDIO_CONTAINER') == '1':
        return 8765
    value = '8765'
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            key, separator, setting = line.partition('=')
            if separator and key.strip() == 'IMAGE_STUDIO_PORT':
                value = setting.split('#', 1)[0].strip().strip('\"\'')
    value = os.environ.get('IMAGE_STUDIO_PORT', value)
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise ValueError('IMAGE_STUDIO_PORT must be an integer from 1 to 65535')
    return int(value)


def write(path, value):
    with FileLock(str(path)+'.lock', timeout=5):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)


def read(path):
    # Windows can deny reads while another process replaces the same state file.
    with FileLock(str(path)+'.lock', timeout=5):
        return json.loads(path.read_text(encoding='utf-8'))


def current(runtime):
    pointer = runtime / 'current.json'
    if not pointer.exists():
        return None
    owner = read(pointer)
    run_id = str(UUID(owner['run_id']))
    run = runtime / run_id
    if run.resolve().parent != runtime.resolve():
        raise ValueError('Invalid server run directory')
    return run


def status(runtime):
    run = current(runtime)
    if run is None:
        return {'state': 'unmanaged'}
    path = run / 'state.json'
    state = read(path) if path.exists() else {'state': 'starting'}
    try:
        with FileLock(run / 'alive.lock', timeout=0):
            if state['state'] in ('starting', 'running', 'stopping'):
                state = dict(state, state='not_running')
    except Timeout:
        if state['state'] in ('stopped', 'failed'):
            state = dict(state, state='stopping')
    return dict(state, log=str(run / 'server.log'), run_id=run.name)


async def serve(run, data_root, port):
    import uvicorn
    from backend.app.main import create_app
    server = uvicorn.Server(uvicorn.Config(create_app(data_root, port=port), host='127.0.0.1', port=port,
                                          access_log=False, timeout_graceful_shutdown=10))
    with FileLock(run / 'alive.lock', timeout=0):
        write(run / 'state.json', {'state': 'starting', 'pid': os.getpid(), 'port': port})
        task = asyncio.create_task(server.serve())
        phase = 'starting'
        try:
            while not task.done():
                next_phase = 'stopping' if (run / 'stop').exists() else 'running' if server.started else 'starting'
                if next_phase == 'stopping':
                    server.should_exit = True
                if next_phase != phase:
                    phase = next_phase
                    write(run / 'state.json', {'state': phase, 'pid': os.getpid(), 'port': port})
                await asyncio.sleep(.2)
            await task
        finally:
            server.should_exit = True
            if not task.done():
                await task
            write(run / 'state.json', {'state': 'stopped' if server.started else 'failed',
                                       'pid': os.getpid(), 'port': port})


def start(runtime, data_root, port):
    runtime.mkdir(parents=True, exist_ok=True)
    with FileLock(runtime / 'control.lock', timeout=0):
        before = status(runtime)
        if before['state'] in ('running', 'starting', 'stopping'):
            return before
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError('Port already in use; existing server was not started by this manager. No process was stopped.')
        run = runtime / str(uuid4())
        run.mkdir()
        write(runtime / 'current.json', {'run_id': run.name})
        with (run / 'server.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-m', 'scripts.server', '_serve',
                '--runtime', str(runtime), '--data-root', str(data_root), '--port', str(port), '--run-id', run.name],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target=process.wait, daemon=True).start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = status(runtime)
            if state['state'] == 'running':
                return state
            if process.poll() is not None:
                raise RuntimeError('Server startup failed; see ' + str(run / 'server.log'))
            time.sleep(.1)
        (run / 'stop').touch(exist_ok=True)
        raise RuntimeError('Startup timed out; graceful stop requested. See ' + str(run / 'server.log'))


def stop(runtime):
    if not runtime.exists():
        return {'state': 'unmanaged'}
    with FileLock(runtime / 'control.lock', timeout=0):
        state = status(runtime)
        if state['state'] not in ('starting', 'running', 'stopping'):
            return state
        (current(runtime) / 'stop').touch(exist_ok=True)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            state = status(runtime)
            if state['state'] not in ('starting', 'running', 'stopping'):
                return state
            time.sleep(.1)
        raise RuntimeError('Graceful shutdown is still pending; no process was force-killed. See ' + state['log'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop', '_serve'))
    parser.add_argument('--runtime', type=Path, default=ROOT / 'data/runtime/server')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--port', type=int)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    runtime, data_root = args.runtime.resolve(), args.data_root.resolve()
    try:
        if args.action in ('start', '_serve'):
            args.port = args.port if args.port is not None else default_port()
            if not 1 <= args.port <= 65535:
                raise ValueError('Port must be from 1 to 65535')
        if args.action == '_serve':
            run = runtime / str(UUID(args.run_id))
            if run != current(runtime):
                raise ValueError('Server run ownership changed')
            asyncio.run(serve(run, data_root, args.port))
            return
        if args.action=='start':
            runtime.mkdir(parents=True,exist_ok=True)
            with FileLock(runtime/'update.lock',timeout=0):
                result=start(runtime,data_root,args.port)
        else:result=stop(runtime) if args.action=='stop' else status(runtime)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, Timeout) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
