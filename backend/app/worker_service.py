"""Serialized image worker; EOF cancels inference and releases all CUDA memory."""
import contextlib
import json
import os
import queue
import re
import sys
import threading
import traceback
from pathlib import Path


def serve(folder,data_root=None):
    folder=Path(folder).resolve()
    from .models import ROOT
    data_root=Path(data_root or ROOT/'data').resolve()
    runtime=(data_root/'runtime').resolve()
    if not folder.is_relative_to(runtime):raise ValueError('Invalid worker directory')
    folder.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(folder/'hf-cache')
    os.environ['HF_HUB_CACHE']=str(folder/'hf-cache/hub')
    from .offline import install_guard
    install_guard(folder/'network-audit.jsonl')
    from backend.app.inference.runtime.pipeline_cache import PipelineCache
    from .worker import run
    # This host can deadlock NumPy DLL loading while another thread blocks on stdin.
    import numpy
    cache=PipelineCache(data_root/'indexes/model-verification')
    commands=queue.Queue()
    def read_commands():
        for line in sys.stdin.buffer:
            commands.put(line)
        os._exit(0)
    threading.Thread(target=read_commands,daemon=True).start()
    while True:
        command=json.loads(commands.get())
        if isinstance(command,dict):
            if set(command)!={'command','token'} or command['command']!='suspend' or not isinstance(command['token'],str) or not re.fullmatch('[0-9a-f]{32}',command['token']):
                raise ValueError('Invalid worker command')
            try:result=dict(code=0,cached=cache.suspend())
            except Exception:
                cache.clear()
                result=dict(code=1,cached=False)
                traceback.print_exc()
            target=folder/('suspend-'+command['token']+'.json')
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(result),encoding='utf-8');os.replace(temporary,target)
            continue
        request=Path(command).resolve()
        if request.name!='request.json' or not request.is_relative_to(runtime/'jobs'):
            raise ValueError('Invalid worker request')
        work=request.parent
        code=0
        with (work/'worker.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                (work/'network-audit.jsonl').write_text(json.dumps(dict(event='persistent_guard',audit=str(folder/'network-audit.jsonl')))+'\n',encoding='utf-8')
                run(request,cache)
            except Exception:
                code=1
                cache.clear()
                traceback.print_exc()
                import gc,torch
                gc.collect();torch.cuda.empty_cache()
        temporary=work/'worker-done.tmp'
        temporary.write_text(json.dumps(dict(code=code,cpu_cached=cache.suspended)),encoding='utf-8')
        os.replace(temporary,work/'worker-done.json')


if __name__=='__main__':serve(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
