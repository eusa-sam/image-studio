"""Disposable worker status; session state remains the authority for completion."""
import json
import os
import time
from pathlib import Path

STAGES={'verify','load','condition','generate','decode','detect','detail','upscale','save'}
MODES={'full','partial','model_offload','sequential_offload','auto_model_offload','denoiser_resident','block_offload'}


class Progress:
    def __init__(self, folder):
        self.path=Path(folder)/'progress.json'
        self.policy=None
        self.warned=False
        self.stage=None
        self.stage_started=time.monotonic()
        self.completed_seconds={}

    def timings(self):
        result=dict(self.completed_seconds)
        if self.stage:
            result[self.stage]=result.get(self.stage,0)+time.monotonic()-self.stage_started
        return result

    def update(self, stage, **details):
        if stage not in STAGES:raise ValueError('Unknown progress stage')
        if stage!=self.stage:
            now=time.monotonic()
            if self.stage:
                self.completed_seconds[self.stage]=self.completed_seconds.get(self.stage,0)+now-self.stage_started
            self.stage,self.stage_started=stage,now
        row=dict(stage=stage,**details)
        if self.policy:row['memory_policy']=self.policy
        temporary=self.path.with_suffix('.tmp')
        try:
            temporary.write_text(json.dumps(row),encoding='utf-8')
            os.replace(temporary,self.path)
        except OSError as error:
            # Progress is advisory: an unavailable status file must not discard a valid image.
            if not self.warned:print('Progress unavailable: '+str(error),flush=True)
            self.warned=True


def read_progress(root, job):
    if job['status']!='RUNNING':return None
    path=Path(root)/'runtime/jobs'/job['id']/'progress.json'
    try:
        with path.open('rb') as stream:data=stream.read(8193)
        if len(data)>8192:return None
        row=json.loads(data)
        if not isinstance(row,dict) or row.get('stage') not in STAGES:return None
        result={'stage':row['stage']}
        for key in ('step','total','region','regions'):
            value=row.get(key)
            if type(value) is int and 0<=value<=1_000_000:result[key]=value
        if row.get('target') in ('face','hand'):result['target']=row['target']
        if row.get('decoding') is True:result['decoding']=True
        policy=row.get('memory_policy')
        if isinstance(policy,dict) and policy.get('mode') in MODES:
            result['memory_policy']={'mode':policy['mode']}
            for key in ('free_vram_bytes','resident_budget_bytes','cpu_cache_budget_bytes'):
                value=policy.get(key)
                if type(value) is int and 0<=value<=2**50:result['memory_policy'][key]=value
        return result
    except (OSError,ValueError,TypeError):return None
