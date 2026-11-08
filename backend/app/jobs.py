"""Single GPU queue backed by session job files; only this process writes records."""
import json
import os
import subprocess
import sys
import threading
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import ROOT, preflight
from .storage import ConflictError, encode, digest, ARTIFACT_NAMES
from .ollama import Ollama

ACTIVE = ('AWAITING_VALIDATION', 'QUEUED', 'RUNNING')


def execution_plan(store, session, settings):
    plan = preflight(settings)
    if plan['settings'].get('ip_adapter') or plan['settings'].get('flow')=='reference_edit':
        from backend.app.inference.conditioning.ip_adapter import reference_image
        _,plan['reference']=reference_image(store,session,plan['settings'])
    if plan.get('runner','').endswith(('_img2img','_inpaint')):
        from PIL import Image
        artifact = plan['settings'].get('source_artifact','image.png')
        path = store.artifact(session, plan['settings']['source_job'], artifact)
        with path.open('rb') as stream:
            content = stream.read(32 * 1024 * 1024 + 1)
        if len(content) > 32 * 1024 * 1024:
            raise ValueError('입력 이미지가 32 MiB를 초과합니다.')
        import io
        with Image.open(io.BytesIO(content)) as image:
            if image.format != 'PNG' or max(image.size) > 4096:
                raise ValueError('입력 이미지는 4096px 이하 PNG여야 합니다.')
            size = list(image.size)
            image.verify()
        plan['source'] = {'job_id':plan['settings']['source_job'], 'artifact':artifact, 'sha256':digest(content), 'size':size}
        if plan['runner'].endswith('_inpaint'):
            from .masks import read_mask
            _,plan['mask']=read_mask(store,session,plan['settings']['mask_id'],plan['settings']['source_job'],artifact)
    return plan


def original_artifacts(work, terminal_status=None):
    from PIL import Image
    original = work / 'original.png'
    if not original.exists():
        return {}
    with Image.open(original) as image:
        image.verify()
    report = work / 'report.json'
    try:
        report_bytes = report.read_bytes()
        parsed = json.loads(report_bytes)
        if not isinstance(parsed,dict) or not isinstance(parsed.get('stages',[]),list) or any(not isinstance(stage,dict) for stage in parsed.get('stages',[])):
            raise ValueError('Invalid generation report structure')
        if terminal_status:
            for stage in parsed.get('stages',[]):
                if stage.get('status') == 'RUNNING':
                    stage['status'] = terminal_status
            report_bytes=encode(parsed)
    except (OSError, ValueError) as error:
        report_bytes = encode({'schema_version':1,'status':'INTERRUPTED_AFTER_GENERATION','report_error':str(error)})
    artifacts = {'original.png':original.read_bytes(), 'report.json':report_bytes}
    for name in ARTIFACT_NAMES:
        if not name.startswith('detail') or not (work/name).exists():
            continue
        try:
            with Image.open(work/name) as image:
                image.verify()
            artifacts[name]=(work/name).read_bytes()
        except (OSError,ValueError):
            pass  # A cancelled partial PNG must not hide the valid original.
    return artifacts


class JobQueue:
    def __init__(self, store):
        self.store = store
        self.ollama=Ollama()
        self.output_path=store.root/'outputs'
        self.lock = threading.RLock()
        self.gpu = threading.Lock()
        self.stop = threading.Event()
        self.pending = []
        self.process = None
        self.persistent = False
        self.service = None
        self.suspended = False
        self.idle_since = None
        self.idle_checked_at = 0
        self.current = None
        self.failure = None
        for session in store.sessions(tolerate_errors=True):
            for job in store.jobs(session['ref'],tolerate_errors=True):
                if job['status'] == 'QUEUED':
                    self.pending.append((session['ref'], job['id']))
                elif job['status'] in ACTIVE:
                    store.transition(session['ref'], job['id'], ACTIVE, 'INTERRUPTED',
                                     details={'error': '서버가 종료되어 중단되었습니다. 새 작업으로 다시 생성하세요.'})
                elif job['status']=='SUCCEEDED' and job.get('output_status')=='PENDING':
                    store.transition(session['ref'],job['id'],('SUCCEEDED',),'SUCCEEDED',details={
                        'output_status':'INTERRUPTED','output_error':'출력 복사 중 서버가 종료되었습니다. Workspace 원본은 보존됐습니다. 출력 폴더를 확인하세요.'})
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def enqueue(self, session, job_id, plan):
        with self.lock:
            state = self.store.job(session, job_id)['state']
            if state['status'] == 'AWAITING_VALIDATION':
                if plan.get('settings', {}).get('random_seed', False):
                    plan = {**plan, 'settings': {**plan['settings'], 'seed': secrets.randbits(32)}}
                self.store.transition(session, job_id, ACTIVE, 'QUEUED', details={'plan': plan})
                self.pending.append((session, job_id))

    def cancel(self, session, job_id):
        with self.lock:
            state = self.store.job(session, job_id)['state']
            if state['status'] not in ACTIVE:
                return state
            if self.current == (session, job_id) and self.process:
                self._terminate()
            return self.store.transition(session, job_id, ACTIVE, 'CANCELLED')

    def close(self):
        self.stop.set()
        with self.lock:
            if self.process and self.process.poll() is None:
                self._terminate()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Worker coordinator did not stop')

    def _terminate(self):
        # Windows venv python is a launcher: killing only its PID can orphan the interpreter.
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               capture_output=True, timeout=15, check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.process.kill()
            self.process.wait(timeout=15)

    def _loop(self):
        try:
            while not self.stop.wait(0.2):
                with self.lock:
                    item = self.pending.pop(0) if self.pending else None
                if item:
                    self._execute(*item)
                else:self.maintain_idle_model()
        except Exception as exc:
            self.failure = str(exc)

    def maintain_idle_model(self):
        now=time.monotonic()
        with self.lock:
            if self.idle_since is None or now-max(self.idle_since,self.idle_checked_at)<10:return
        if not self.gpu.acquire(blocking=False):return
        try:
            with self.lock:
                if self.current is not None or self.idle_since is None or not self.persistent or not self.process or self.process.poll() is not None:return
                self.idle_checked_at=now
            from backend.app.inference.runtime.memory_policy import driver_memory, available_ram
            try:
                low_vram=driver_memory()[0]<2*1024**3
                low_ram=available_ram()<8*1024**3
            except (OSError,ValueError,subprocess.SubprocessError):
                self.release_image_model()
                return
            if low_ram:self.release_image_model()
            elif low_vram and not self.suspended:self.suspend_image_model()
        finally:self.gpu.release()

    def suspend_image_model(self):
        # The caller holds gpu until the worker acknowledges VRAM release.
        with self.lock:
            if self.current is not None:raise RuntimeError('Cannot suspend an active image job')
            if self.suspended:return
            if not self.persistent or not self.process or self.process.poll() is not None or self.service is None:
                self.release_image_model()
                return
            token=secrets.token_hex(16)
            done=self.service/('suspend-'+token+'.json')
            try:
                self.process.stdin.write((json.dumps(dict(command='suspend',token=token))+'\n').encode())
                self.process.stdin.flush()
                deadline=time.monotonic()+30
                while not done.exists() and self.process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                result=json.loads(done.read_text(encoding='utf-8')) if done.exists() else {}
                if isinstance(result,dict) and type(result.get('code')) is int and result['code']==0 and result.get('cached') is True:
                    self.suspended=True
                    return
            except (OSError,ValueError):pass
            self.release_image_model()

    def release_image_model(self):
        # Call while holding gpu; chat/model switches must not overlap a warm worker.
        with self.lock:
            if self.current is not None:raise RuntimeError('Cannot release an active image job')
            if self.process and self.process.poll() is None:self._terminate()
            self.process=None
            self.persistent=False
            self.idle_since=None
            self.suspended=False
            self.service=None

    def image_model_status(self):
        with self.lock:
            live=self.process is not None and self.process.poll() is None
            return {'state':'running' if self.current is not None else ('cpu' if self.suspended else 'warm') if live and self.persistent else 'released'}

    def _execute(self, session, job_id):
        while not self.stop.is_set():
            if self.gpu.acquire(timeout=0.2):
                try:
                    self._run_image(session, job_id)
                finally:
                    self.gpu.release()
                return

    def _run_image(self, session, job_id):
        work = self.store.root / 'runtime/jobs' / job_id
        try:
            with self.lock:
                job = self.store.job(session, job_id)
                if job['state']['status'] != 'QUEUED' or self.stop.is_set():
                    return
                self.ollama.image_ready()
                replay_settings = dict(job['settings'])
                if replay_settings.get('random_seed', False):
                    replay_settings['seed'] = job['state']['plan']['settings']['seed']
                plan = execution_plan(self.store, session, replay_settings)
                if plan != job['state']['plan']:
                    raise ValueError('모델 또는 실행 구성이 변경되었습니다. 새 작업으로 생성하세요.')
                work.mkdir(parents=True, exist_ok=True)
                if 'source' in plan:
                    source = self.store.artifact(session, plan['source']['job_id'], plan['source'].get('artifact','image.png')).read_bytes()
                    if digest(source) != plan['source']['sha256']:
                        raise ValueError('SOURCE_CHANGED')
                    (work / 'source.png').write_bytes(source)
                if 'mask' in plan:
                    from .masks import read_mask
                    mask,_=read_mask(self.store,session,plan['mask']['id'],plan['source']['job_id'],plan['source'].get('artifact','image.png'))
                    (work/'mask.png').write_bytes(mask.read_bytes())
                if 'reference' in plan:
                    from backend.app.inference.conditioning.ip_adapter import reference_image
                    reference,meta=reference_image(self.store,session,plan['settings'])
                    data=reference.read_bytes()
                    if meta!=plan['reference'] or digest(data)!=meta['sha256']:raise ValueError('REFERENCE_CHANGED')
                    (work/'reference.png').write_bytes(data)
                request = work / 'request.json'
                from .models import MODEL_ROOT
                request.write_text(json.dumps(dict(plan=plan, positive=job['positive'], negative=job['negative'],model_root=str(MODEL_ROOT))), encoding='utf-8')
                self.store.transition(session, job_id, ('QUEUED',), 'RUNNING')
                self.current = (session, job_id)
                keep=plan['runner'].startswith(('krea2_','flux2_','zimage_','sdxl_','sd1_','sd2_','anima_'))
                if self.process and (not keep or not self.persistent or self.process.poll() is not None):
                    if self.process.poll() is None:self._terminate()
                    self.process=None
                if keep:
                    if self.process is None:
                        service=self.store.root/'runtime'/('worker-'+secrets.token_hex(8))
                        service.mkdir(parents=True)
                        self.service=service
                        with (service/'lifecycle.log').open('wb') as log:
                            self.process=subprocess.Popen([sys.executable,'-m','backend.app.worker_service',str(service),str(self.store.root)],
                                cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    self.persistent=True
                    self.idle_since=None
                    self.suspended=False
                    self.process.stdin.write((json.dumps(str(request))+'\n').encode('utf-8'))
                    self.process.stdin.flush()
                else:
                    self.persistent=False
                    with (work / 'worker.log').open('wb') as log:
                        self.process = subprocess.Popen([sys.executable, '-m', 'backend.app.worker', str(request)],
                            cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                process=self.process
            if keep:
                done=work/'worker-done.json'
                while not done.exists() and process.poll() is None:time.sleep(.1)
                try:
                    completion=json.loads(done.read_text(encoding='utf-8')) if done.exists() else {'code':process.returncode or 1}
                    if not isinstance(completion,dict) or type(completion.get('code')) is not int or ('cpu_cached' in completion and type(completion['cpu_cached']) is not bool):
                        raise ValueError('Invalid completion fields')
                    code=completion['code']
                except (OSError,ValueError,TypeError) as error:
                    with self.lock:
                        if process.poll() is None:self._terminate()
                        self.persistent=False
                    raise ValueError('이미지 worker 완료 보고서가 손상되었거나 읽을 수 없습니다.') from error
            else:code=process.wait()
            with self.lock:
                if keep and code==0:self.suspended=completion.get('cpu_cached') is True
                if keep and code!=0 and process.poll() is None:
                    self._terminate()
                    self.persistent=False
                state = self.store.job(session, job_id)['state']
                if state['status'] == 'CANCELLED':
                    artifacts = original_artifacts(work,'CANCELLED')
                    if artifacts:
                        self.store.transition(session, job_id, ('CANCELLED',), 'CANCELLED',
                            details={'has_original':True}, artifacts=artifacts)
                    return
                if self.stop.is_set():
                    artifacts = original_artifacts(work,'INTERRUPTED')
                    self.store.transition(session, job_id, ('RUNNING',), 'INTERRUPTED',
                        details={'has_original':bool(artifacts)}, artifacts=artifacts)
                elif code != 0:
                    error = f'이미지 worker가 종료됐습니다 (exit code {code}).'
                    logs=[work/'worker.log']
                    if keep and self.service is not None:logs.append(self.service/'lifecycle.log')
                    for logfile in logs:
                        try:
                            with logfile.open('rb') as stream:
                                stream.seek(0,os.SEEK_END);stream.seek(max(0,stream.tell()-4000))
                                detail=stream.read(4000).decode('utf-8',errors='replace').strip()
                        except OSError:continue
                        if detail:
                            error+='\n'+detail
                            break
                    artifacts = original_artifacts(work,'FAILED')
                    self.store.transition(session, job_id, ('RUNNING',), 'PARTIAL_FAILED' if artifacts else 'FAILED',
                        details={'error': error, 'has_original':bool(artifacts)}, artifacts=artifacts)
                else:
                    from PIL import Image
                    with Image.open(work / 'image.png') as image:
                        image.verify()
                    self.store.transition(session, job_id, ('RUNNING',), 'SUCCEEDED',
                        details={'has_original':(work / 'original.png').exists(),'output_status':'PENDING'},
                        artifacts={**original_artifacts(work), **{name: (work / name).read_bytes() for name in ('image.png', 'report.json')}})
                    self._export_result(session,job_id)
        except (OSError, ValueError, RuntimeError) as exc:
            # A broken storage transaction must remain recoverable, never be overwritten as FAILED.
            if self.store._broken:
                raise
            with self.lock:
                try:
                    error=str(exc)
                    try:artifacts=original_artifacts(work,'FAILED')
                    except (OSError,ValueError) as original_error:
                        artifacts={}
                        error+='\n원본 복구 실패: '+str(original_error)
                    self.store.transition(session, job_id, ACTIVE, 'PARTIAL_FAILED' if artifacts else 'FAILED',
                        details={'error':error,'has_original':bool(artifacts)},artifacts=artifacts)
                except ConflictError:
                    pass
        finally:
            with self.lock:
                if self.persistent and self.process and self.process.poll() is None:
                    self.idle_since=time.monotonic()
                else:
                    if self.process and self.process.stdin:self.process.stdin.close()
                    self.process=None
                    self.persistent=False
                self.current = None

    def retry_export(self,session,job_id):
        if not self.gpu.acquire(blocking=False):
            raise ConflictError('진행 중인 작업이 끝난 뒤 출력 저장을 재시도하세요.')
        try:
            with self.lock:
                if self.current or self.pending:
                    raise ConflictError('대기 작업이 끝난 뒤 출력 저장을 재시도하세요.')
                state=self.store.job(session,job_id)['state']
                if state['status']!='SUCCEEDED' or state.get('output_status') not in ('FAILED','INTERRUPTED'):
                    raise ConflictError('출력 저장이 실패하거나 중단된 완료 작업만 재시도할 수 있습니다.')
                self.store.transition(session,job_id,('SUCCEEDED',),'SUCCEEDED',details={
                    'output_status':'PENDING','output_error':None,
                    'output_retry_started_at':datetime.now(timezone.utc).isoformat()})
                return self._export_result(session,job_id,recover_existing=True)
        finally:self.gpu.release()

    def _export_result(self,session,job_id,*,recover_existing=False):
        from .app_settings import export_images
        try:
            exported=export_images(self.store,session,job_id,self.output_path,recover_existing=recover_existing)
            details={'output_path':exported,'output_error':None,'output_status':'SUCCEEDED'}
        except (OSError,ValueError) as error:
            details={'output_error':str(error),'output_status':'FAILED'}
        details['output_finished_at']=datetime.now(timezone.utc).isoformat()
        return self.store.transition(session,job_id,('SUCCEEDED',),'SUCCEEDED',details=details)
