"""Single-writer session files with recoverable prompt/job commits."""
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from filelock import FileLock

ARTIFACT_NAMES = ('image.png','original.png','report.json','detailed.png','detail-mask.png') + tuple(
    f'detail-{i}-{side}.png' for i in range(8) for side in ('before','after'))


class ConflictError(ValueError):
    pass


class RecoveryRequired(RuntimeError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def identifier(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError('Expected canonical UUID')
    return value


class SessionStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        runtime = self._path('runtime')
        runtime.mkdir(exist_ok=True)
        self._lock = FileLock(str(runtime / 'writer.lock'))
        self._lock.acquire(timeout=0)
        self._mutex = threading.RLock()
        self._closed = False
        self._broken = False
        self.read_errors = {}
        try:
            self.recover()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._mutex:
            if not self._closed:
                self._closed = True
                self._lock.release()

    def _path(self, relative):
        p = PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
            raise ValueError('Invalid relative path')
        target = self.root.joinpath(*p.parts)
        if not target.resolve().is_relative_to(self.root):
            raise ValueError('Path escapes storage root')
        return target

    def _session_path(self, session):
        parts = session.split('/')
        if len(parts) != 2 or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', parts[0]):
            raise ValueError('Invalid session reference')
        identifier(parts[1])
        return f'sessions/{session}'

    def _ready(self):
        if self._closed:
            raise RuntimeError('Store closed')
        if self._broken:
            raise RecoveryRequired('Recover pending writes before reading or writing')

    def _replace(self, relative, content):
        target = self._path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.' + str(uuid4()) + '.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _hash(self, relative):
        path = self._path(relative)
        return digest(path.read_bytes()) if path.exists() else None

    def _apply(self, operation):
        manifest = json.loads(self._path(operation + '/manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), list):
            raise RecoveryRequired('Unsupported manifest')
        prepared = []
        seen = set()
        for row in manifest['files']:
            relative = row['path']
            parts = PurePosixPath(relative).parts
            if len(parts) < 4 or parts[0] != 'sessions' or relative in seen:
                raise RecoveryRequired('Invalid transaction target')
            self._session_path('/'.join(parts[1:3]))
            seen.add(relative)
            content = self._path(operation + '/payload/' + str(len(prepared))).read_bytes()
            if digest(content) != row['after']:
                raise RecoveryRequired('Corrupt transaction payload')
            current = self._hash(relative)
            if current not in (row['before'], row['after']):
                raise RecoveryRequired(f'External edit conflicts with recovery: {relative}')
            prepared.append((relative, content, row['after']))
        for relative, content, expected in prepared:
            if self._hash(relative) != expected:
                self._replace(relative, content)
        self._replace(operation + '/COMMITTED.json', encode({'schema_version': 1}))

    def _commit(self, changes):
        operation = 'transactions/' + datetime.now(timezone.utc).strftime('%Y-%m') + '/' + str(uuid4())
        rows = []
        # READY is written only after every payload is durable; incomplete preparation is invisible.
        for relative, content in changes.items():
            rows.append({'path': relative, 'before': self._hash(relative), 'after': digest(content)})
            self._replace(operation + '/payload/' + str(len(rows) - 1), content)
        self._replace(operation + '/manifest.json', encode({'schema_version': 1, 'files': rows}))
        self._broken = True
        self._replace(operation + '/READY.json', encode({'schema_version': 1}))
        self._apply(operation)
        self._broken = False

    def recover(self):
        with self._mutex:
            if self._closed:
                raise RuntimeError('Store closed')
            self._broken = True
            for marker in sorted(self._path('transactions').glob('*/*/READY.json')):
                operation = marker.parent.relative_to(self.root).as_posix()
                if not self._path(operation + '/COMMITTED.json').exists():
                    self._apply(operation)
            self._broken = False

    def _read_json(self, relative):
        record = json.loads(self._path(relative).read_text(encoding='utf-8'))
        if not isinstance(record,dict) or record.get('schema_version') != 1:
            raise ValueError('Unsupported schema version')
        return record

    def create_session(self, title):
        if not isinstance(title, str):
            raise ValueError('Title must be text')
        with self._mutex:
            self._ready()
            now = datetime.now(timezone.utc)
            session_id = str(uuid4())
            bucket = now.astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m')
            session = bucket + '/' + session_id
            record = dict(schema_version=1, id=session_id, created_at=now.isoformat(), title=title, revision=1)
            self._commit({self._session_path(session) + '/session.json': encode(record)})
            return session

    def save_prompt(self, session, prompt_id, *, expected_revision, positive, negative, settings, request_id, generate=False):
        identifier(prompt_id)
        identifier(request_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError('Invalid revision')
        if not isinstance(positive, str) or not isinstance(negative, str) or not isinstance(settings, dict) or type(generate) is not bool:
            raise ValueError('Invalid prompt fields')
        request = dict(prompt_id=prompt_id, expected_revision=expected_revision, positive=positive,
                       negative=negative, settings=settings, generate=generate)
        request_hash = digest(encode(request))
        settings = json.loads(encode(settings))
        with self._mutex:
            self._ready()
            base = self._session_path(session)
            metadata = self._read_json(base + '/session.json')
            if metadata['id'] != session.split('/')[1]:
                raise ValueError('Session identity mismatch')
            receipt_path = base + '/requests/' + request_id + '.json'
            if self._path(receipt_path).exists():
                receipt = self._read_json(receipt_path)
                if receipt['request_hash'] != request_hash:
                    raise ConflictError('Request ID reused with different content')
                return receipt
            head_path = base + '/prompts/' + prompt_id + '/head.json'
            current = self._read_json(head_path)['revision'] if self._path(head_path).exists() else 0
            if current != expected_revision:
                raise ConflictError(f'Expected revision {expected_revision}, found {current}')
            revision = current + 1
            now = datetime.now(timezone.utc).isoformat()
            version = base + f'/prompts/{prompt_id}/revisions/{revision:06d}'
            if self._path(version).exists():
                raise ConflictError('Revision directory already exists')
            body = {'positive.txt': positive.encode('utf-8'), 'negative.txt': negative.encode('utf-8'), 'settings.json': encode(settings)}
            changes = {version + '/' + name: content for name, content in body.items()}
            changes[version + '/meta.json'] = encode(dict(schema_version=1, id=prompt_id, revision=revision, created_at=now, hashes={k: digest(v) for k, v in body.items()}))
            changes[head_path] = encode(dict(schema_version=1, id=prompt_id, revision=revision))
            job_id = str(uuid4()) if generate else None
            if generate:
                job = base + '/jobs/' + job_id
                changes.update({job + '/snapshot/' + name: content for name, content in body.items()})
                changes[job + '/request.json'] = encode(dict(schema_version=1, id=job_id, created_at=now, prompt_id=prompt_id, prompt_revision=revision))
                # Preflight has not resolved components yet; never enqueue an unresolved model snapshot.
                changes[job + '/state.json'] = encode(dict(schema_version=1, id=job_id, revision=1, created_at=now, status='AWAITING_VALIDATION'))
            receipt = dict(schema_version=1, id=request_id, created_at=now, request_hash=request_hash, prompt_id=prompt_id, revision=revision, job_id=job_id)
            changes[receipt_path] = encode(receipt)
            self._commit(changes)
            return receipt

    def read_prompt(self, session, prompt_id):
        identifier(prompt_id)
        with self._mutex:
            self._ready()
            base = self._session_path(session) + '/prompts/' + prompt_id
            head = self._read_json(base + '/head.json')
            version = base + f'/revisions/{head["revision"]:06d}'
            meta = self._read_json(version + '/meta.json')
            body = {}
            for name in ('positive.txt', 'negative.txt', 'settings.json'):
                content = self._path(version + '/' + name).read_bytes()
                if digest(content) != meta['hashes'][name]:
                    raise ValueError('Prompt file hash mismatch')
                body[name] = content.decode('utf-8')
            return dict(revision=head['revision'], positive=body['positive.txt'], negative=body['negative.txt'], settings=json.loads(body['settings.json']))

    def _list_records(self, scope, paths, reader, tolerate_errors):
        result=[];errors=[]
        for path in sorted(paths):
            try:
                result.append(reader(path))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                if not tolerate_errors:raise
                errors.append(dict(path=path.relative_to(self.root).as_posix(),error=str(exc)))
        if tolerate_errors:self.read_errors[scope]=errors
        return result

    def sessions(self, *, tolerate_errors=False):
        with self._mutex:
            self._ready()
            def read(path):
                ref=path.parent.relative_to(self.root/'sessions').as_posix()
                row=self._read_json(self._session_path(ref)+'/session.json')
                if row['id']!=ref.split('/')[1] or not isinstance(row['title'],str) or not isinstance(row['created_at'],str) or type(row['revision']) is not int:
                    raise ValueError('Invalid Workspace metadata')
                return dict(row,ref=ref)
            return self._list_records('sessions',(p/'session.json' for p in self._path('sessions').glob('*/*') if p.is_dir()),read,tolerate_errors)

    def update_session(self, session, *, expected_revision, title, archived):
        if type(expected_revision) is not int or expected_revision<1 or not isinstance(title,str) or not title.strip() or len(title)>200 or type(archived) is not bool:
            raise ValueError('Invalid Workspace metadata')
        with self._mutex:
            self._ready()
            path=self._session_path(session)+'/session.json'
            record=self._read_json(path)
            if record['id']!=session.split('/')[1]:raise ValueError('Session identity mismatch')
            if record['revision']!=expected_revision:raise ConflictError('Workspace가 다른 탭에서 변경되었습니다. 목록을 새로고침하세요.')
            record.update(title=title.strip(),archived=archived,revision=expected_revision+1,updated_at=datetime.now(timezone.utc).isoformat())
            self._commit({path:encode(record)})
            return dict(record,ref=session)

    def prompts(self, session, *, tolerate_errors=False):
        with self._mutex:
            self._ready()
            base=self._session_path(session)
            self._read_json(base+'/session.json')
            return self._list_records(base+'/prompts',(p/'head.json' for p in self._path(base+'/prompts').glob('*') if p.is_dir()),
                lambda p:dict(self.read_prompt(session,p.parent.name),id=p.parent.name),tolerate_errors)

    def jobs(self, session, *, tolerate_errors=False, job_id=None):
        with self._mutex:
            self._ready()
            base=self._session_path(session)
            self._read_json(base+'/session.json')
            def read(path):
                row=self._read_json(path.relative_to(self.root).as_posix())
                if row['id']!=identifier(path.parent.name) or not isinstance(row['created_at'],str) or type(row['revision']) is not int:
                    raise ValueError('Invalid job metadata')
                if row['status'] not in ('AWAITING_VALIDATION','QUEUED','RUNNING','SUCCEEDED','FAILED','PARTIAL_FAILED','CANCELLED','INTERRUPTED'):
                    raise ValueError('Unknown job status')
                if row['status'] in ('AWAITING_VALIDATION','QUEUED','RUNNING'):self.job(session,row['id'])
                return row
            paths=[self._path(base+'/jobs/'+identifier(job_id)+'/state.json')] if job_id is not None else (p/'state.json' for p in self._path(base+'/jobs').glob('*') if p.is_dir())
            scope=base+'/jobs'+('/'+identifier(job_id) if job_id is not None else '')
            return self._list_records(scope,paths,read,tolerate_errors)

    def job(self, session, job_id):
        with self._mutex:
            self._ready()
            base = self._session_path(session) + '/jobs/' + identifier(job_id)
            request = self._read_json(base + '/request.json')
            # Validate the immutable revision rather than trusting edited snapshot files.
            version = self._session_path(session) + f'/prompts/{identifier(request["prompt_id"])}/revisions/{request["prompt_revision"]:06d}'
            meta = self._read_json(version + '/meta.json')
            body = {}
            for name in ('positive.txt', 'negative.txt', 'settings.json'):
                content = self._path(base + '/snapshot/' + name).read_bytes()
                if digest(content) != meta['hashes'][name]:
                    raise ValueError('Job snapshot hash mismatch')
                body[name] = content.decode('utf-8')
            return dict(state=self._read_json(base + '/state.json'), positive=body['positive.txt'],
                        negative=body['negative.txt'], settings=json.loads(body['settings.json']))

    def transition(self, session, job_id, expected, status, *, details=None, artifacts=None):
        with self._mutex:
            self._ready()
            base = self._session_path(session) + '/jobs/' + identifier(job_id)
            state = self._read_json(base + '/state.json')
            if state['status'] not in expected:
                raise ConflictError('Job state changed')
            state.update(status=status, revision=state['revision'] + 1,
                         updated_at=datetime.now(timezone.utc).isoformat())
            if status=='RUNNING' and not state.get('started_at'):
                state['started_at']=state['updated_at']
            if status in ('SUCCEEDED','FAILED','PARTIAL_FAILED','CANCELLED','INTERRUPTED') and not state.get('finished_at'):
                state['finished_at']=state['updated_at']
            if details:
                state.update(details)
            changes = {}
            hashes = dict(state.get('artifact_hashes', {}))
            for name, content in (artifacts or {}).items():
                if name not in ARTIFACT_NAMES:
                    raise ValueError('Invalid artifact name')
                relative = base + '/artifacts/' + name
                checksum = digest(content)
                previous = self._hash(relative)
                if previous is not None and previous != checksum:
                    raise ConflictError('Existing artifact cannot be overwritten')
                if name in hashes and hashes[name] != checksum:
                    raise ConflictError('Artifact hash cannot be changed')
                hashes[name] = checksum
                changes[relative] = content
            if artifacts:
                state['artifacts'] = sorted(set(state.get('artifacts',[])) | set(artifacts))
                state['artifact_hashes'] = hashes
            changes[base + '/state.json'] = encode(state)
            self._commit(changes)
            return state

    def artifact(self, session, job_id, name):
        with self._mutex:
            self._ready()
            if name not in ARTIFACT_NAMES:
                raise ValueError('Invalid artifact name')
            base = self._session_path(session) + '/jobs/' + identifier(job_id)
            state = self._read_json(base + '/state.json')
            if state['status'] not in ('SUCCEEDED', 'PARTIAL_FAILED', 'CANCELLED', 'INTERRUPTED'):
                raise FileNotFoundError('Result not complete')
            path = self._path(base + '/artifacts/' + name)
            if not path.is_file():
                raise FileNotFoundError('Artifact not available')
            if 'artifact_hashes' in state:
                if not isinstance(state['artifact_hashes'], dict):
                    raise ValueError('Invalid artifact hashes')
                expected = state['artifact_hashes'].get(name)
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if expected != actual:
                    raise ValueError('Artifact hash mismatch')
            return path

    def chats(self, session, *, tolerate_errors=False, include_application=False, conversation_id=None):
        with self._mutex:
            self._ready()
            base = self._session_path(session)
            self._read_json(base + '/session.json')
            turns = self._list_records(base+'/messages',(p/'meta.json' for p in self._path(base+'/messages').glob('*') if p.is_dir()),
                lambda p:self._chat(session,p.parent.name,include_application=include_application),tolerate_errors)
            if conversation_id is not None:
                from .conversations import read, default_id
                read(self, session, conversation_id)
                turns = [t for t in turns if t.get('conversation_id', default_id(session)) == conversation_id]
            return sorted(turns, key=lambda row: (row['created_at'], row['id']))

    def _chat(self, session, request_id, *, include_application=False):
        base = self._session_path(session) + '/messages/' + identifier(request_id)
        record = self._read_json(base + '/meta.json')
        if record['id']!=request_id or not isinstance(record['created_at'],str) or record['status'] not in ('PENDING','SUCCEEDED','FAILED','CANCELLED'):
            raise ValueError('Invalid chat metadata')
        context_bytes = self._path(base + '/context/settings.json').read_bytes()
        if digest(context_bytes) != record['hashes']['context/settings.json']:
            raise ValueError('Chat context hash mismatch')
        source = json.loads(context_bytes)
        from .conversations import default_id
        record['conversation_id']=source.get('conversation_id') or default_id(session)
        record['image_context']=source.get('image_context')
        record['user_references']=source.get('user_references',[])
        record['think']=source.get('think')
        record['context'] = {k:source[k] for k in ('session','prompt_id','expected_revision','settings')}
        for field in ('positive','negative'):
            name = 'context/' + field + '.txt'
            content = self._path(base + '/' + name).read_bytes()
            if digest(content) != record['hashes'][name]:
                raise ValueError('Chat context hash mismatch')
            record['context'][field] = content.decode('utf-8')
        for name, field in [('user.txt','message'), ('assistant.txt','reply'),
                            ('positive.txt','positive'), ('negative.txt','negative'), ('memory.txt','memory'), ('reasoning.txt','reasoning')]:
            if name in record['hashes']:
                data = self._path(base + '/' + name).read_bytes()
                if digest(data) != record['hashes'][name]:
                    raise ValueError('Chat file hash mismatch')
                record[field] = data.decode('utf-8')
        if include_application and record['status']=='SUCCEEDED' and record.get('update'):
            try:
                receipt_path=self._session_path(session)+'/requests/'+request_id+'.json'
                if self._path(receipt_path).exists():
                    receipt=self._read_json(receipt_path)
                    context=record['context']
                    revision=receipt['revision']
                    if (receipt['schema_version']!=1 or receipt['id']!=request_id or receipt['prompt_id']!=context['prompt_id'] or
                            type(revision) is not int or revision!=context['expected_revision']+1):
                        raise ValueError('Applied prompt receipt mismatch')
                    if receipt['job_id'] is not None:identifier(receipt['job_id'])
                    version=self._session_path(session)+'/prompts/'+identifier(receipt['prompt_id'])+f'/revisions/{revision:06d}'
                    meta=self._read_json(version+'/meta.json')
                    if meta['id']!=receipt['prompt_id'] or meta['revision']!=revision:
                        raise ValueError('Applied prompt revision mismatch')
                    saved={}
                    for name in ('positive.txt','negative.txt','settings.json'):
                        content=self._path(version+'/'+name).read_bytes()
                        if digest(content)!=meta['hashes'][name]:raise ValueError('Applied prompt hash mismatch')
                        saved[name]=content.decode('utf-8')
                    request=dict(prompt_id=receipt['prompt_id'],expected_revision=context['expected_revision'],
                                 positive=saved['positive.txt'],negative=saved['negative.txt'],
                                 settings=json.loads(saved['settings.json']),generate=receipt['job_id'] is not None)
                    if (digest(encode(request))!=receipt['request_hash'] or request['positive']!=record['positive'] or
                            request['negative']!=record['negative']):raise ValueError('Applied proposal mismatch')
                    record['application']={key:receipt[key] for key in ('prompt_id','revision','job_id')}
            except (OSError,ValueError,KeyError,TypeError) as error:
                record['application_error']=str(error)
        return record

    def begin_chat(self, session, request_id, payload):
        with self._mutex:
            self._ready()
            base = self._session_path(session)
            self._read_json(base + '/session.json')
            path = base + '/messages/' + identifier(request_id)
            request_hash = digest(encode(payload))
            if self._path(path + '/meta.json').exists():
                record = self._chat(session, request_id)
                if record['request_hash'] != request_hash:
                    raise ConflictError('Request ID reused with different chat content')
                if record['status'] == 'PENDING':
                    raise ConflictError('대화 응답을 기다리는 중입니다.')
                return record, False
            from .conversations import begin_changes
            conversation_changes = begin_changes(self, session, payload)
            context = dict(payload)
            text = {name: context.pop(field).encode('utf-8') for name, field in
                    [('user.txt','message'), ('context/positive.txt','positive'), ('context/negative.txt','negative')]}
            text['context/settings.json'] = encode(context)
            meta = dict(schema_version=1, id=request_id, request_hash=request_hash, status='PENDING',
                        created_at=datetime.now(timezone.utc).isoformat(), prompt_id=payload['prompt_id'],
                        expected_revision=payload['expected_revision'], model=payload['model'],
                        hashes={name:digest(content) for name,content in text.items()})
            changes = {path + '/' + name:content for name,content in text.items()}
            if payload.get('conversation_id'):meta['conversation_id']=payload['conversation_id']
            changes.update(conversation_changes)
            changes[path + '/meta.json'] = encode(meta)
            self._commit(changes)
            return self._chat(session, request_id), True

    def finish_chat(self, session, request_id, *, proposal=None, error=None, model_digest=None, cancelled=False, partial_reply=None, reasoning=''):
        with self._mutex:
            self._ready()
            base = self._session_path(session) + '/messages/' + identifier(request_id)
            meta = self._read_json(base + '/meta.json')
            if meta['status'] != 'PENDING':
                raise ConflictError('Chat already completed')
            changes = {}
            if reasoning:
                data = reasoning.encode('utf-8')
                changes[base + '/reasoning.txt'] = data
                meta['hashes']['reasoning.txt'] = digest(data)
            meta.update(status='FAILED' if error else 'SUCCEEDED', updated_at=datetime.now(timezone.utc).isoformat())
            if cancelled:
                meta['status'] = 'CANCELLED'
            if error:
                meta['error'] = error
                if partial_reply:
                    data = partial_reply.encode('utf-8')
                    changes[base + '/assistant.txt'] = data
                    meta['hashes']['assistant.txt'] = digest(data)
            else:
                meta.update(update=proposal['update'], model_digest=model_digest)
                for name, field in [('assistant.txt','reply'), ('positive.txt','positive'), ('negative.txt','negative')]:
                    data = proposal[field].encode('utf-8')
                    changes[base + '/' + name] = data
                    meta['hashes'][name] = digest(data)
                if 'memory' in proposal:
                    data = proposal['memory'].encode('utf-8')
                    changes[base + '/memory.txt'] = data
                    meta['hashes']['memory.txt'] = digest(data)
            changes[base + '/meta.json'] = encode(meta)
            self._commit(changes)
            return self._chat(session, request_id)
