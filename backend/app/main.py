"""Local web entry point: uv run python -m backend.app.main."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
import json
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .jobs import JobQueue, execution_plan
from .models import ROOT, Settings, scan
from .storage import ConflictError, SessionStore
from .chat import ChatInput, ChatService
from .ollama import OllamaError
from .prompt_tokens import count_tokens
from .app_settings import AppSettings,read_settings,validate_directories,apply_settings,settings_file


class TokenInput(BaseModel):
    mode: Literal['native','weighted'] = 'native'
    family: Literal['sdxl','sd1','sd2','anima','krea2','flux2','zimage'] = 'sdxl'
    model: str | None = None
    embeddings: list[str] = Field(default_factory=list,max_length=4)
    positive: str = Field(default='', max_length=16000)
    negative: str = Field(default='', max_length=16000)


class SessionInput(BaseModel):
    title: str = Field(default='새 Workspace', max_length=200)


class SessionUpdate(BaseModel):
    session: str
    expected_revision: int = Field(ge=1,strict=True)
    title: str = Field(min_length=1,max_length=200)
    archived: bool = Field(strict=True)


class MaskInput(BaseModel):
    session: str
    source_job: UUID
    source_artifact: Literal['image.png','original.png'] = 'image.png'
    png: str = Field(max_length=8*1024*1024)


class ReferenceInput(BaseModel):
    session: str
    filename: str = Field(min_length=1,max_length=255)
    data: str = Field(max_length=16*1024*1024)


class PromptInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session: str
    prompt_id: UUID
    request_id: UUID
    expected_revision: int = Field(ge=0, strict=True)
    positive: str = Field(max_length=16000)
    negative: str = Field(default='', max_length=16000)
    settings: Settings
    generate: bool = False


def create_app(data_root=None):
    @asynccontextmanager
    async def lifespan(app):
        with SessionStore(data_root or ROOT / 'data') as store:
            queue = JobQueue(store)
            app.state.store, app.state.queue = store, queue
            try:
                app.state.chat = ChatService(store, queue)
                from . import models,prompt_tokens
                previous_model_root=models.MODEL_ROOT
                app.state.preferences=read_settings(store)
                apply_settings(app.state.preferences,queue,app.state.chat)
                yield
            finally:
                await app.state.chat.close()
                queue.close()
                if 'previous_model_root' in locals():
                    models.MODEL_ROOT=prompt_tokens.MODEL_ROOT=previous_model_root
                    prompt_tokens.tokenizers.cache_clear()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        host = request.headers.get('host', '')
        if host not in ('127.0.0.1:8765', 'localhost:8765'):
            return JSONResponse({'detail': 'Local host required'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin not in ('http://127.0.0.1:8765', 'http://localhost:8765'):
            return JSONResponse({'detail': 'Same origin required'}, status_code=403)
        if request.method in ('POST', 'PUT', 'PATCH'):
            if request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return JSONResponse({'detail': 'JSON required'}, status_code=415)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if request.url.path == '/' or request.url.path.endswith(('.js','.mjs','.css','.html')):
            response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(ConflictError)
    async def conflict(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({'detail': '요청한 파일이 없습니다.'}, status_code=404)

    @app.get('/api/models')
    def models():
        return scan()

    @app.get('/api/settings')
    def preferences():
        return app.state.preferences.model_dump()

    @app.post('/api/settings/connection-test')
    def connection_test(body: AppSettings):
        from .ollama import Ollama
        client=Ollama(body.ollama_url,body.ollama_endpoint)
        return dict(version=client.request('version').get('version'),models=client.models())

    @app.post('/api/settings')
    def save_preferences(body: AppSettings):
        queue=app.state.queue
        if not queue.gpu.acquire(blocking=False):raise ConflictError('진행 중인 작업이 끝난 뒤 설정을 저장하세요.')
        try:
            with queue.lock:
                if queue.current or queue.pending or app.state.chat.active:raise ConflictError('진행 중인 작업이 끝난 뒤 설정을 저장하세요.')
                if body.revision!=app.state.preferences.revision:raise ConflictError('다른 창에서 설정이 변경됐습니다. 다시 불러오세요.')
                try:validate_directories(body,app.state.store)
                except OSError as error:raise ValueError('폴더에 접근하거나 저장할 수 없습니다: '+str(error)) from error
                queue.release_image_model()
                body=body.model_copy(update={'revision':body.revision+1})
                from .storage import encode
                try:
                    with app.state.store._mutex:
                        app.state.store._ready()
                        app.state.store._replace(settings_file(),encode(body.model_dump()))
                except OSError as error:
                    raise ValueError('설정 파일을 저장하지 못했습니다. 기존 설정은 유지됩니다: '+str(error)) from error
                apply_settings(body,queue,app.state.chat)
                app.state.preferences=body
                return body.model_dump()
        finally:queue.gpu.release()

    @app.post('/api/references')
    def reference_upload(body: ReferenceInput):
        import base64
        from .references import save_reference
        return save_reference(app.state.store,body.session,body.filename,base64.b64decode(body.data,validate=True))

    @app.get('/api/references/{reference_id}')
    def reference_preview(reference_id: UUID,session: str):
        from .references import read_reference
        path,_=read_reference(app.state.store,session,str(reference_id))
        return FileResponse(path)

    @app.get('/api/storage-errors')
    def storage_errors():
        with app.state.store._mutex:
            return [row for errors in app.state.store.read_errors.values() for row in errors]

    @app.get('/api/llm/model-info')
    def llm_model_info(name: str):
        _,info=app.state.chat.ollama.local_model(name)
        return dict(capabilities=info.get('capabilities',[]),thinking=info.get('thinking'))

    @app.post('/api/masks')
    def mask_upload(body: MaskInput):
        import base64
        from .masks import save_mask
        return save_mask(app.state.store,body.session,str(body.source_job),base64.b64decode(body.png,validate=True),body.source_artifact)

    @app.get('/api/masks/{mask_id}')
    def mask_image(mask_id: UUID,session: str):
        from .masks import read_mask
        path,_=read_mask(app.state.store,session,str(mask_id))
        return FileResponse(path)

    @app.post('/api/prompt-tokens')
    def prompt_tokens(body: TokenInput):
        try:
            if body.embeddings:
                from .models import MODEL_ROOT,embedding_components
                from .textual_inversion import embedding_tokenizers
                components=embedding_components(body.embeddings,scan(),body.family)
                encoders=embedding_tokenizers(body.family,[(row,MODEL_ROOT/row['path']) for row in components.values()])
                return count_tokens(body.positive,body.negative,encoders=encoders,family=body.family,mode=body.mode)
            if body.family=='krea2':
                from .krea2 import count_prompt_tokens
                return count_prompt_tokens(body.positive,body.negative)
            if body.family in ('flux2','zimage'):
                if body.family=='zimage':from .z_image import count_prompt_tokens
                else:from .flux2 import count_prompt_tokens
                from .models import MODEL_ROOT
                model=next((m for m in scan() if m['id']==body.model and m['family']==body.family and m['available']),None)
                if not model:return JSONResponse({'detail':'준비된 Klein/Z-Image bundle을 선택하세요.'},status_code=422)
                return count_prompt_tokens(MODEL_ROOT/model['path'],body.positive,body.negative)
            if body.family=='anima':
                from .anima import count_prompt_tokens
                from .models import MODEL_ROOT
                model=next((m for m in scan() if m['id']==body.model and m['family']=='anima' and m['available']),None)
                if not model:return JSONResponse({'detail':'준비된 ANIMA bundle을 선택하세요.'},status_code=422)
                return count_prompt_tokens(MODEL_ROOT/model.get('anima',{}).get('bundle',model['path']),body.positive,body.negative)
            return count_tokens(body.positive, body.negative, family=body.family,mode=body.mode)
        except OSError as exc:
            return JSONResponse({'detail':'해당 계열의 로컬 토크나이저를 준비해야 합니다.'},status_code=503)

    @app.get('/api/llm/models')
    def llm_models():
        try:
            return {'models': app.state.chat.ollama.models(), 'error': None}
        except OllamaError as exc:
            return {'models': [], 'error': str(exc)}

    @app.get('/api/chat')
    def chat_history(session: str):
        return app.state.store.chats(session,tolerate_errors=True,include_application=True)

    @app.get('/api/chat/page')
    def chat_page(session: str,limit: int=Query(default=50,ge=1,le=200),before: UUID | None=None):
        from .chat_index import page
        return page(app.state.store,session,limit=limit,before=str(before) if before is not None else None)

    @app.post('/api/chat')
    def chat(body: ChatInput):
        return app.state.chat.send(body)

    @app.post('/api/chat/stream')
    async def stream_chat(body: ChatInput):
        async def events():
            async for event in app.state.chat.stream(body):
                yield json.dumps(event,ensure_ascii=False) + '\n'
        return StreamingResponse(events(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store'})

    @app.post('/api/chat/{request_id}/cancel')
    async def cancel_chat(request_id: UUID, session: str):
        return app.state.chat.cancel(session,str(request_id))

    @app.get('/api/sessions')
    def sessions(include_archived: bool = False):
        from .workspace_index import workspaces
        return [row for row in workspaces(app.state.store) if include_archived or not row.get('archived',False)]

    @app.post('/api/sessions/update')
    def update_session(body: SessionUpdate):
        return app.state.store.update_session(**body.model_dump())

    @app.post('/api/backups')
    def backup():
        from .backup import create_backup
        return create_backup(app.state.store)

    @app.get('/api/images')
    def image_list(q: str = Query(default='', max_length=200), page: int = Query(default=1,ge=1), size: int = Query(default=24,ge=1,le=100)):
        from .image_index import images
        return images(app.state.store,query=q,page=page,size=size)

    @app.get('/api/images/{job_id}/thumbnail')
    def image_thumbnail(job_id: UUID, session: str, name: Literal['image.png','original.png','detailed.png'] = 'image.png'):
        from .image_index import thumbnail
        return Response(thumbnail(app.state.store,session,str(job_id),name),media_type='image/jpeg')

    @app.post('/api/images/rebuild')
    def rebuild_images():
        from .image_index import images
        return images(app.state.store,rebuild=True)

    @app.post('/api/sessions')
    def new_session(body: SessionInput):
        return {'ref': app.state.store.create_session(body.title)}

    @app.get('/api/prompts')
    def prompts(session: str):
        return app.state.store.prompts(session,tolerate_errors=True)

    @app.get('/api/prompts/summaries')
    def prompt_summaries(session: str):
        from .prompt_index import summaries
        return summaries(app.state.store,session)

    @app.get('/api/prompts/{prompt_id}')
    def prompt_document(prompt_id: UUID, session: str):
        return dict(app.state.store.read_prompt(session,str(prompt_id)),id=str(prompt_id))

    @app.post('/api/prompts')
    def save_prompt(body: PromptInput):
        settings = body.settings.model_dump()
        plan = execution_plan(app.state.store, body.session, settings) if body.generate else None
        if body.generate and settings['prompt_mode']=='weighted':
            family=plan['runner'].split('_')[0]
            encoders=None
            if settings['embeddings']:
                from .models import MODEL_ROOT
                from .textual_inversion import embedding_tokenizers
                encoders=embedding_tokenizers(family,[(row,MODEL_ROOT/row['path']) for key,row in plan['components'].items() if key.startswith('embedding')])
            count_tokens(body.positive,body.negative,encoders=encoders,family=family,mode='weighted')
        if app.state.queue.failure:
            return JSONResponse({'detail': '작업 큐 오류: ' + app.state.queue.failure}, status_code=503)
        receipt = app.state.store.save_prompt(body.session, str(body.prompt_id),
            expected_revision=body.expected_revision, positive=body.positive, negative=body.negative,
            settings=settings, request_id=str(body.request_id), generate=body.generate)
        if body.generate:
            app.state.queue.enqueue(body.session, receipt['job_id'], plan)
        return receipt

    @app.get('/api/jobs')
    def jobs(session: str):
        from .progress import read_progress
        rows=app.state.store.jobs(session,tolerate_errors=True)
        for row in rows:
            progress=read_progress(app.state.store.root,row)
            if progress:row['progress']=progress
        return rows

    @app.get('/api/image-runtime')
    def image_runtime():
        return app.state.queue.image_model_status()

    @app.post('/api/image-runtime/release')
    def release_image_runtime():
        queue=app.state.queue
        if not queue.gpu.acquire(blocking=False):
            raise ConflictError('생성 또는 대화 중에는 해제할 수 없습니다.')
        try:
            queue.release_image_model()
            return queue.image_model_status()
        finally:queue.gpu.release()

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: UUID, session: str):
        return app.state.queue.cancel(session, str(job_id))

    @app.post('/api/jobs/{job_id}/export')
    def retry_export(job_id: UUID, session: str):
        return app.state.queue.retry_export(session,str(job_id))

    @app.get('/api/jobs/{job_id}/{name}')
    def artifact(job_id: UUID, name: str, session: str):
        return FileResponse(app.state.store.artifact(session, str(job_id), name))

    app.mount('/', StaticFiles(directory=Path(__file__).parent / 'web', html=True), name='web')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765, access_log=False)
