"""Local application preferences; workspace records remain independent."""
import ipaddress
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from .storage import encode, digest


class AppSettings(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    schema_version: int=Field(default=1,ge=1,le=1)
    revision: int=Field(default=0,ge=0)
    model_path: str
    output_path: str
    ollama_url: str='http://127.0.0.1:11434'
    ollama_endpoint: str='/api'
    llm_provider: Literal['ollama','openai_compatible']='ollama'
    openai_base_url: str='http://127.0.0.1:1234/v1'
    openai_response_format: Literal['json_object','json_schema','text']='json_object'
    openai_token_field: Literal['max_tokens','max_completion_tokens']='max_tokens'
    openai_vision_models: list[str]=Field(default_factory=list,max_length=200)
    openai_key_id: str | None=Field(default=None,pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
    openai_api_key: SecretStr | None=Field(default=None,exclude=True)
    openai_key_registered: bool=Field(default=False,exclude=True)

    @field_validator('openai_base_url')
    @classmethod
    def compatible_url(cls,value):
        from .openai_compatible import local_server
        value=value.strip().rstrip('/')
        url=urlsplit(value)
        if (url.scheme not in ('http','https') or not url.hostname or url.username or url.password or
            url.query or url.fragment or any(c.isspace() for c in value) or
            not re.fullmatch(r'(?:/[A-Za-z0-9_.~-]+)*',url.path)):
            raise ValueError('Base URL은 http(s)://호스트/v1 같은 API 기본 주소입니다.')
        _=url.port
        if url.scheme!='https' and not local_server(value):
            raise ValueError('외부 API에는 HTTPS 주소를 사용하세요.')
        return value

    @field_validator('openai_api_key')
    @classmethod
    def valid_key(cls,value):
        if value is not None and (len(value.get_secret_value())>4096 or any(ord(c)<33 or ord(c)>126 for c in value.get_secret_value())):
            raise ValueError('API 키 형식을 확인하세요.')
        return value

    @field_validator('model_path','output_path')
    @classmethod
    def absolute_path(cls,value):
        path=Path(value).expanduser()
        if not path.is_absolute():raise ValueError('폴더는 전체 경로를 입력하세요.')
        return str(path.resolve())

    @field_validator('ollama_url')
    @classmethod
    def local_url(cls,value):
        url=urlsplit(value.strip())
        if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('','/'):
            raise ValueError('Ollama 서버 주소는 http(s)://호스트:포트 형식입니다. API 경로는 별도로 입력하세요.')
        try:
            address=ipaddress.ip_address(url.hostname)
            local=address.is_loopback or (address.is_private and not address.is_unspecified and not address.is_link_local and not address.is_multicast)
        except ValueError:local=url.hostname=='localhost' or (os.environ.get('IMAGE_STUDIO_CONTAINER')=='1' and url.hostname=='host.docker.internal')
        if not local:raise ValueError('로컬 추론 서버의 localhost 또는 사설 IP 주소를 입력하세요.')
        _=url.port
        return value.strip().rstrip('/')

    @field_validator('ollama_endpoint')
    @classmethod
    def api_prefix(cls,value):
        value=value.rstrip('/')
        if not re.fullmatch(r'(?:/[A-Za-z0-9_-]+)*',value):raise ValueError('API 경로는 /api 같은 경로 접두사입니다.')
        return value


def settings_file():
    return 'settings/container.json' if os.environ.get('IMAGE_STUDIO_CONTAINER')=='1' else 'settings/application.json'


def read_settings(store):
    from .models import ROOT,MODEL_ROOT
    path=store._path(settings_file())
    if path.exists():return AppSettings.model_validate_json(path.read_bytes())
    output=ROOT/'outputs' if store.root==ROOT/'data' else store.root/'outputs'
    url='http://host.docker.internal:11434' if os.environ.get('IMAGE_STUDIO_CONTAINER')=='1' else 'http://127.0.0.1:11434'
    return AppSettings(model_path=str(MODEL_ROOT.resolve()),output_path=str(output.resolve()),ollama_url=url)


def credentials_file(key_id):
    from .storage import identifier
    return 'credentials/'+identifier(key_id)+'.json'


def api_key(store,settings):
    if settings.openai_api_key is not None:return settings.openai_api_key.get_secret_value()
    if not settings.openai_key_id:return ''
    path=store._path(credentials_file(settings.openai_key_id))
    if not path.exists():return ''
    saved=json.loads(path.read_bytes())
    return saved['api_key'] if saved['base_url']==settings.openai_base_url else ''


def public_settings(store,settings):
    return settings.model_dump() | {'openai_key_registered':bool(api_key(store,settings))}


def compatible_client(store,settings):
    from .openai_compatible import OpenAICompatible
    return OpenAICompatible(settings.openai_base_url,api_key(store,settings),
        vision_models=settings.openai_vision_models,response_format=settings.openai_response_format,token_field=settings.openai_token_field)


def validate_directories(settings,store):
    model=Path(settings.model_path);output=Path(settings.output_path)
    if not model.is_dir():raise ValueError('모델 폴더가 없습니다. 모델과 configs가 있는 폴더를 선택하세요.')
    if output==model or output.is_relative_to(model):raise ValueError('이미지 출력 폴더를 모델 폴더 밖으로 지정하세요.')
    if output==store.root or output.is_relative_to(store.root/'sessions'):
        raise ValueError('Workspace 기록 폴더를 이미지 출력 폴더로 사용할 수 없습니다.')
    output.mkdir(parents=True,exist_ok=True)
    probe=output/('.write-test-'+str(uuid4()))
    try:
        with probe.open('xb') as stream:stream.write(b'');stream.flush();os.fsync(stream.fileno())
    finally:probe.unlink(missing_ok=True)


def apply_settings(settings,queue,chat):
    from backend.app import models; from backend.app.inference.conditioning import prompt_tokens
    models.MODEL_ROOT=Path(settings.model_path)
    prompt_tokens.MODEL_ROOT=models.MODEL_ROOT
    prompt_tokens.tokenizers.cache_clear()
    queue.output_path=Path(settings.output_path)
    chat.ollama.base_url=settings.ollama_url
    chat.ollama.endpoint=settings.ollama_endpoint
    chat.compatible=compatible_client(chat.store,settings) if settings.llm_provider=='openai_compatible' else None
    queue.ollama=chat.client


def export_images(store,session,job_id,output_root,*,recover_existing=False):
    """Publish a complete new result folder; never overwrite existing exports."""
    job=store.job(session,job_id)
    parent=Path(output_root)/session
    parent.mkdir(parents=True,exist_ok=True)
    destination=parent/job_id
    files={'positive.txt':job['positive'].encode('utf-8'),'negative.txt':job['negative'].encode('utf-8'),
           'settings.json':encode(job['settings'])}
    if destination.exists():
        if recover_existing:
            expected=dict(job['state']['artifact_hashes']) | {name:digest(content) for name,content in files.items()}
            manifest=json.loads((destination/'manifest.json').read_bytes())
            if manifest==dict(schema_version=1,session=session,job_id=job_id,hashes=expected):
                for name,checksum in expected.items():
                    with (destination/name).open('rb') as stream:
                        if hashlib.file_digest(stream,'sha256').hexdigest()!=checksum:
                            raise ValueError('기존 출력 파일이 변경됐습니다. 덮어쓰지 않았습니다: '+name)
                return str(destination)
        raise ValueError('출력 폴더가 이미 존재합니다. 기존 파일은 덮어쓰지 않았습니다.')
    temporary=parent/('.'+job_id+'-'+str(uuid4())+'.partial')
    temporary.mkdir()
    hashes={}
    for name,expected in job['state']['artifact_hashes'].items():
        checksum=hashlib.sha256()
        with store.artifact(session,job_id,name).open('rb') as source,(temporary/name).open('xb') as target:
            while chunk:=source.read(1024*1024):
                target.write(chunk);checksum.update(chunk)
            target.flush();os.fsync(target.fileno())
        if checksum.hexdigest()!=expected:raise ValueError('Artifact changed while exporting')
        hashes[name]=expected
    for name,content in files.items():
        with (temporary/name).open('xb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
        hashes[name]=digest(content)
    with (temporary/'manifest.json').open('xb') as stream:
        stream.write(encode(dict(schema_version=1,session=session,job_id=job_id,hashes=hashes)))
        stream.flush();os.fsync(stream.fileno())
    temporary.rename(destination)
    return str(destination)
