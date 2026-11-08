"""Chat Completions transport, normalized to the existing chat event contract."""
import ipaddress
import json
from urllib.parse import urlsplit

import httpx

from .ollama import OllamaError


def local_server(url):
    host = urlsplit(url).hostname
    if host in ('localhost', 'host.docker.internal'):
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or address.is_private
    except ValueError:
        return False


class OpenAICompatible:
    def __init__(self, base_url, api_key='', *, vision_models=(), response_format='json_object', token_field='max_tokens'):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.vision_models = vision_models
        self.response_format = response_format
        self.token_field = token_field
        self.uses_local_gpu = local_server(base_url)

    def headers(self):
        return {'Authorization': 'Bearer ' + self.api_key} if self.api_key else {}

    @staticmethod
    def read_json(response):
        response.raise_for_status()
        raw = bytearray()
        for part in response.iter_bytes():
            raw.extend(part)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError('Response too large')
        return json.loads(raw)

    def models(self):
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
                with client.stream('GET', self.base_url + '/models', headers=self.headers()) as response:
                    data = self.read_json(response)
                return [dict(name=m['id'], digest=None, size=0) for m in data['data'] if isinstance(m.get('id'), str)]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise self.error(exc) from None

    def local_model(self, name):
        return dict(name=name, digest=None), dict(capabilities=['completion'] + (['vision'] if name in self.vision_models else []))

    def image_ready(self, selected_model=None):
        pass

    def release(self, name):
        # shortcut: Chat Completions has no unload API; local servers own model residency.
        pass

    def body(self, request):
        messages = []
        for source in request['messages']:
            message = dict(role=source['role'], content=source['content'])
            if source.get('images'):
                message['content'] = [dict(type='text', text=source['content'])] + [
                    dict(type='image_url', image_url={'url': 'data:image/png;base64,' + image}) for image in source['images']]
            messages.append(message)
        if self.response_format != 'json_schema':
            messages[0]['content'] += ' Output JSON schema: ' + json.dumps(request['format'], ensure_ascii=False)
        body = dict(model=request['model'], messages=messages, stream=request.get('stream', False))
        if self.response_format == 'json_schema':
            body['response_format'] = dict(type='json_schema', json_schema=dict(name='image_studio_suggestion', schema=request['format'], strict=False))
        elif self.response_format == 'json_object':
            body['response_format'] = dict(type='json_object')
        for key in ('temperature', 'top_p', 'seed', 'presence_penalty', 'frequency_penalty', 'stop'):
            if key in request['options']:
                body[key] = request['options'][key]
        limit = request['options'].get('num_predict')
        if limit is not None and limit > 0:
            body[self.token_field] = limit
        return body

    def error(self, exc):
        if isinstance(exc, httpx.HTTPStatusError):
            return OllamaError(f'OpenAI 호환 서버 응답 오류 (HTTP {exc.response.status_code}). 연결 주소·API 키·모델·지원 옵션을 확인하세요.')
        return OllamaError('OpenAI 호환 서버 연결 또는 응답 형식 오류입니다. 서버 주소와 연결 상태를 확인하세요.')

    @staticmethod
    def chunk(data, *, streaming):
        if not isinstance(data, dict) or data.get('error'):
            raise ValueError('Invalid response')
        choices = data.get('choices')
        if choices == [] and streaming:
            return None
        choice = choices[0]
        message = choice.get('delta' if streaming else 'message', {})
        content = message.get('content') or ''
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError('Invalid content')
        reason = choice.get('finish_reason')
        if reason not in (None, 'stop', 'length') or message.get('refusal'):
            raise ValueError('Unsupported completion')
        return dict(message=dict(content=content, thinking=reasoning), done=reason is not None, done_reason=reason)

    def request(self, path, body, timeout=180):
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout) as client:
                with client.stream('POST', self.base_url + '/chat/completions', headers=self.headers(), json=self.body(body)) as response:
                    return self.chunk(self.read_json(response), streaming=False)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise self.error(exc) from None

    async def stream_chat(self, body):
        try:
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=180) as client:
                async with client.stream('POST', self.base_url + '/chat/completions', headers=self.headers(), json=self.body(body)) as response:
                    response.raise_for_status()
                    buffer = ''; event = []; total = 0
                    async for part in response.aiter_text():
                        total += len(part.encode('utf-8'))
                        if total > 2 * 1024 * 1024:
                            raise ValueError('Response too large')
                        buffer += part
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.rstrip('\r')
                            if line.startswith('data:'):
                                event.append(line[5:].lstrip(' '))
                            elif not line and event:
                                value = '\n'.join(event); event = []
                                if value == '[DONE]':
                                    return
                                chunk = self.chunk(json.loads(value), streaming=True)
                                if chunk is not None:
                                    yield chunk
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise self.error(exc) from None
