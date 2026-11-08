"""Loopback-only Ollama access. Never pull models or follow redirects."""
import json
import time
import urllib.error
import urllib.request


def local_listener(port):
    """Avoid Windows' delayed refused connection; unknown state keeps normal HTTP checks."""
    import os
    if os.name!='nt':return None
    import ctypes,struct
    try:
        query=ctypes.WinDLL('iphlpapi.dll',winmode=0x00000800).GetExtendedTcpTable
        query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong),ctypes.c_int,ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        query.restype=ctypes.c_ulong
        # OWNER_PID_LISTENER rows: IPv4 and IPv6 (including dual-stack listeners).
        for family,row_size,port_offset in ((2,24,8),(23,56,20)):
            size=ctypes.c_ulong()
            if query(None,ctypes.byref(size),False,family,3,0)!=122 or not 4<=size.value<=16*1024**2:return None
            buffer=ctypes.create_string_buffer(size.value)
            if query(buffer,ctypes.byref(size),False,family,3,0)!=0:return None
            raw=buffer.raw;count=struct.unpack_from('<I',raw)[0]
            if 4+count*row_size>len(raw):return None
            if any(struct.unpack_from('!H',raw,4+i*row_size+port_offset)[0]==port for i in range(count)):return True
        return False
    except OSError:return None


class OllamaError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Ollama:
    def __init__(self,base_url='http://127.0.0.1:11434',endpoint='/api'):
        self.base_url=base_url
        self.endpoint=endpoint

    async def stream_chat(self, body):
        import httpx
        try:
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=180) as client:
                async with client.stream('POST', self.base_url+self.endpoint+'/chat', json=body) as response:
                    response.raise_for_status()
                    placement_checked = body.get('options',{}).get('num_gpu') == 0
                    async for line in response.aiter_lines():
                        if len(line.encode('utf-8')) > 2 * 1024 * 1024:
                            raise OllamaError('Ollama 응답이 너무 큽니다.')
                        if not line:
                            continue
                        result = json.loads(line)
                        if not isinstance(result, dict) or result.get('error'):
                            raise OllamaError('Ollama 스트림 응답 오류')
                        if not placement_checked and not result.get('done'):
                            placement_checked = True
                            try:
                                placement = await client.get(self.base_url+self.endpoint+'/ps',timeout=2)
                                placement.raise_for_status()
                                row = next((m for m in placement.json().get('models',[]) if m.get('name',m.get('model')) == body['model']),{})
                                if type(row.get('size_vram')) is int:
                                    result['_gpu_bytes'] = row['size_vram']
                            except (httpx.HTTPError,ValueError,TypeError,AttributeError):
                                pass  # Placement diagnostics must not interrupt a valid response.
                        yield result
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError('로컬 Ollama 스트림 실패: ' + str(exc)) from exc

    def request(self, path, body=None, timeout=10):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(self.base_url+self.endpoint+'/' + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        try:
            from urllib.parse import urlsplit
            url=urlsplit(self.base_url)
            if url.hostname in ('127.0.0.1','localhost','::1') and local_listener(url.port or (443 if url.scheme=='https' else 80)) is False:
                raise urllib.error.URLError(ConnectionRefusedError(f'Ollama is not listening on local port {url.port or (443 if url.scheme=="https" else 80)}'))
            with opener.open(request, timeout=timeout) as response:
                content = response.read(2 * 1024 * 1024 + 1)
                if len(content) > 2 * 1024 * 1024:
                    raise OllamaError('Ollama 응답이 너무 큽니다.')
                result = json.loads(content)
                if not isinstance(result, dict) or result.get('error'):
                    raise OllamaError('Ollama 응답 오류: ' + str(result.get('error') if isinstance(result, dict) else '잘못된 형식'))
                return result
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise OllamaError('로컬 Ollama 요청 실패: ' + str(exc)) from exc

    def models(self):
        return [dict(name=m['name'], size=m.get('size', 0), digest=m.get('digest'))
                for m in self.request('tags').get('models', [])
                if m.get('details', {}).get('format') == 'gguf'
                and not m.get('remote_host') and not m.get('remote_model')
                and 'cloud' not in m['name'].lower()
                and ('completion' in m.get('capabilities', ['completion']))]

    def local_model(self, name):
        record = next((m for m in self.models() if m['name'] == name), None)
        if not record:
            raise ValueError('설치된 로컬 대화 모델을 선택하세요.')
        info = self.request('show', {'model': name})
        if info.get('remote_host') or info.get('remote_model') or 'completion' not in info.get('capabilities', []):
            raise ValueError('원격 모델 또는 대화 미지원 모델은 사용할 수 없습니다.')
        if 'thinking' in info.get('capabilities',[]) and not info.get('thinking',{}).get('values'):
            info = dict(info, thinking=dict(info.get('thinking',{}),values=[True,False]))
        return record, info

    def running(self):
        return self.request('ps').get('models', [])

    def image_ready(self, selected_model=None):
        try:
            models = self.running()
        except OllamaError as exc:
            # An absent Ollama service must not prevent manual image generation.
            reason = getattr(exc.__cause__, 'reason', None)
            if isinstance(reason, ConnectionRefusedError):
                return
            raise
        if any(m.get('size_vram', 0) > 0 and m.get('name', m.get('model')) != selected_model for m in models):
            raise OllamaError('다른 Ollama 모델이 GPU를 사용 중입니다. 해당 작업을 종료한 뒤 다시 생성하세요.')

    def release(self, name):
        self.request('generate', {'model': name, 'keep_alive': 0}, timeout=30)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not any(m.get('name', m.get('model')) == name for m in self.running()):
                return
            time.sleep(0.2)
        raise OllamaError('Ollama 모델 메모리 해제를 확인하지 못했습니다.')
