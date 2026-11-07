"""User-uploaded reference images, separate from generated job artifacts."""
import io
from datetime import datetime,timezone
from uuid import uuid4
from PIL import Image,ImageOps
from .storage import digest,encode,identifier


def save_reference(store,session,filename,content):
    if not content or len(content)>12*1024*1024:raise ValueError('참고 이미지는 최대12MiB입니다.')
    try:
        with Image.open(io.BytesIO(content)) as image:
            extension={'PNG':'png','JPEG':'jpg','WEBP':'webp'}.get(image.format)
            if not extension or getattr(image,'n_frames',1)!=1:raise ValueError('정지 PNG/JPEG/WebP 이미지를 선택하세요.')
            if max(image.size)>8192 or image.width*image.height>32_000_000:raise ValueError('이미지 크기는 최대8192px/3200만 화소입니다.')
            size=list(image.size)
            preview=ImageOps.exif_transpose(image).convert('RGB');preview.thumbnail((768,768))
            buffer=io.BytesIO();preview.save(buffer,format='PNG');pixels=buffer.getvalue()
    except (OSError,Image.DecompressionBombError) as exc:raise ValueError('이미지 파일을 읽을 수 없습니다.') from exc
    with store._mutex:
        store._ready();base=store._session_path(session)
        store._read_json(base+'/session.json')
        reference=str(uuid4());base+='/references/'+reference
        meta=dict(schema_version=1,id=reference,kind='user_reference',created_at=datetime.now(timezone.utc).isoformat(),
                  filename=filename,original='original.'+extension,size=size,sha256=digest(content),preview_sha256=digest(pixels))
        store._commit({base+'/'+meta['original']:content,base+'/preview.png':pixels,base+'/meta.json':encode(meta)})
        return meta


def read_reference(store,session,reference):
    with store._mutex:
        store._ready();base=store._session_path(session)+'/references/'+identifier(reference)
        meta=store._read_json(base+'/meta.json')
        if meta['id']!=reference or meta['kind']!='user_reference' or meta['original'] not in ('original.png','original.jpg','original.webp'):
            raise ValueError('참고 이미지 기록 불일치')
        original=store._path(base+'/'+meta['original']);preview=store._path(base+'/preview.png')
        if digest(original.read_bytes())!=meta['sha256'] or digest(preview.read_bytes())!=meta['preview_sha256']:
            raise ValueError('참고 이미지 해시 불일치')
        return preview,meta
