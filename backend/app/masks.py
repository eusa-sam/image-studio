"""Immutable masks tied to a workspace image; no arbitrary file references."""
import io
from datetime import datetime,timezone
from uuid import uuid4
from PIL import Image
from .storage import digest,encode,identifier


def save_mask(store,session,source_job,content,source_artifact='image.png'):
    if source_artifact not in ('image.png','original.png'):raise ValueError('Invalid mask source artifact')
    if len(content)>6*1024*1024:raise ValueError('Mask exceeds 6 MiB')
    with store._mutex:
        store._ready()
        source=store.artifact(session,source_job,source_artifact)
        with Image.open(source) as image:source_size=image.size
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format!='PNG' or image.size!=source_size or max(image.size)>4096:raise ValueError('Mask must be PNG with the source image dimensions')
                mask=image.convert('L').point(lambda value:255 if value>=128 else 0)
        except OSError as exc:raise ValueError('Invalid mask PNG') from exc
        if mask.getbbox() is None:raise ValueError('수정할 영역을 칠하세요.')
        buffer=io.BytesIO();mask.save(buffer,format='PNG');data=buffer.getvalue()
        mask_id=str(uuid4());base=store._session_path(session)+'/masks/'+mask_id
        meta=dict(schema_version=1,id=mask_id,created_at=datetime.now(timezone.utc).isoformat(),source_job=source_job,source_artifact=source_artifact,source_sha256=digest(source.read_bytes()),sha256=digest(data),size=list(source_size))
        store._commit({base+'/mask.png':data,base+'/meta.json':encode(meta)})
        return meta


def read_mask(store,session,mask_id,source_job=None,source_artifact=None):
    with store._mutex:
        store._ready()
        base=store._session_path(session)+'/masks/'+identifier(mask_id)
        meta=store._read_json(base+'/meta.json')
        if meta['id']!=mask_id or (source_job is not None and meta['source_job']!=source_job):raise ValueError('Mask source mismatch')
        artifact=meta.get('source_artifact','image.png')
        if artifact not in ('image.png','original.png') or (source_artifact is not None and artifact!=source_artifact):raise ValueError('Mask source mismatch')
        path=store._path(base+'/mask.png')
        if digest(path.read_bytes())!=meta['sha256']:raise ValueError('Mask hash mismatch')
        source=store.artifact(session,meta['source_job'],artifact)
        if digest(source.read_bytes())!=meta['source_sha256']:raise ValueError('Mask source changed')
        return path,meta
