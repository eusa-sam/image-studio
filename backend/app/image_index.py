"""Rebuildable, month-sharded image summaries; job files remain authoritative."""
from collections import defaultdict
import json
import os

from .storage import digest, encode, identifier


def images(store, *, query='', page=1, size=24, rebuild=False):
    if page < 1 or not 1 <= size <= 100:raise ValueError('Invalid page')
    with store._mutex:
        store._ready()
        sources=[]
        for pattern in ('*/*/session.json','*/*/jobs/*/state.json','*/*/jobs/*/artifacts/*.png'):
            for path in sorted(store._path('sessions').glob(pattern)):
                relative=os.path.relpath(path,store.root).replace(os.sep,'/')
                try:
                    stat=store._path(relative).stat()
                    sources.append((relative,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino))
                except (ValueError,OSError) as exc:sources.append((relative,str(exc)))
        signature=digest(encode(sources))
        rows=[];errors=[]
        try:
            if rebuild:raise ValueError('Explicit rebuild')
            head=store._read_json('indexes/images/head.json')
            checksum=head.pop('sha256')
            if digest(encode(head))!=checksum:raise ValueError('Corrupt image index header')
            if head['signature']!=signature:raise ValueError('Stale index')
            for part in head['pages']:
                if not part['path'].startswith('indexes/images/'):raise ValueError('Invalid index page')
                content=store._path(part['path']).read_bytes()
                if digest(content)!=part['sha256']:raise ValueError('Corrupt index page')
                rows.extend(json.loads(content)['items'])
            errors=head['errors']
            store.read_errors.pop('image_index',None)
        except (ValueError,OSError,KeyError,TypeError):
            rows=[];errors=[];titles={}
            for source in sources:
                relative=source[0]
                if not relative.endswith('/session.json'):continue
                session='/'.join(relative.split('/')[1:3])
                try:
                    store._session_path(session)
                    record=store._read_json(relative)
                    if record['id']!=session.split('/')[1]:raise ValueError('Session identity mismatch')
                    titles[session]=record['title']
                except (ValueError,OSError,KeyError,TypeError) as exc:errors.append(dict(path=relative,error=str(exc)))
            for source in sources:
                relative=source[0]
                if not relative.endswith('/state.json'):continue
                parts=relative.split('/');session='/'.join(parts[1:3])
                try:
                    if session not in titles:continue
                    job=identifier(parts[4]);state=store._read_json(relative)
                    if state['id']!=job:raise ValueError('Job identity mismatch')
                    if state['status'] not in ('SUCCEEDED','PARTIAL_FAILED','CANCELLED','INTERRUPTED'):continue
                    for name in ('image.png','original.png','detailed.png'):
                        if name not in state.get('artifacts',[]):continue
                        store.artifact(session,job,name)
                        rows.append(dict(session=session,title=titles[session],job_id=job,name=name,status=state['status'],
                                         created_at=state['created_at'],model=state.get('plan',{}).get('model_path','')))
                except (ValueError,OSError,KeyError,TypeError) as exc:errors.append(dict(path=relative,error=str(exc)))
            rows.sort(key=lambda row:(row['created_at'],row['job_id'],row['name']),reverse=True)
            months=defaultdict(list)
            for row in rows:months[row['session'].split('/')[0]].append(row)
            try:
                pages=[]
                for month,items in sorted(months.items()):
                    for offset in range(0,len(items),200):
                        path=f'indexes/images/{month}/page-{offset//200+1:06d}.json'
                        content=encode(dict(schema_version=1,items=items[offset:offset+200]))
                        store._replace(path,content)
                        pages.append(dict(path=path,sha256=digest(content)))
                head=dict(schema_version=1,signature=signature,pages=pages,errors=errors)
                head['sha256']=digest(encode(head))
                store._replace('indexes/images/head.json',encode(head))
                store.read_errors.pop('image_index',None)
            except (OSError,ValueError) as error:
                diagnostic=dict(path='indexes/images',error=str(error))
                store.read_errors['image_index']=[diagnostic]
                errors=[*errors,diagnostic]
        rows.sort(key=lambda row:(row['created_at'],row['job_id'],row['name']),reverse=True)
        term=query.strip().casefold()
        if term:rows=[row for row in rows if term in ' '.join(str(value) for value in row.values()).casefold()]
        return dict(items=rows[(page-1)*size:page*size],total=len(rows),page=page,size=size,errors=errors)


def thumbnail(store, session, job, name):
    """Disposable previews keyed by verified source bytes; originals remain untouched."""
    import io
    from PIL import Image, ImageOps
    if name not in ('image.png', 'original.png', 'detailed.png'):
        raise ValueError('Unsupported preview image')
    with store._mutex:
        source = store.artifact(session, job, name).read_bytes()
        key = digest(source)
        base = f'indexes/thumbnails/v1/{key[:2]}/{key}'
        try:
            meta = store._read_json(base + '.json')
            path = store._path(base + '.jpg')
            if path.stat().st_size > 1024*1024:raise ValueError('Oversized preview')
            content = path.read_bytes()
            if digest(content) != meta['sha256']:raise ValueError('Corrupt preview')
            store.read_errors.pop(base,None)
            return content
        except (OSError, ValueError, KeyError, TypeError):
            pass
        with Image.open(io.BytesIO(source)) as original:
            if original.width*original.height > 32_000_000:
                raise ValueError('Preview source exceeds 32 megapixels')
            preview = ImageOps.exif_transpose(original).convert('RGBA')
            preview.thumbnail((384,384), Image.Resampling.LANCZOS)
            background = Image.new('RGB', preview.size, 'white')
            background.paste(preview, mask=preview.getchannel('A'))
            buffer = io.BytesIO()
            background.save(buffer, format='JPEG', quality=82, optimize=True)
        content = buffer.getvalue()
        try:
            store._replace(base + '.jpg', content)
            store._replace(base + '.json', encode(dict(schema_version=1, sha256=digest(content))))
            store.read_errors.pop(base,None)
        except (OSError,ValueError) as error:
            store.read_errors[base]=[dict(path=base,error=str(error))]
        return content
