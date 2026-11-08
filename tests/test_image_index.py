import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.image_index import images
from backend.app.storage import SessionStore, encode, digest


class ImageIndexTests(unittest.TestCase):
    def test_newest_job_first_and_final_before_original_on_rebuild_cache_and_pages(self):
        with tempfile.TemporaryDirectory(dir='data') as temporary, SessionStore(temporary) as store:
            session=store.create_session('Order')
            expected=[]
            for day in (1, 2):
                job=str(uuid4());base=f'sessions/{session}/jobs/{job}'
                names=['original.png','detailed.png','image.png']
                store._replace(base+'/state.json',encode(dict(schema_version=1,id=job,status='SUCCEEDED',created_at=f'2026-11-0{day}T12:00:00Z',artifacts=names)))
                for name in names:store._replace(base+'/artifacts/'+name,b'fixture')
                expected=[(job,name) for name in ('image.png','original.png','detailed.png')]+expected
            for rebuild in (True, False):
                result=images(store,rebuild=rebuild)
                self.assertEqual([(r['job_id'],r['name']) for r in result['items']],expected)
            self.assertEqual(images(store,size=1,page=1)['items'][0]['name'],'image.png')
            self.assertEqual(images(store,size=1,page=2)['items'][0]['name'],'original.png')

    def test_replaced_source_with_preserved_size_and_mtime_invalidates_index_after_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            with SessionStore(temporary) as store:
                session=store.create_session('before');self.seed(store,session,1)
                self.assertEqual(images(store)['items'][0]['title'],'before')
                original=store.root/'sessions'/session/'session.json'
            stamp=original.stat()
            replacement=original.with_suffix('.replacement')
            replacement.write_bytes(original.read_bytes().replace(b'before',b'after!'))
            os.utime(replacement,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            os.replace(replacement,original)
            self.assertEqual(original.stat().st_size,stamp.st_size)
            self.assertEqual(original.stat().st_mtime_ns,stamp.st_mtime_ns)
            with SessionStore(temporary) as store:
                self.assertEqual(images(store)['items'][0]['title'],'after!')
                self.assertEqual(images(store,query='before')['total'],0)
                self.assertEqual(images(store,query='after!')['total'],1)

    def test_index_write_failure_returns_originals_and_recovers_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary,SessionStore(temporary) as store:
            session=store.create_session('preserved');self.seed(store,session,2)
            originals={p:digest(p.read_bytes()) for p in Path(temporary).glob('sessions/**/*') if p.is_file()}
            for failed in ('page-', 'head.json'):
                replace=store._replace
                def fail(path,content):
                    if failed in path:raise OSError('disk full')
                    return replace(path,content)
                with patch.object(store,'_replace',side_effect=fail):
                    result=images(store,rebuild=True)
                self.assertEqual(result['total'],2)
                self.assertTrue(any('disk full' in row['error'] for row in result['errors']))
                self.assertIn('image_index',store.read_errors)
                self.assertEqual(images(store)['errors'],[])
                self.assertNotIn('image_index',store.read_errors)
            self.assertEqual(originals,{p:digest(p.read_bytes()) for p in originals})

    def seed(self,store,session,count):
        for number in range(count):
            job=str(uuid4());base=f'sessions/{session}/jobs/{job}'
            store._replace(base+'/state.json',encode(dict(schema_version=1,id=job,status='SUCCEEDED',created_at=f'2026-11-06T00:00:{number%60:02d}Z',artifacts=['image.png'],plan={'model_path':'checkpoint/wai.safetensors'})))
            store._replace(base+'/artifacts/image.png',b'fixture image')

    def test_pages_rebuild_missing_corrupt_and_source_change(self):
        with tempfile.TemporaryDirectory() as temporary,SessionStore(temporary) as store:
            session=store.create_session('꽃밭');self.seed(store,session,201)
            original={p:digest(p.read_bytes()) for p in Path(temporary).glob('sessions/**/*') if p.is_file()}
            first=images(store,size=100)
            self.assertEqual(first['total'],201)
            self.assertEqual(len(images(store,page=3,size=100)['items']),1)
            head=store._read_json('indexes/images/head.json')
            self.assertEqual(len(head['pages']),2)
            for mode in ('missing','corrupt','head-empty','head-duplicate','legacy'):
                path=store._path(head['pages'][0]['path'])
                if mode=='missing':path.unlink()
                elif mode=='corrupt':path.write_text('broken',encoding='utf-8')
                else:
                    damaged=dict(head,pages=[] if mode=='head-empty' else head['pages']*2)
                    if mode=='legacy':damaged=dict(head);damaged.pop('sha256',None)
                    store._replace('indexes/images/head.json',encode(damaged))
                self.assertEqual(images(store,size=100),first)
            with patch.object(store,'artifact',side_effect=AssertionError('Valid cache must avoid full source reads')):
                self.assertEqual(images(store,size=100),first)
            self.assertEqual(images(store,query='꽃밭')['total'],201)
            self.assertEqual(images(store,query='NO MATCH')['total'],0)
            self.assertEqual(images(store,query='WAI')['total'],201)
            self.assertEqual(original,{p:digest(p.read_bytes()) for p in original})
            self.seed(store,session,1)
            self.assertEqual(images(store)['total'],202)
            bad=next(store._path(f'sessions/{session}/jobs').glob('*/state.json'))
            bad.write_text('{broken',encoding='utf-8')
            result=images(store)
            self.assertEqual(result['total'],201);self.assertEqual(len(result['errors']),1)

    def test_api_bounds_and_empty_index(self):
        with tempfile.TemporaryDirectory() as temporary,TestClient(create_app(temporary),base_url='http://127.0.0.1:8765') as client:
            self.assertEqual(client.get('/api/images').json()['total'],0)
            for query in ('page=0','size=101','q='+'a'*201):
                self.assertEqual(client.get('/api/images?'+query).status_code,422)
            self.assertEqual(client.post('/api/images/rebuild',json={}).status_code,200)


    def test_thumbnail_cache_rebuild_original_preservation_and_reference_guards(self):
        import io
        from PIL import Image
        from unittest.mock import patch
        from backend.app.image_index import thumbnail
        with tempfile.TemporaryDirectory() as temporary, TestClient(create_app(temporary),base_url='http://127.0.0.1:8765') as client:
            store=client.app.state.store;session=store.create_session('preview');other=store.create_session('other')
            job=str(uuid4());base=f'sessions/{session}/jobs/{job}'
            buffer=io.BytesIO();Image.new('RGBA',(1200,600),(200,30,50,0)).save(buffer,format='PNG');source=buffer.getvalue()
            state=dict(schema_version=1,id=job,status='SUCCEEDED',artifacts=['image.png'],artifact_hashes={'image.png':digest(source)})
            store._replace(base+'/state.json',encode(state));store._replace(base+'/artifacts/image.png',source)
            url=f'/api/images/{job}/thumbnail?session={session}'
            response=client.get(url);self.assertEqual(response.status_code,200);self.assertEqual(response.headers['content-type'],'image/jpeg')
            with Image.open(io.BytesIO(response.content)) as image:
                self.assertEqual(image.size,(384,192));self.assertEqual(image.getpixel((0,0)),(255,255,255))
            with patch('PIL.Image.open',side_effect=AssertionError('Cache must avoid decoding')):
                self.assertEqual(thumbnail(store,session,job,'image.png'),response.content)
            cache=next(Path(temporary).glob('indexes/thumbnails/**/*.jpg'));cache.write_bytes(b'broken')
            self.assertEqual(client.get(url).content,response.content)
            cache.unlink();self.assertEqual(client.get(url).content,response.content)
            for failed in ('.jpg','.json'):
                cache.unlink(missing_ok=True)
                replace=store._replace
                def fail(path,content):
                    if path.startswith('indexes/thumbnails/') and path.endswith(failed):raise OSError('disk full')
                    return replace(path,content)
                with patch.object(store,'_replace',side_effect=fail):
                    self.assertEqual(client.get(url).content,response.content)
                self.assertTrue(any('disk full' in row['error'] for row in client.get('/api/storage-errors').json()))
                self.assertEqual(client.get(url).content,response.content)
                self.assertFalse(any('disk full' in row['error'] for row in client.get('/api/storage-errors').json()))
            self.assertEqual(store._path(base+'/artifacts/image.png').read_bytes(),source)
            self.assertEqual(client.get(url+'&name=report.json').status_code,422)
            self.assertNotEqual(client.get(f'/api/images/{job}/thumbnail?session={other}').status_code,200)
            store._path(base+'/artifacts/image.png').write_bytes(b'tampered')
            self.assertNotEqual(client.get(url).status_code,200)
