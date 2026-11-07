import io
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from PIL import Image
from backend.app.storage import SessionStore,encode
from backend.app.masks import save_mask,read_mask


def png(size,value):
    output=io.BytesIO();Image.new('L',size,value).save(output,format='PNG');return output.getvalue()


class MaskTests(unittest.TestCase):
    def test_source_identity_dimensions_empty_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary,SessionStore(temporary) as store:
            session=store.create_session('mask');job=str(uuid4());base=f'sessions/{session}/jobs/{job}'
            store._replace(base+'/state.json',encode(dict(schema_version=1,id=job,status='SUCCEEDED')))
            original=png((64,96),100);store._replace(base+'/artifacts/image.png',original)
            for content in (b'broken',png((32,32),255),png((64,96),0)):
                with self.assertRaises(ValueError):save_mask(store,session,job,content)
            meta=save_mask(store,session,job,png((64,96),255));path,loaded=read_mask(store,session,meta['id'],job)
            self.assertEqual(meta,loaded)
            with self.assertRaises(ValueError):read_mask(store,session,meta['id'],str(uuid4()))
            other=store.create_session('other')
            with self.assertRaises(FileNotFoundError):read_mask(store,other,meta['id'])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):read_mask(store,session,meta['id'])
            self.assertEqual(store._path(base+'/artifacts/image.png').read_bytes(),original)
