import json
import struct
import tempfile
import unittest
from pathlib import Path
from scripts.inspect_models import inspect, ROOT


class HeaderTests(unittest.TestCase):
    def test_runtime_inventory_excludes_sources_and_keeps_rejections_unselectable(self):
        from unittest.mock import patch
        from backend.app.models import scan
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('sources/revision/model.safetensors','checkpoint/model.safetensors','clip_vision/vision.safetensors','text_encoder/rejected.safetensors'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            rejected=root/'text_encoder/rejected.safetensors'
            rejected.with_name(rejected.name+'.rejected.json').write_text('{}')
            def inspect_file(path,**kwargs):
                self.assertNotIn('sources',path.parts)
                return {'family':'vision_encoder' if path.parent.name=='clip_vision' else 'sdxl_candidate'}
            with patch('backend.app.models.MODEL_ROOT',root),patch('backend.app.models.inspect',side_effect=inspect_file):
                rows=scan()
            self.assertEqual(len(rows),3)
            by_path={row['path']:row for row in rows}
            self.assertTrue(by_path['checkpoint/model.safetensors']['available'])
            self.assertIn('이미지 특징',by_path['clip_vision/vision.safetensors']['reason'])
            row=by_path['text_encoder/rejected.safetensors']
            self.assertFalse(row['available']);self.assertNotIn('component',row);self.assertIn('검증 거절',row['reason'])
            self.assertTrue((root/'sources/revision/model.safetensors').exists())

    def test_krea_without_filename_or_metadata_is_identified_by_shapes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path=Path(folder)/'renamed.safetensors'
            shapes={'first.weight':[6144,64],'txtfusion.projector.weight':[1,12],
                    'blocks.0.attn.wk.weight':[1536,6144],'blocks.27.attn.wq.weight':[6144,6144]}
            def write():
                header=json.dumps({k:dict(shape=v,dtype='F8_E4M3',data_offsets=[0,0]) for k,v in shapes.items()}).encode()
                path.write_bytes(struct.pack('<Q',len(header))+header)
            write();self.assertEqual(inspect(path)['family'],'krea2_candidate')
            shapes['first.weight']=[6144,16];write();self.assertEqual(inspect(path)['family'],'unresolved')

    def test_real_headers(self):
        for path in (ROOT / 'models').rglob('*.safetensors'):
            self.assertGreater(inspect(path)['tensor_count'], 0)

    def test_truncated_and_invalid_offsets(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'broken.safetensors'
            path.write_bytes(b'123')
            with self.assertRaises(ValueError):
                inspect(path)
            header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header)
            with self.assertRaises(ValueError):
                inspect(path)


if __name__ == '__main__':
    unittest.main()
