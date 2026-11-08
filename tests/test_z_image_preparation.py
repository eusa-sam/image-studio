import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_z_image_bf16 import prepare,convert
from backend.app.inference.models import z_image


class BF16PreparationTests(unittest.TestCase):
    def test_conversion_resume_and_modified_output_rejected(self):
        import torch
        from safetensors.torch import save_file,load_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();(source/'transformer').mkdir()
            name='transformer/weights.safetensors';original=source/name
            weights=torch.tensor([1.001,-2.003,0.0],dtype=torch.float32)
            save_file({'weight':weights},str(original),metadata={'format':'pt'})
            candidate=root/'expected';convert(original,candidate)
            sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
            spec=dict(repository='test',revision='pinned',variant='bf16-local-v1',files=[dict(path=name,size=candidate.stat().st_size,sha256=sha)])
            target=root/'prepared';before=original.read_bytes()
            with patch('backend.app.inference.models.z_image.verify_bundle') as verify_source:
                prepare(source,target,spec);verify_source.assert_called_once_with(source.resolve())
            self.assertTrue(torch.equal(load_file(str(target/name))['weight'],weights.bfloat16()))
            with patch('backend.app.inference.models.z_image.verify_bundle',side_effect=AssertionError('No source read on complete repeat')):
                prepare(source,target,spec)
            self.assertEqual(original.read_bytes(),before)
            manifest=json.loads((target/'download_manifest.json').read_text())
            with patch.object(z_image,'BF16_SPEC',spec):
                z_image.verify_bundle(target)
                manifest['variant']='foreign';(target/'download_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'variant'):z_image.verify_bundle(target,False)
                manifest['variant']=spec['variant'];(target/'download_manifest.json').write_text(json.dumps(manifest))
                data=bytearray((target/name).read_bytes());data[-1]^=1;(target/name).write_bytes(data)
                with self.assertRaisesRegex(ValueError,'hash mismatch'):z_image.verify_bundle(target)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):prepare(source,target,spec)
            self.assertEqual(original.read_bytes(),before)

    def test_invalid_paths_and_source_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            with self.assertRaisesRegex(ValueError,'replace source'):prepare(source,source)
            spec=dict(files=[dict(path='../escape')])
            with self.assertRaisesRegex(ValueError,'bundle path'):prepare(source,root/'out',spec)

    def test_failed_conversion_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();target=root/'out'
            spec=dict(files=[dict(path='transformer/w.safetensors')])
            with patch('backend.app.inference.models.z_image.verify_bundle'),patch('scripts.prepare_z_image_bf16.convert',side_effect=RuntimeError('conversion failed')):
                with self.assertRaisesRegex(RuntimeError,'failed'):prepare(source,target,spec)
            self.assertFalse((target/'download_manifest.json').exists())
            self.assertEqual(list((target/'transformer').iterdir()),[])
