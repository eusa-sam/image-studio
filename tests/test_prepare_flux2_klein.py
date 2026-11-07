import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_flux2_klein as prepare


class PrepareKleinTests(unittest.TestCase):
    def test_longcat_preparation_is_pinned_and_stays_outside_inventory(self):
        from scripts import prepare_longcat
        spec = prepare_longcat.SPEC
        self.assertEqual(spec['repository'], 'meituan-longcat/LongCat-Image-Edit-Turbo')
        self.assertEqual(spec['revision'], '6a7262de5549f0bf0ec54c08ef7d283ef41f3214')
        self.assertEqual(len({row['path'] for row in spec['files']}), len(spec['files']))
        self.assertTrue(all(row['sha256'] or row['blob_id'] for row in spec['files']))
        with patch.object(prepare_longcat, 'prepare') as run:
            prepare_longcat.main()
            run.assert_called_once_with(spec=spec, slug='longcat-image-edit-turbo')

    def test_second_bundle_uses_explicit_spec_and_isolated_directory(self):
        from scripts import prepare_z_image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'{}')
            row=dict(path='model_index.json',size=2,sha256=None,blob_id=hashlib.sha1(b'blob 2\0{}').hexdigest())
            spec=dict(repository='official/z-image',revision='b'*40,files=[row])
            with patch.object(prepare,'ROOT',root),patch.object(prepare_z_image,'SPEC',spec),patch.object(prepare,'hf_hub_download',return_value=str(source)) as download:
                prepare_z_image.main();prepare_z_image.main()
                download.assert_called_once_with(spec['repository'],row['path'],revision=spec['revision'],token=False)
                self.assertTrue((root/'models/sources/z-image-turbo/download_manifest.json').is_file())
                self.assertFalse((root/'models/sources/flux2-klein-4b').exists())
                for slug in ('','..','../escape'):
                    with self.assertRaisesRegex(ValueError,'Invalid bundle directory'):prepare.main(spec=spec,slug=slug)

    def test_prepare_resume_and_preserve_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'weights')
            row=dict(path='transformer/model.safetensors',size=7,sha256=hashlib.sha256(b'weights').hexdigest(),blob_id=None)
            spec=dict(repository='official/model',revision='a'*40,files=[row])
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'SPEC',spec),patch.object(prepare,'hf_hub_download',return_value=str(source)) as download:
                prepare.main();prepare.main()
                download.assert_called_once_with(spec['repository'],row['path'],revision=spec['revision'],token=False)
                folder=root/'models/sources/flux2-klein-4b'
                installed=root/'models/checkpoint/flux2-klein-4b';installed.parent.mkdir(parents=True)
                folder.rename(installed);folder=installed
                prepare.main();self.assertEqual(download.call_count,1)
                self.assertEqual(json.loads((folder/'download_manifest.json').read_text())['files'][0]['sha256'],row['sha256'])
                target=folder/row['path'];target.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):prepare.main()
                self.assertEqual(target.read_bytes(),b'changed')

    def test_git_blob_and_size_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config';path.write_bytes(b'{}')
            row=dict(path='config.json',size=2,sha256=None,blob_id=hashlib.sha1(b'blob 2\0{}').hexdigest())
            self.assertEqual(prepare.verify(path,row),hashlib.sha256(b'{}').hexdigest())
            path.write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):prepare.verify(path,row)
            path.write_bytes(b'incomplete')
            with self.assertRaisesRegex(ValueError,'size mismatch'):prepare.verify(path,row)

    def test_corrupt_download_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'bad')
            row=dict(path='model.safetensors',size=3,sha256='0'*64,blob_id=None)
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'SPEC',dict(repository='repo',revision='a'*40,files=[row])),patch.object(prepare,'hf_hub_download',return_value=str(source)):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):prepare.main()
                self.assertFalse((root/'models/sources/flux2-klein-4b/model.safetensors').exists())

    def test_path_escape_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(prepare,'ROOT',Path(directory)),patch.object(prepare,'SPEC',dict(files=[dict(path='../escape')])),patch.object(prepare,'hf_hub_download') as download:
                with self.assertRaisesRegex(ValueError,'Invalid bundle path'):prepare.main()
                download.assert_not_called()
