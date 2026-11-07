import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from backend.app import models

class BundleInventoryTests(unittest.TestCase):
    def test_scan_does_not_traverse_sources_or_bundle_internals(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=root/'checkpoint'/'bundle';bundle.mkdir(parents=True)
            (bundle/'model_index.json').write_text('{"_class_name":"FuturePipeline"}')
            sources=root/'sources';sources.mkdir()
            standalone=root/'lora';standalone.mkdir();(standalone/'candidate.gguf').write_bytes(b'fixture')
            original=os.scandir
            def guarded(path):
                if Path(path) in (sources,bundle):raise AssertionError('unnecessary subtree traversal')
                return original(path)
            with patch.object(models,'MODEL_ROOT',root),patch('os.scandir',side_effect=guarded):
                rows=models.scan()
            self.assertEqual([row['path'] for row in rows],['checkpoint/bundle','lora/candidate.gguf'])

    def test_folder_bundles_hide_internal_weights_even_when_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);expected={}
            for name,filename,kind in [('klein','model_index.json','Flux2KleinPipeline'),('z','model_index.json','ZImagePipeline'),('anima','modular_model_index.json','AnimaModularPipeline'),('unknown','model_index.json','FuturePipeline')]:
                bundle=root/'checkpoint'/name;bundle.mkdir(parents=True)
                (bundle/filename).write_text(json.dumps({'_class_name':kind}))
                if kind=='ZImagePipeline':(bundle/'download_manifest.json').write_text('{}')
                for component in ('transformer','text_encoder','vae'):
                    part=bundle/component;part.mkdir()
                    (part/'weights.safetensors').write_bytes(b'not a standalone checkpoint')
                expected['checkpoint/'+name]=name!='unknown'
            with patch.object(models,'MODEL_ROOT',root),patch.object(models,'fingerprint',return_value={'bundle':'fixture'}):rows=models.scan()
            self.assertEqual({row['path']:row['available'] for row in rows},expected)
            self.assertEqual(len({row['id'] for row in rows}),4)
            self.assertTrue(all('/transformer/' not in row['path'] for row in rows))

    def test_zimage_variant_comes_from_manifest_not_folder_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,variant in [('misleading-bf16',None),('renamed-prepared','bf16-local-v1'),('unknown','untrusted')]:
                bundle=root/'checkpoint'/name;bundle.mkdir(parents=True)
                (bundle/'model_index.json').write_text('{"_class_name":"ZImagePipeline"}')
                (bundle/'download_manifest.json').write_text(json.dumps({} if variant is None else {'variant':variant}))
            with patch.object(models,'MODEL_ROOT',root),patch.object(models,'fingerprint',return_value={'bundle':'fixture'}):
                rows={row['name']:row for row in models.scan()}
            self.assertEqual(rows['misleading-bf16']['variant'],'original')
            self.assertEqual(rows['renamed-prepared']['variant'],'bf16-local-v1')
            self.assertFalse(rows['unknown']['available'])
