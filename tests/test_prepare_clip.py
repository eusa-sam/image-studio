import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts import prepare_clip

class PrepareClipTests(unittest.TestCase):
    def test_resolved_model_root_exports_inside_store_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);project=root/'project';project.mkdir();store=root/'nvme-models';store.mkdir()
            logical=project/'models';source=store/'checkpoint.safetensors';source.write_bytes(b'owned checkpoint')
            original=Path.resolve
            def resolve(path,*args,**kwargs):
                return store if path==logical else original(path,*args,**kwargs)
            encoder=torch.nn.Linear(3,4)
            args=['prepare_clip','checkpoint.safetensors','text_encoders/export.safetensors','--family','sd1']
            with patch.object(prepare_clip,'__file__',str(project/'scripts/prepare_clip.py')),patch.object(Path,'resolve',resolve),patch('sys.argv',args),patch('diffusers.StableDiffusionPipeline.from_single_file',return_value=SimpleNamespace(text_encoder=encoder)) as load:
                prepare_clip.main()
                self.assertEqual(load.call_args.args[0],str(source))
                self.assertEqual(load.call_args.kwargs['config'],str(store/'configs/sd1'))
                self.assertTrue(load.call_args.kwargs['local_files_only'])
                target=store/'text_encoders/export.safetensors'
                profile=json.loads(target.with_name(target.name+'.profile.json').read_text())
                self.assertEqual(profile['sha256'],hashlib.sha256(target.read_bytes()).hexdigest())
                self.assertEqual(source.read_bytes(),b'owned checkpoint')
                with self.assertRaisesRegex(ValueError,'already exists'):prepare_clip.main()
                args[1]='../outside.safetensors';args[2]='text_encoders/second.safetensors'
                with self.assertRaisesRegex(ValueError,'Use models paths'):prepare_clip.main()
                self.assertEqual(load.call_count,1)
