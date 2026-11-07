import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import z_image
from backend.app.pipeline_cache import PipelineCache


class ZImageBundleTests(unittest.TestCase):
    def test_calibrated_reserve_uses_current_free_memory_and_keeps_large_fallback(self):
        import torch
        from unittest.mock import Mock
        from backend.app.memory_policy import configure_pipeline,residency_plan
        gb=1024**3
        transformer=torch.nn.Module();transformer.register_parameter('weight',torch.nn.Parameter(torch.empty(6_150_000_000,device='meta',dtype=torch.bfloat16)))
        encoder=torch.nn.Module();encoder.register_parameter('weight',torch.nn.Parameter(torch.empty(4_000_000_000,device='meta',dtype=torch.bfloat16)))
        pipe=Mock();pipe.components={'transformer':transformer,'text_encoder':encoder}
        events=[]
        with patch('torch.cuda.empty_cache',side_effect=lambda:events.append('release')),patch('backend.app.memory_policy.available_vram',side_effect=lambda:(events.append('measure') or (15*gb,32*gb,{}))),patch('backend.app.memory_policy.available_ram',return_value=32*gb):
            configure_pipeline(pipe,768,1024,activation_reserve_bytes=z_image.activation_reserve(768,1024))
        self.assertEqual(events,['release','measure'])
        with patch('backend.app.memory_policy.available_ram',return_value=32*gb):
            for free,expected in [(15*gb,'model_offload'),(13*gb,'sequential_offload'),(24*gb,'full')]:
                with patch('backend.app.memory_policy.available_vram',return_value=(free,32*gb,{})):
                    policy=configure_pipeline(pipe,1024,1024,activation_reserve_bytes=z_image.activation_reserve(1024,1024))
                self.assertEqual(policy['mode'],expected)
        self.assertIsNone(z_image.activation_reserve(1024,1536))
        self.assertEqual(residency_plan(16*gb,16*gb,12*gb,1024,1024)['activation_reserve_bytes'],4*gb)
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):residency_plan(16*gb,16*gb,12*gb,1024,1024,activation_reserve_bytes=value)

    def test_preflight_rejects_incompatible_flows_and_preserves_steps(self):
        from backend.app.models import preflight
        model=dict(id='z',family='zimage',available=True,path='checkpoint/z',fingerprint={})
        up=dict(id='up',component='upscaler',path='upscale_models/up.pth',fingerprint={},profile={'scale':2})
        base=dict(model='z',prediction_type='flow',guidance=0,steps=9)
        with patch('backend.app.models.scan',return_value=[model,up]):
            self.assertEqual(preflight(base)['runner'],'zimage_txt2img')
            self.assertEqual(preflight(base|dict(steps=12))['settings']['steps'],12)
            self.assertEqual(preflight(base|dict(upscaler='up'))['components']['upscaler']['path'],up['path'])
            for changed in ({'guidance':1},{'prediction_type':'epsilon'},{'prompt_mode':'weighted'},
                            {'flow':'reference_edit'},{'clip':'wrong'},{'vae':'wrong'},
                            {'face_detector':'face'},{'loras':[dict(model='lora')]},{'reference_attachment':'reference'}):
                with self.assertRaises(ValueError):preflight(base|changed)

    def test_inventory_recognizes_only_complete_source_manifest(self):
        from backend.app.models import scan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'checkpoint/z';folder.mkdir(parents=True)
            content=b'{"_class_name":"ZImagePipeline"}';(folder/'model_index.json').write_bytes(content)
            sha=hashlib.sha256(content).hexdigest()
            spec=dict(repository='official/z',revision='a'*40,files=[dict(path='model_index.json',size=len(content),sha256=sha)])
            manifest=dict(repository=spec['repository'],revision=spec['revision'],files=[dict(path='model_index.json',sha256=sha)])
            with patch('backend.app.models.MODEL_ROOT',root),patch.object(z_image,'SPEC',spec):
                self.assertFalse(scan()[0]['available'])
                (folder/'download_manifest.json').write_text(json.dumps(manifest))
                rows=scan();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['family'],'zimage');self.assertTrue(rows[0]['available'])
                staged=root/'sources/z';staged.parent.mkdir();folder.rename(staged)
                self.assertEqual(scan(),[])

    def test_cpu_retention_and_resume_preserve_pipeline(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        pipe=SimpleNamespace(memory_policy=dict(family='zimage',mode='sequential_offload'),components={'transformer':torch.nn.Linear(2,2)},remove_all_hooks=Mock(),to=Mock())
        cache=PipelineCache();cache.pipe=pipe;cache.key='z'
        cache.conditioning={'one':{'prompt_embeds':torch.ones(2,3)}}
        with patch('backend.app.memory_policy.available_ram',return_value=16*1024**3),patch('backend.app.memory_policy.configure_pipeline',return_value={'mode':'model_offload'}):
            self.assertTrue(cache.suspend());self.assertIs(cache.pipe,pipe);self.assertEqual(cache.key,'z')
            z_image.resume_pipeline(pipe,768,1024)
            self.assertEqual(pipe.memory_policy,dict(mode='model_offload',family='zimage',text_encoder_released=True))
            pipe.text_encoder=torch.nn.Linear(2,2)
            z_image.resume_pipeline(pipe,768,1024)
            self.assertFalse(pipe.memory_policy['text_encoder_released'])
        with patch('backend.app.memory_policy.available_ram',return_value=0):
            with self.assertRaisesRegex(RuntimeError,'RAM'):z_image.resume_pipeline(pipe,768,1024)
            self.assertFalse(cache.suspend());self.assertIsNone(cache.pipe)

    def test_prompt_cache_has_no_graph_and_replaces_previous_text(self):
        import torch
        from unittest.mock import Mock
        weight=torch.nn.Parameter(torch.ones(2,3))
        pipe=Mock();pipe._execution_device=torch.device('cpu')
        pipe.encode_prompt.side_effect=lambda **kwargs:([weight*2],[])
        cache={}
        result=z_image.prompt_arguments(pipe,'one',cache)
        self.assertFalse(result['prompt_embeds'][0].requires_grad)
        self.assertIsNone(result['prompt_embeds'][0].grad_fn)
        again=z_image.prompt_arguments(pipe,'one',cache)
        self.assertIs(result['prompt_embeds'][0],again['prompt_embeds'][0])
        pipe.encode_prompt.assert_called_once_with(prompt='one',device=pipe._execution_device,do_classifier_free_guidance=False,max_sequence_length=512)
        z_image.prompt_arguments(pipe,'two',cache)
        self.assertEqual(set(cache),{'two'})

    def test_tokens_match_z_image_template_and_ignore_negative(self):
        from unittest.mock import Mock
        tokenizer=Mock();tokenizer.return_value={'input_ids':list(range(513))}
        result=z_image.count_prompt_tokens(None,'subject','unused',tokenizer)
        tokenizer.apply_chat_template.assert_called_once_with([dict(role='user',content='subject')],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        self.assertTrue(result['truncated']);self.assertFalse(result['negative_used'])
        self.assertEqual(result['encoders'][0]['negative'],0)

    def test_pinned_source_cache_and_foreign_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);weight=folder/'weight';weight.write_bytes(b'weights')
            sha=hashlib.sha256(b'weights').hexdigest()
            spec=dict(repository='official/z',revision='a'*40,files=[dict(path='weight',size=7,sha256=sha)])
            manifest=dict(repository=spec['repository'],revision=spec['revision'],files=[dict(path='weight',sha256=sha)])
            target=folder/'download_manifest.json';target.write_text(json.dumps(manifest))
            cache=PipelineCache()
            with patch.object(z_image,'SPEC',spec):
                z_image.verify_bundle(folder,cache=cache);z_image.verify_bundle(folder,cache=cache)
                self.assertEqual([row['reused'] for row in cache.hash_checks],[False,True])
                self.assertEqual(set(z_image.bundle_fingerprint(folder)['files']),{'download_manifest.json','weight'})
                weight.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Z-Image file hash mismatch'):z_image.verify_bundle(folder,cache=cache)
                manifest['repository']='other/model';target.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'revision'):z_image.verify_bundle(folder,False)

    def test_malformed_records_fail_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);spec=dict(repository='official/z',revision='a'*40,files=[])
            with patch.object(z_image,'SPEC',spec):
                for records in (None,{},'files',[None],[{}],[dict(path=None)],[dict(path='../escape')]):
                    target=folder/'download_manifest.json'
                    target.write_text(json.dumps(dict(repository=spec['repository'],revision=spec['revision'],files=records)))
                    with self.assertRaises(ValueError):z_image.verify_bundle(folder)
