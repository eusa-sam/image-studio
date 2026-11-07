import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import flux2


class KleinBundleTests(unittest.TestCase):
    def test_lora_plan_preserves_verified_selection_and_rejects_wrong_family_or_duplicates(self):
        from backend.app.models import preflight,ComponentProfile
        model=dict(id='klein',family='flux2',available=True,path='checkpoint/klein',fingerprint={})
        profile=ComponentProfile(schema_version=1,kind='lora',family='flux2',sha256='a'*64,scale=1,source='fixture').model_dump()
        adapter=dict(id='style',component='lora',profile=profile,path='lora/style.safetensors',fingerprint={'bytes':123})
        settings=dict(model='klein',prediction_type='flow',guidance=1.0,loras=[dict(model='style',weight=.7)])
        with patch('backend.app.models.scan',return_value=[model,adapter]):
            plan=preflight(settings)
            self.assertEqual(plan['loras'],[{key:adapter[key] for key in ('id','path','fingerprint','profile')} | {'weight':.7}])
            with self.assertRaisesRegex(ValueError,'중복'):preflight(settings|{'loras':settings['loras']*2})
            adapter['profile']['family']='sdxl'
            with self.assertRaisesRegex(ValueError,'FLUX.2'):preflight(settings)

    def test_real_klein_lora_injection_and_strength_preserve_base_weights(self):
        import torch
        from safetensors.torch import save_file
        from diffusers import Flux2KleinPipeline,Flux2Transformer2DModel
        from backend.app.lora import apply_loras,set_lora_weights
        transformer=Flux2Transformer2DModel(in_channels=8,num_layers=1,num_single_layers=1,
            attention_head_dim=16,num_attention_heads=2,joint_attention_dim=32,
            timestep_guidance_channels=32,axes_dims_rope=(4,4,4,4),guidance_embeds=False)
        pipe=Flux2KleinPipeline(transformer=transformer,vae=None,scheduler=None,text_encoder=None,tokenizer=None)
        base=transformer.x_embedder.weight.detach().clone();bias=None if transformer.x_embedder.bias is None else transformer.x_embedder.bias.detach().clone()
        generator=torch.Generator().manual_seed(31)
        a=torch.randn(2,8,generator=generator)*.1;b=torch.randn(32,2,generator=generator)*.1
        x=torch.randn(3,8,generator=generator)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'style.safetensors'
            save_file({'transformer.x_embedder.lora_A.weight':a,'transformer.x_embedder.lora_B.weight':b},path)
            selection=[({'weight':1.0},path)];apply_loras(pipe,selection)
            for strength in (1.,.5,0.,1.):
                selection[0][0]['weight']=strength;set_lora_weights(pipe,selection)
                torch.testing.assert_close(transformer.x_embedder(x),torch.nn.functional.linear(x,base+strength*(b@a),bias),rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(transformer.x_embedder.base_layer.weight,base,rtol=0,atol=0)

    def test_lora_loading_precedes_device_policy_and_is_never_fused(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        pipe=SimpleNamespace();calls=[];selections=[({'weight':.4},Path('local.safetensors'))]
        def inject(actual,rows):
            self.assertIs(actual,pipe);self.assertIs(rows,selections);calls.append('lora')
        def configure(actual,width,height):
            self.assertEqual(calls,['lora']);self.assertEqual((width,height),(768,1024));return {}
        with patch('diffusers.Flux2KleinPipeline.from_pretrained',return_value=pipe) as load,patch('backend.app.lora.apply_loras',side_effect=inject),patch('backend.app.memory_policy.configure_pipeline',side_effect=configure):
            self.assertIs(flux2.load_pipeline('local',768,1024,loras=selections),pipe)
            self.assertTrue(load.call_args.kwargs['local_files_only'])
            self.assertEqual(pipe.memory_policy['family'],'flux2')

    def test_profile_scan_accepts_flux2_lora_without_enabling_external_components(self):
        import torch
        from safetensors.torch import save_file
        from backend.app.models import scan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'lora';folder.mkdir();path=folder/'style.safetensors'
            save_file({'transformer.x_embedder.lora_A.weight':torch.ones(2,8),'transformer.x_embedder.lora_B.weight':torch.ones(32,2)},path)
            profile=dict(schema_version=1,kind='lora',family='flux2',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=1,source='test fixture')
            sidecar=path.with_name(path.name+'.profile.json');sidecar.write_text(json.dumps(profile))
            with patch('backend.app.models.MODEL_ROOT',root):
                rows=scan();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['component'],'lora')
                self.assertEqual(rows[0]['profile']['family'],'flux2')
                profile['kind']='vae';sidecar.write_text(json.dumps(profile))
                self.assertNotIn('component',scan()[0])

    def test_worker_passes_verified_lora_files_and_cache_reuses_strength(self):
        from backend.app.worker import _run
        from backend.app.pipeline_cache import PipelineCache
        class StopAfterLoad(Exception):pass
        loras=[({'weight':.7},Path('verified.safetensors'))]
        plan=dict(runner='flux2_txt2img',settings=dict(width=768,height=1024),loras=[dict(id='style',weight=.7)])
        key=PipelineCache.identity(plan,(768,1024))
        plan['loras'][0]['weight']=.3
        self.assertEqual(key,PipelineCache.identity(plan,(768,1024)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'request.json';path.write_text(json.dumps(dict(plan=plan)))
            with patch('backend.app.worker.verify_inputs',return_value=(Path('bundle'),'hash',{},loras)),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.empty_cache'),patch('backend.app.flux2.load_pipeline',side_effect=StopAfterLoad) as load:
                with self.assertRaises(StopAfterLoad):_run(path)
                load.assert_called_once_with(Path('bundle'),768,1024,loras=loras)

    def test_cpu_retention_and_resume_reconfigure_without_reloading(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        from backend.app.pipeline_cache import PipelineCache
        module=torch.nn.Linear(4,4)
        pipe=SimpleNamespace(memory_policy=dict(family='flux2',mode='model_offload'),components={'transformer':module},remove_all_hooks=Mock(),to=Mock())
        cache=PipelineCache();cache.pipe=pipe;cache.key='model-key';cache.positive='one'
        cache.conditioning={'one':{'prompt_embeds':torch.ones(2,4)}}
        with patch('backend.app.memory_policy.available_ram',return_value=16*1024**3):
            self.assertTrue(cache.suspend())
            self.assertIs(cache.pipe,pipe);self.assertEqual(cache.key,'model-key')
            pipe.remove_all_hooks.assert_called_once();pipe.to.assert_called_once_with('cpu')
            with patch('backend.app.memory_policy.configure_pipeline',return_value={'mode':'sequential_offload'}) as configure:
                flux2.resume_pipeline(pipe,768,1024)
                configure.assert_called_once_with(pipe,768,1024)
                self.assertEqual(pipe.memory_policy,dict(mode='sequential_offload',family='flux2'))
        with patch('backend.app.memory_policy.available_ram',return_value=0):
            with self.assertRaisesRegex(RuntimeError,'RAM'):flux2.resume_pipeline(pipe,768,1024)
            self.assertFalse(cache.suspend());self.assertIsNone(cache.pipe)
            self.assertEqual(pipe.remove_all_hooks.call_count,1)

    def test_malformed_bundle_does_not_break_inventory(self):
        from backend.app.models import scan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'checkpoint/klein';folder.mkdir(parents=True)
            for content in ('[]','{}','{"_class_name":"OtherPipeline"}'):
                (folder/'model_index.json').write_text(content)
                with patch('backend.app.models.MODEL_ROOT',root):rows=scan()
                self.assertEqual(len(rows),1);self.assertFalse(rows[0]['available'])
                self.assertEqual(rows[0]['family'],'unknown')
            (folder/'download_manifest.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'Invalid Klein manifest'):flux2.verify_bundle(folder)

    def test_prompt_cache_retains_no_autograd_graph(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        weights=torch.nn.Linear(4,4)
        encode=Mock(side_effect=lambda **kwargs:(weights(torch.ones(2,4)),None))
        pipe=SimpleNamespace(encode_prompt=encode,_execution_device='cpu')
        cache={}
        first=flux2.prompt_arguments(pipe,'one',cache)
        self.assertFalse(first['prompt_embeds'].requires_grad)
        self.assertIsNone(first['prompt_embeds'].grad_fn)
        self.assertIs(flux2.prompt_arguments(pipe,'one',cache)['prompt_embeds'],first['prompt_embeds'])
        encode.assert_called_once()

    def test_flow_preflight_preserves_reference_kind_and_rejects_other_components(self):
        from backend.app.models import preflight
        from uuid import uuid4
        rows=[dict(id='klein',family='flux2',available=True,path='checkpoint/klein',fingerprint={})]
        base=dict(model='klein',prediction_type='flow',guidance=1.0,steps=4)
        with patch('backend.app.models.scan',return_value=rows):
            self.assertEqual(preflight(base)['runner'],'flux2_txt2img')
            for kind in ('reference_job','reference_attachment'):
                ref=str(uuid4());plan=preflight(base|dict(flow='reference_edit',**{kind:ref}))
                self.assertEqual(plan['runner'],'flux2_reference_edit')
                self.assertEqual(plan['settings'][kind],ref)
            for changed in ({'flow':'img2img'},{'guidance':0.0},{'prediction_type':'epsilon'},
                            {'flow':'reference_edit'},{'reference_job':str(uuid4())},{'clip':'wrong'},
                            {'face_detector':'face'},{'loras':[dict(model='lora')]},
                            {'flow':'reference_edit','reference_job':str(uuid4()),'reference_attachment':str(uuid4())}):
                with self.assertRaises(ValueError):preflight(base|changed)

    def test_tokens_use_real_template_and_negative_is_not_counted(self):
        from unittest.mock import Mock
        tokenizer=Mock();tokenizer.apply_chat_template.return_value='templated'
        tokenizer.return_value={'input_ids':list(range(513))}
        result=flux2.count_prompt_tokens(None,'subject','unused',tokenizer)
        self.assertTrue(result['truncated']);self.assertFalse(result['negative_used'])
        tokenizer.apply_chat_template.assert_called_once_with([dict(role='user',content='subject')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        self.assertEqual(result['encoders'][0]['negative'],0)

    def test_bundle_source_hash_size_revision_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            content=b'weights';path=folder/'model.safetensors';path.write_bytes(content)
            digest=hashlib.sha256(content).hexdigest()
            spec=dict(repository='test/model',revision='a'*40,files=[dict(path=path.name,size=len(content),sha256=digest,blob_id=None)])
            manifest=dict(repository=spec['repository'],revision=spec['revision'],files=[dict(path=path.name,sha256=digest)])
            target=folder/'download_manifest.json'
            target.write_text(json.dumps(manifest))
            with patch.object(flux2,'SPEC',spec):
                self.assertEqual(flux2.verify_bundle(folder),manifest)
                self.assertEqual(set(flux2.bundle_fingerprint(folder)['files']),{path.name,target.name})
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):flux2.verify_bundle(folder)
                path.write_bytes(b'short')
                with self.assertRaisesRegex(ValueError,'incomplete'):flux2.verify_bundle(folder,False)
                path.write_bytes(content)
                manifest['revision']='b'*40;target.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'revision'):flux2.verify_bundle(folder)

    def test_reject_manifest_injection_duplicate_missing_and_rewritten_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'file').write_bytes(b'x')
            digest=hashlib.sha256(b'x').hexdigest()
            spec=dict(repository='test',revision='a'*40,files=[dict(path='file',size=1,sha256=digest)])
            row=dict(path='file',sha256=digest)
            cases=[([dict(path='../escape',sha256=digest)],'path'),([row,row],'duplicate'),([], 'Incomplete'),([dict(path='file',sha256='0'*64)],'source hash')]
            with patch.object(flux2,'SPEC',spec):
                for records,error in cases:
                    (folder/'download_manifest.json').write_text(json.dumps(dict(repository='test',revision='a'*40,files=records)))
                    with self.assertRaisesRegex(ValueError,error):flux2.verify_bundle(folder,False)

    def test_config_blob_identity_even_if_local_manifest_is_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'config.json').write_bytes(b'[]')
            spec=dict(repository='test',revision='a'*40,files=[dict(path='config.json',size=2,sha256=None,blob_id=hashlib.sha1(b'blob 2\0{}').hexdigest())])
            (folder/'download_manifest.json').write_text(json.dumps(dict(repository='test',revision='a'*40,files=[dict(path='config.json',sha256=hashlib.sha256(b'[]').hexdigest())])))
            with patch.object(flux2,'SPEC',spec),self.assertRaisesRegex(ValueError,'hash mismatch'):
                flux2.verify_bundle(folder)
