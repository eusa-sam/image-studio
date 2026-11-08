import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

class ZImageLoraTests(unittest.TestCase):
    def test_lora_plan_preserves_verified_selection_and_rejects_wrong_family_or_duplicates(self):
        from backend.app.models import preflight,ComponentProfile
        model=dict(id='klein',family='zimage',available=True,path='checkpoint/klein',fingerprint={})
        profile=ComponentProfile(schema_version=1,kind='lora',family='zimage',sha256='a'*64,scale=1,source='fixture').model_dump()
        adapter=dict(id='style',component='lora',profile=profile,path='lora/style.safetensors',fingerprint={'bytes':123})
        settings=dict(model='klein',prediction_type='flow',guidance=0.0,loras=[dict(model='style',weight=.7)])
        with patch('backend.app.models.scan',return_value=[model,adapter]):
            plan=preflight(settings)
            self.assertEqual(plan['loras'],[{key:adapter[key] for key in ('id','path','fingerprint','profile')} | {'weight':.7}])
            with self.assertRaisesRegex(ValueError,'중복'):preflight(settings|{'loras':settings['loras']*2})
            adapter['profile']['family']='sdxl'
            with self.assertRaisesRegex(ValueError,'Z-Image'):preflight(settings)

    def test_profile_scan_accepts_zimage_lora_without_enabling_external_components(self):
        import torch
        from safetensors.torch import save_file
        from backend.app.models import scan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'lora';folder.mkdir();path=folder/'style.safetensors'
            save_file({'transformer.x_embedder.lora_A.weight':torch.ones(2,8),'transformer.x_embedder.lora_B.weight':torch.ones(32,2)},path)
            profile=dict(schema_version=1,kind='lora',family='zimage',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=1,source='test fixture')
            sidecar=path.with_name(path.name+'.profile.json');sidecar.write_text(json.dumps(profile))
            with patch('backend.app.models.MODEL_ROOT',root):
                rows=scan();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['component'],'lora')
                self.assertEqual(rows[0]['profile']['family'],'zimage')
                profile['kind']='vae';sidecar.write_text(json.dumps(profile))
                self.assertNotIn('component',scan()[0])

    def test_worker_passes_verified_lora_files_and_cache_reuses_strength(self):
        from backend.app.worker import _run
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        class StopAfterLoad(Exception):pass
        loras=[({'weight':.7},Path('verified.safetensors'))]
        plan=dict(runner='zimage_txt2img',settings=dict(width=768,height=1024),loras=[dict(id='style',weight=.7)])
        key=PipelineCache.identity(plan,(768,1024))
        plan['loras'][0]['weight']=.3
        self.assertEqual(key,PipelineCache.identity(plan,(768,1024)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'request.json';path.write_text(json.dumps(dict(plan=plan)))
            with patch('backend.app.worker.verify_inputs',return_value=(Path('bundle'),'hash',{},loras)),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.empty_cache'),patch('backend.app.inference.models.z_image.load_pipeline',side_effect=StopAfterLoad) as load:
                with self.assertRaises(StopAfterLoad):_run(path)
                load.assert_called_once_with(Path('bundle'),768,1024,loras=loras)


    def test_real_zimage_adapter_preserves_base_and_returns_to_zero(self):
        import torch
        from diffusers import ZImagePipeline,ZImageTransformer2DModel
        from safetensors.torch import save_file
        from backend.app.inference.conditioning.lora import apply_loras, set_lora_weights
        model=ZImageTransformer2DModel(dim=32,n_layers=1,n_refiner_layers=1,n_heads=2,n_kv_heads=2,cap_feat_dim=32,axes_dims=[4,6,6],axes_lens=[32,32,32])
        pipe=ZImagePipeline(transformer=model,vae=None,scheduler=None,text_encoder=None,tokenizer=None)
        layer=model.layers[0].attention.to_q;original=layer.weight.detach().clone()
        x=torch.arange(64,dtype=torch.float32).reshape(2,32)/64;baseline=layer(x).detach().clone()
        a=torch.full((2,32),.1);b=torch.full((32,2),.1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lora.safetensors'
            save_file({'diffusion_model.layers.0.attention.to_q.lora_A.weight':a,'diffusion_model.layers.0.attention.to_q.lora_B.weight':b},str(path))
            selections=[({'weight':1.},path)];apply_loras(pipe,selections)
            self.assertTrue(torch.equal(model.layers[0].attention.to_q.get_base_layer().weight,original))
            for weight in (1.,.7,0.,1.):
                selections[0][0]['weight']=weight;set_lora_weights(pipe,selections)
                torch.testing.assert_close(model.layers[0].attention.to_q(x),baseline+weight*torch.nn.functional.linear(torch.nn.functional.linear(x,a),b),atol=1e-6,rtol=1e-5)
            from backend.app.inference.runtime.pipeline_cache import PipelineCache
            cache=PipelineCache();cache.pipe=pipe
            plan=dict(runner='zimage_txt2img',model_path='same',loras=[dict(path=str(path),weight=1.)])
            cache.key=cache.identity(plan,(768,1024))
            plain=cache.identity(plan|{'loras':[]},(768,1024))
            self.assertIs(cache.get(plain,'same prompt',replace_loras=[]),pipe)
            self.assertEqual(cache.key,plain)
            self.assertFalse(model.layers[0].attention.to_q.lora_A)
            self.assertTrue(torch.equal(model.layers[0].attention.to_q(x),baseline))
            cache.suspended=True
            self.assertIs(cache.get(cache.identity(plan,(768,1024)),'same prompt',replace_loras=selections),pipe)
            torch.testing.assert_close(model.layers[0].attention.to_q(x),baseline+torch.nn.functional.linear(torch.nn.functional.linear(x,a),b),atol=1e-6,rtol=1e-5)
            self.assertTrue(torch.equal(model.layers[0].attention.to_q.get_base_layer().weight,original))
            second=Path(folder)/'second.safetensors'
            save_file({'diffusion_model.layers.0.attention.to_q.lora_A.weight':a,'diffusion_model.layers.0.attention.to_q.lora_B.weight':b*2},str(second))
            changed=cache.identity(plan|{'loras':[dict(path=str(second))]},(768,1024))
            self.assertIs(cache.get(changed,'same prompt',replace_loras=[({'weight':1.},second)]),pipe)
            torch.testing.assert_close(model.layers[0].attention.to_q(x),baseline+2*torch.nn.functional.linear(torch.nn.functional.linear(x,a),b),atol=1e-6,rtol=1e-5)
            self.assertTrue(torch.equal(model.layers[0].attention.to_q.get_base_layer().weight,original))

    def test_adapter_removal_does_not_reuse_changed_base_or_retain_failed_pipeline(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        from unittest.mock import Mock
        for changed in ('model_path','fingerprint','components'):
            cache=PipelineCache();pipe=Mock();cache.pipe=pipe
            plan=dict(runner='zimage_txt2img',model_path='old',loras=[dict(path='a')])
            cache.key=cache.identity(plan,(768,1024))
            value={'vae':'changed'} if changed=='components' else 'changed'
            key=cache.identity(plan|{'loras':[],changed:value},(768,1024))
            self.assertIsNone(cache.get(key,'prompt',replace_loras=[]));pipe.delete_adapters.assert_not_called()
        cache=PipelineCache();cache.pipe=Mock(_cpu_weights=None);cache.pipe.delete_adapters.side_effect=ValueError('partial deletion')
        cache.key=cache.identity(plan,(768,1024))
        with self.assertRaisesRegex(ValueError,'partial deletion'):
            cache.get(cache.identity(plan|{'loras':[]},(768,1024)),'prompt',replace_loras=[])
        self.assertIsNone(cache.pipe);self.assertIsNone(cache.key)

    def test_adapter_addition_replans_residency_and_drops_failed_injection(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        from unittest.mock import Mock
        plan=dict(runner='zimage_txt2img',model_path='same',loras=[])
        selection=[({'weight':.7},Path('style.safetensors'))]
        for retained in (True,False):
            cache=PipelineCache();pipe=Mock();cache.pipe=pipe
            cache.key=cache.identity(plan,(768,1024))
            def suspend():
                if retained:cache.suspended=True
                else:cache.clear()
                return retained
            cache.suspend=Mock(side_effect=suspend)
            key=cache.identity(plan|{'loras':[dict(path='style.safetensors')]},(768,1024))
            with patch('backend.app.inference.conditioning.lora.apply_loras',side_effect=ValueError('invalid target')) as apply:
                if retained:
                    with self.assertRaisesRegex(ValueError,'invalid target'):cache.get(key,'prompt',replace_loras=selection)
                    apply.assert_called_once_with(pipe,selection)
                else:
                    self.assertIsNone(cache.get(key,'prompt',replace_loras=selection));apply.assert_not_called()
            cache.suspend.assert_called_once();pipe.delete_adapters.assert_not_called()
            self.assertIsNone(cache.pipe);self.assertIsNone(cache.key)
