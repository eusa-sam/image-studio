import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from backend.app.inference.models.anima import verify_bundle, configure_pipeline
from backend.app.inference.runtime.pipeline_cache import PipelineCache


class Prepared(Exception): pass


class AnimaReuseTests(unittest.TestCase):
    def test_flow_scheduler_reuse_resets_steps_and_image_start_index(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from backend.app.inference.runtime.memory_policy import fresh_scheduler
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3)
        for steps,start in ((4,0),(6,2),(3,0),(4,0)):
            def sample(current):
                current.set_timesteps(steps,device='cpu')
                self.assertIsNone(current.step_index);self.assertIsNone(current.begin_index)
                current.set_begin_index(start)
                value=torch.ones(1,2,4,4)
                for timestep in current.timesteps[start:]:value=current.step(value*.1,timestep,value,return_dict=False)[0]
                return value
            expected=sample(fresh_scheduler(scheduler));actual=sample(scheduler)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_image_and_detail_crop_sizes_are_passed_to_modular_pipeline(self):
        from PIL import Image
        from backend.app.worker import pipeline_output
        pipe=Mock(memory_policy={'family':'anima'},weight_maps=(),vae=SimpleNamespace(use_tiling=True,decode=Mock()))
        for size in ((768,1024),(512,384)):
            image=Image.new('RGB',size)
            pipeline_output(pipe,image=image,strength=.5)
            self.assertEqual(pipe.call_args.kwargs['width'],size[0])
            self.assertEqual(pipe.call_args.kwargs['height'],size[1])
        pipeline_output(pipe,image=image,width=256,height=256)
        self.assertEqual((pipe.call_args.kwargs['width'],pipe.call_args.kwargs['height']),(256,256))
        pipe.memory_policy={'family':'sdxl'}
        pipeline_output(pipe,image=image)
        self.assertNotIn('width',pipe.call_args.kwargs)

    def test_cached_hash_is_compared_with_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'inside').write_bytes(b'x')
            (root/'download_manifest.json').write_text(json.dumps({'files':[{'path':'inside','sha256':hashlib.sha256(b'x').hexdigest()}]}))
            cache=Mock();cache.file_hash.return_value='0'*64
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_bundle(root,cache=cache)
            cache.file_hash.assert_called_once_with(root/'inside')

    def test_bundle_changes_invalidate_single_checkpoint_cache(self):
        plan=dict(runner='anima_txt2img',model_path='single',anima={'bundle':'base'},bundle_fingerprint={'files':{'config':1}})
        key=PipelineCache.identity(plan,(768,1024))
        for changes in ({'anima':{'bundle':'other'}},{'bundle_fingerprint':{'files':{'config':2}}}):
            self.assertNotEqual(key,PipelineCache.identity(plan|changes,(768,1024)))

    def test_strength_reuses_identity_but_file_changes_do_not(self):
        plan=dict(runner='anima_txt2img',loras=[dict(path='first',fingerprint={'bytes':1},weight=1),dict(path='second',weight=.7)])
        key=PipelineCache.identity(plan,(768,1024))
        changed=[dict(row,weight=0) for row in plan['loras']]
        self.assertEqual(key,PipelineCache.identity(plan|{'loras':changed},(768,1024)))
        for rows in (changed[:1],[],[dict(changed[0],fingerprint={'bytes':2}),changed[1]]):
            self.assertNotEqual(key,PipelineCache.identity(plan|{'loras':rows},(768,1024)))

    def test_adapter_removal_suspends_manager_before_mutating_components(self):
        plan=dict(runner='anima_txt2img',loras=[dict(path='first',weight=1),dict(path='second',weight=.7)])
        cache=PipelineCache();cache.key=cache.identity(plan,(768,1024))
        manager=Mock();pipe=SimpleNamespace(memory_policy={'family':'anima'},anima_manager=manager,
            components={'transformer':torch.nn.Linear(2,2)},to=Mock(),delete_adapters=Mock())
        pipe.delete_adapters.side_effect=lambda names: self.assertTrue(cache.suspended)
        cache.pipe=pipe
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
            self.assertIs(cache.get(cache.identity(plan|{'loras':[]},(768,1024)),'same',replace_loras=[]),pipe)
        manager.disable_auto_cpu_offload.assert_called_once()
        pipe.delete_adapters.assert_called_once_with(['lora_0','lora_1'])
        choices=[({'weight':1},'first')]
        with patch('backend.app.inference.conditioning.lora.apply_loras') as apply:
            self.assertIs(cache.get(cache.identity(plan|{'loras':plan['loras'][:1]},(768,1024)),'same',replace_loras=choices),pipe)
            apply.assert_called_once_with(pipe,choices)

    def test_manager_budget_and_cpu_retention(self):
        gb=1024**3
        for ram in (32*gb,0):
            manager=Mock();module=torch.nn.Linear(2,2)
            pipe=SimpleNamespace(anima_manager=manager,components={'transformer':module},to=Mock(),
                                 **{name:module for name in ('transformer','text_encoder','text_conditioner','vae')})
            with patch('backend.app.inference.runtime.memory_policy.available_vram',return_value=(12*gb,16*gb,{'cuda_free_bytes':13*gb})),patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=ram):
                configure_pipeline(pipe,(768,1024))
                manager.enable_auto_cpu_offload.assert_called_once_with(device='cuda',memory_reserve_margin=pipe.memory_policy['activation_reserve_bytes']+gb)
                cache=PipelineCache();cache.pipe=pipe
                self.assertEqual(cache.suspend(),bool(ram))
            if ram:
                manager.disable_auto_cpu_offload.assert_called_once();pipe.to.assert_called_once_with('cpu')
                self.assertIs(cache.pipe,pipe)
            else:
                manager.disable_auto_cpu_offload.assert_not_called();pipe.to.assert_not_called();self.assertIsNone(cache.pipe)

    def test_reuse_updates_guidance_without_registering_scheduler_after_cpu_resume(self):
        from diffusers import FlowMatchEulerDiscreteScheduler,ClassifierFreeGuidance
        from backend.app.worker import _run
        for guidance,suspended in ((4.,False),(1.,True)):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);path=root/'request.json'
                plan=dict(runner='anima_txt2img',settings=dict(width=768,height=1024,guidance=guidance))
                path.write_text(json.dumps(dict(plan=plan,positive='same')))
                scheduler=FlowMatchEulerDiscreteScheduler(shift=3)
                pipe=SimpleNamespace(anima_manager=Mock(),scheduler=scheduler,guider=ClassifierFreeGuidance(guidance_scale=2),update_components=Mock(),memory_policy={})
                cache=Mock();cache.get.return_value=pipe;cache.suspended=suspended
                loras=[({'weight':0},'first'),({'weight':.7},'second')]
                with patch('backend.app.worker.verify_inputs',return_value=(root/'bundle','verified',{},loras)),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3),patch('backend.app.inference.models.anima.configure_pipeline') as configure,patch('backend.app.inference.models.anima.load_pipeline') as load,patch('backend.app.inference.conditioning.lora.set_lora_weights') as scales:
                    def stop_after_prepare(message,**kwargs):
                        if str(message).startswith('{"memory_policy"'):raise Prepared
                    with patch('backend.app.worker.print',side_effect=stop_after_prepare):
                        with self.assertRaises(Prepared):_run(path,cache)
                load.assert_not_called()
                scales.assert_called_once_with(pipe,loras)
                self.assertEqual(cache.get.call_args.kwargs,{'replace_loras':loras})
                options=pipe.update_components.call_args.kwargs
                self.assertEqual(options['guider'].config.guidance_scale,guidance)
                self.assertNotIn('scheduler',options)
                self.assertIs(pipe.scheduler,scheduler)
                self.assertFalse(cache.suspended)
                self.assertEqual(configure.call_count,2 if suspended else 1)
                configure.assert_called_with(pipe,(768,1024))
