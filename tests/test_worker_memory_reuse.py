import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch


class ReachedPipeline(Exception):pass


class WorkerMemoryReuseTests(unittest.TestCase):
    def test_klein_reuse_applies_current_lora_strength_after_resume(self):
        from backend.app.worker import _run
        gb=1024**3
        for strength,suspended in ((0.,False),(.0625,True)):
            with self.subTest(strength=strength),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                plan=dict(runner='flux2_txt2img',settings=dict(width=768,height=1024))
                path=root/'request.json';path.write_text(json.dumps(dict(plan=plan,positive='same')))
                pipe=SimpleNamespace(memory_policy={'free_vram_bytes':16*gb},set_adapters=Mock(side_effect=ReachedPipeline))
                cache=Mock();cache.get.return_value=pipe;cache.suspended=suspended
                selections=[({'weight':strength},root/'style.safetensors')]
                with patch('backend.app.worker.verify_inputs',return_value=(root/'bundle','checked',{},selections)),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.memory_reserved',return_value=0),patch('backend.app.memory_policy.available_vram',return_value=(16*gb,16*gb,{})),patch('backend.app.memory_policy.available_ram',return_value=32*gb),patch('backend.app.flux2.resume_pipeline') as resume,patch('backend.app.flux2.load_pipeline') as load:
                    with self.assertRaises(ReachedPipeline):_run(path,cache)
                pipe.set_adapters.assert_called_once_with(['lora_0'],adapter_weights=[strength])
                self.assertEqual(cache.get.call_args.kwargs['replace_loras'],selections)
                load.assert_not_called()
                if suspended:resume.assert_called_once_with(pipe,768,1024)
                else:resume.assert_not_called()
                self.assertFalse(cache.suspended)

    def test_klein_and_zimage_reclaim_released_external_vram(self):
        from backend.app.worker import _run
        gb=1024**3
        for family,module in [('flux2','flux2'),('zimage','z_image')]:
            with self.subTest(family=family),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                path=root/'request.json'
                path.write_text(json.dumps(dict(plan=dict(runner=family+'_txt2img',settings=dict(width=768,height=1024)),positive='same')))
                pipe=SimpleNamespace(memory_policy={'free_vram_bytes':8*gb},set_adapters=Mock(side_effect=ReachedPipeline))
                cache=Mock();cache.get.return_value=pipe;cache.suspended=False
                def suspend():cache.suspended=True;return True
                cache.suspend.side_effect=suspend
                selections=[({'weight':.5},root/'style.safetensors')]
                with patch('backend.app.worker.verify_inputs',return_value=(root/'bundle','checked',{},selections)),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.memory_reserved',return_value=0),patch('backend.app.memory_policy.available_vram',return_value=(16*gb,16*gb,{})),patch('backend.app.memory_policy.available_ram',return_value=32*gb),patch('backend.app.'+module+'.resume_pipeline') as resume,patch('backend.app.'+module+'.load_pipeline') as load:
                    with self.assertRaises(ReachedPipeline):_run(path,cache)
                cache.suspend.assert_called_once();resume.assert_called_once_with(pipe,768,1024)
                load.assert_not_called();self.assertFalse(cache.suspended)

    def test_pressure_and_released_vram_replan_but_low_ram_reloads(self):
        from backend.app.worker import _run
        gb=1024**3
        cases=[(16*gb,True,True,16*gb,None), (7*gb,True,False,16*gb,None),
               (16*gb,False,False,16*gb,None), (16*gb,True,True,8*gb,None),
               (16*gb,True,None,15*gb,None), (16*gb,True,None,15*gb-256*1024**2,None),
               (16*gb,True,None,8*gb,15*gb),
               (16*gb,True,None,15*gb-512*1024**2,None),
               (16*gb,True,None,14*gb,None),
               (16*gb,True,True,14*gb-1,None)]
        for ram,retained,should_resume,previous_free,observed in cases:
            with self.subTest(ram=ram,retained=retained,previous_free=previous_free),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);config=root/'configs/krea2';config.mkdir(parents=True)
                (config/'manifest.json').write_bytes(b'{}')
                plan=dict(runner='krea2_txt2img',settings=dict(width=768,height=1024),krea2={'sha256':'checked'},config_sha256=hashlib.sha256(b'{}').hexdigest())
                path=root/'request.json';path.write_text(json.dumps(dict(plan=plan,positive='same')))
                pipe=SimpleNamespace(memory_policy={'free_vram_bytes':previous_free},weight_maps=[],vae=Mock())
                if observed is not None:pipe.memory_policy['free_vram_with_allocator_bytes']=observed
                pipe.vae.enable_tiling.side_effect=ReachedPipeline
                cache=Mock();cache.get.return_value=pipe;cache.suspended=False
                def suspend():cache.suspended=retained;return retained
                cache.suspend.side_effect=suspend
                with patch('backend.app.worker.MODEL_ROOT',root),patch('backend.app.worker.verify_inputs',return_value=(root/'weight','checked',{'clip':root/'clip','vae':root/'vae'},[])),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.empty_cache'),patch('torch.cuda.memory_reserved',return_value=0),patch('backend.app.memory_policy.available_vram',return_value=(15*gb,16*gb,{})),patch('backend.app.memory_policy.available_ram',return_value=ram),patch('backend.app.krea2.resume_pipeline') as resume,patch('backend.app.krea2.load_pipeline',side_effect=ReachedPipeline) as load:
                    with self.assertRaises(ReachedPipeline):_run(path,cache)
                if should_resume:
                    resume.assert_called_once_with(pipe,(768,1024));load.assert_not_called()
                    self.assertTrue(pipe.memory_policy['pipeline_reused']);self.assertFalse(cache.suspended)
                elif should_resume is None:
                    resume.assert_not_called();load.assert_not_called();cache.suspend.assert_not_called()
                    self.assertEqual(pipe.memory_policy['free_vram_with_allocator_bytes'],observed if observed is not None else previous_free)
                else:
                    resume.assert_not_called();load.assert_called_once()
                if ram<8*gb:cache.clear.assert_called_once();cache.suspend.assert_not_called()
