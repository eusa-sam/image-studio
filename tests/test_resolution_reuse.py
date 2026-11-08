import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from backend.app.inference.runtime.pipeline_cache import PipelineCache


class ResolutionReuseTests(unittest.TestCase):
    def test_resolution_change_retains_weights_only_after_cpu_suspend(self):
        import torch
        for family in ('sd1','sd2','sdxl','flux2','zimage','anima'):
            for suspended in (False,True):
                with self.subTest(family=family,suspended=suspended):
                    model=torch.nn.Linear(2,2);before=model.weight.detach().clone()
                    pipe=SimpleNamespace(memory_policy={'family':family},components={'model':model},to=Mock(),remove_all_hooks=Mock(),anima_manager=SimpleNamespace(disable_auto_cpu_offload=Mock()))
                    plan={'runner':family+'_txt2img','model_path':'same','fingerprint':{'bytes':1}}
                    cache=PipelineCache();cache.pipe=pipe;cache.key=cache.identity(plan,(768,1024));cache.suspended=suspended
                    wanted=cache.identity(plan,(1024,1280))
                    with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3),patch('torch.cuda.empty_cache'):
                        self.assertIs(cache.get(wanted,'same'),pipe)
                    self.assertTrue(cache.suspended);self.assertEqual(cache.key,wanted)
                    self.assertTrue(torch.equal(model.weight,before))
                    if suspended:pipe.to.assert_not_called()
                    else:pipe.to.assert_called_once_with('cpu')

    def test_low_ram_and_changed_base_or_components_do_not_reuse(self):
        for change in ('ram','model_path','components','fingerprint'):
            cache=PipelineCache();cache.pipe=SimpleNamespace(memory_policy={'family':'sdxl'},components={})
            plan={'runner':'sdxl_txt2img','model_path':'same','fingerprint':{'bytes':1}}
            cache.key=cache.identity(plan,(768,1024))
            if change!='ram':plan[change]={'changed':True} if change!='model_path' else 'different'
            with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=1),patch('torch.cuda.empty_cache'):
                self.assertIsNone(cache.get(cache.identity(plan,(1024,1280)),'same'))
            self.assertIsNone(cache.pipe)

    def test_krea_resolution_still_invalidates_its_different_weight_cache(self):
        cache=PipelineCache();cache.pipe=SimpleNamespace(memory_policy={'family':'krea2','mode':'partial'})
        plan={'runner':'krea2_txt2img','model_path':'same'}
        cache.key=cache.identity(plan,(768,1024))
        self.assertIsNone(cache.get(cache.identity(plan,(1024,1024)),'same'))
