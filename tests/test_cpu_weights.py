import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from backend.app.inference.runtime.cpu_weights import remember, restore, forget, backed_tensors


class CpuWeightsTests(unittest.TestCase):
    def pipe(self):
        module=torch.nn.Linear(3,3)
        module.register_buffer('scale',torch.ones(3))
        return SimpleNamespace(transformer=module,vae=torch.nn.Linear(3,3))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_eviction_reattaches_cpu_storage_without_copying_weights_back(self):
        pipe=self.pipe();original=pipe.transformer.weight.detach();pointer=original.data_ptr()
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        pipe.transformer.to('cuda');pipe.vae.to('cuda')
        self.assertTrue(pipe.transformer.weight.is_cuda)
        self.assertTrue(restore(pipe))
        self.assertFalse(pipe.transformer.weight.is_cuda)
        self.assertEqual(pipe.transformer.weight.data_ptr(),pointer)
        torch.testing.assert_close(pipe.transformer.weight,original,rtol=0,atol=0)
        forget(pipe)

    def test_reuses_original_storage_and_preserves_parameter_identity(self):
        pipe=self.pipe();parameter=pipe.transformer.weight;original=parameter.detach()
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        self.assertIn(id(parameter),backed_tensors(pipe))
        parameter.data=parameter.clone()
        self.assertTrue(restore(pipe))
        self.assertIs(pipe.transformer.weight,parameter)
        self.assertEqual(parameter.data_ptr(),original.data_ptr())
        forget(pipe);self.assertFalse(restore(pipe))

    def test_headroom_and_invalidated_shape_never_restore_stale_weights(self):
        pipe=self.pipe()
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=8*1024**3):remember(pipe)
        self.assertFalse(restore(pipe))
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        pipe.transformer.weight=torch.nn.Parameter(torch.zeros(1))
        self.assertFalse(restore(pipe));self.assertFalse(hasattr(pipe,'_cpu_weights'))

    def test_buffers_and_tied_storage_survive_eviction(self):
        pipe=self.pipe();pipe.vae.weight=pipe.transformer.weight
        original=pipe.transformer.scale
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        pipe.transformer.scale=original.clone();pipe.transformer.weight.data=pipe.transformer.weight.clone()
        restore(pipe)
        self.assertEqual(pipe.transformer.scale.data_ptr(),original.data_ptr())
        self.assertIs(pipe.transformer.weight,pipe.vae.weight)

    def test_encoder_original_is_reused_only_within_remaining_budget(self):
        pipe=self.pipe();pipe.text_encoder=torch.nn.Linear(3,3)
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=8*1024**3+150):remember(pipe)
        self.assertIn(id(pipe.text_encoder.weight),backed_tensors(pipe))
        self.assertNotIn(id(pipe.text_encoder.bias),backed_tensors(pipe))
        self.assertIn(id(pipe.transformer.weight),backed_tensors(pipe))
        forget(pipe)
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        self.assertIn(id(pipe.text_encoder.weight),backed_tensors(pipe))

    def test_lora_strength_change_and_release_invalidate_backups(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        pipe=self.pipe();pipe.memory_policy={'family':'flux2'}
        cache=PipelineCache();cache.pipe=pipe;cache.key='same';cache.positive='prompt';cache.lora_weights=(1.,)
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        cache.conditioning={'prompt':{'prompt_embeds':torch.ones(1)}}
        cache.get('same','prompt',replace_loras=[({'weight':1.},'a')]);self.assertTrue(hasattr(pipe,'_cpu_weights'))
        with patch.object(cache,'suspend',return_value=True) as move:
            cache.get('same','prompt',replace_loras=[({'weight':.5},'a')])
        move.assert_called_once();self.assertFalse(hasattr(pipe,'_cpu_weights'));self.assertIn('prompt',cache.conditioning)
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        cache.clear();self.assertFalse(hasattr(pipe,'_cpu_weights'));self.assertIsNone(cache.pipe)

    def test_same_strength_adapter_swap_invalidates_conditioning(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        from unittest.mock import Mock
        plan=dict(runner='anima_txt2img',loras=[dict(path='a',weight=.5)])
        cache=PipelineCache();cache.key=cache.identity(plan,(512,512));cache.positive='same';cache.lora_weights=(.5,)
        cache.pipe=SimpleNamespace(memory_policy={'family':'anima'},delete_adapters=Mock())
        cache.conditioning={'same':{'prompt_embeds':torch.ones(1)}}
        requested=cache.identity(plan|{'loras':[dict(path='b',weight=.5)]},(512,512))
        with patch.object(cache,'suspend',return_value=True),patch('backend.app.inference.conditioning.lora.apply_loras'):
            cache.get(requested,'same',replace_loras=[({'weight':.5},'b')])
        self.assertFalse(cache.conditioning)

    def test_strength_invalidation_restores_cpu_original_before_discarding_backup(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        pipe=self.pipe();pipe.memory_policy={'family':'flux2'}
        cache=PipelineCache();cache.pipe=pipe;cache.key='same';cache.positive='prompt';cache.lora_weights=(1.,)
        original=pipe.transformer.weight.detach();pointer=original.data_ptr()
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        pipe.transformer.weight.data=pipe.transformer.weight.clone()
        def suspend():
            self.assertTrue(restore(pipe));cache.suspended=True;return True
        with patch.object(cache,'suspend',side_effect=suspend) as move:
            self.assertIs(cache.get('same','prompt',replace_loras=[({'weight':.5},'a')]),pipe)
        move.assert_called_once()
        self.assertEqual(pipe.transformer.weight.data_ptr(),pointer)
        self.assertFalse(hasattr(pipe,'_cpu_weights'))

    def test_adapter_removal_reattaches_originals_before_deleting_adapter(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        from unittest.mock import Mock
        pipe=self.pipe();pipe.memory_policy={'family':'flux2'}
        cache=PipelineCache();cache.pipe=pipe;cache.positive='prompt';cache.lora_weights=(1.,)
        plan=dict(runner='flux2_txt2img',loras=[dict(path='a',weight=1.)])
        cache.key=cache.identity(plan,(512,512))
        original=pipe.transformer.weight.detach();pointer=original.data_ptr()
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        pipe.transformer.weight.data=pipe.transformer.weight.clone()
        def suspend():
            self.assertTrue(restore(pipe));cache.suspended=True;return True
        def delete(names):
            self.assertTrue(cache.suspended)
            self.assertFalse(hasattr(pipe,'_cpu_weights'))
            self.assertEqual(pipe.transformer.weight.data_ptr(),pointer)
        pipe.delete_adapters=Mock(side_effect=delete)
        with patch.object(cache,'suspend',side_effect=suspend) as move:
            self.assertIs(cache.get(cache.identity(plan|{'loras':[]},(512,512)),'prompt',replace_loras=[]),pipe)
        move.assert_called_once();pipe.delete_adapters.assert_called_once_with(['lora_0'])

    def test_transformer_only_adapter_swap_keeps_text_conditioning(self):
        from backend.app.inference.runtime.pipeline_cache import PipelineCache
        from unittest.mock import Mock
        for family in ('flux2','zimage'):
            plan=dict(runner=family+'_txt2img',loras=[dict(path='a',weight=.5)])
            cache=PipelineCache();cache.key=cache.identity(plan,(512,512));cache.positive='prompt';cache.lora_weights=(.5,)
            cache.pipe=SimpleNamespace(memory_policy={'family':family},delete_adapters=Mock())
            embedding=torch.ones(1);cache.conditioning={'prompt':{'prompt_embeds':embedding}}
            with patch.object(cache,'suspend',return_value=True),patch('backend.app.inference.conditioning.lora.apply_loras'):
                cache.get(cache.identity(plan|{'loras':[dict(path='b',weight=.5)]},(512,512)),'prompt',replace_loras=[({'weight':.5},'b')])
            self.assertIs(cache.conditioning['prompt']['prompt_embeds'],embedding)
