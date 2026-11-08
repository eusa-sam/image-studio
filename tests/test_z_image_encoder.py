import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch
from backend.app.inference.models import z_image


class EncoderLifetimeTests(unittest.TestCase):
    def pipe(self):
        return SimpleNamespace(_encoder_source=Path('local/text_encoder'),_generation_size=(768,1024),
            _execution_device=torch.device('cpu'),text_encoder=torch.nn.Linear(2,2),transformer=torch.nn.Linear(2,2),
            memory_policy={'mode':'sequential_offload','family':'zimage','pipeline_reused':True},
            remove_all_hooks=Mock(),to=Mock(),encode_prompt=Mock(return_value=([torch.ones(2,3)],[])))

    def test_pressure_releases_encoder_cache_hit_skips_load_and_changed_prompt_restores(self):
        pipe=self.pipe();cache={}
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=7*1024**3),patch('torch.cuda.empty_cache'),patch.object(z_image,'reconfigure_after_encoder_change') as replan,patch('transformers.Qwen3Model.from_pretrained',side_effect=lambda *a,**k:torch.nn.Linear(2,2)) as load:
            first=z_image.prompt_arguments(pipe,'one',cache)
            self.assertIsNone(pipe.text_encoder)
            again=z_image.prompt_arguments(pipe,'one',cache)
            self.assertIs(first['prompt_embeds'][0],again['prompt_embeds'][0])
            load.assert_not_called();self.assertEqual(pipe.encode_prompt.call_count,1)
            z_image.prompt_arguments(pipe,'two',cache)
            load.assert_called_once_with(str(pipe._encoder_source),torch_dtype=torch.bfloat16,local_files_only=True)
            self.assertEqual(set(cache),{'two'});self.assertIsNone(pipe.text_encoder)
            self.assertEqual(replan.call_count,3)
            self.assertTrue(pipe.memory_policy['text_encoder_released'])

    def test_headroom_keeps_encoder_and_failed_restore_preserves_cached_embedding(self):
        pipe=self.pipe();cache={}
        pipe.memory_policy['mode']='model_offload'
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
            z_image.prompt_arguments(pipe,'one',cache)
        self.assertIsNotNone(pipe.text_encoder);pipe.remove_all_hooks.assert_not_called()
        pipe.text_encoder=None;pipe.memory_policy['mode']='full'
        events=[]
        pipe.remove_all_hooks.side_effect=lambda:events.append('unhook')
        pipe.to.side_effect=lambda device:events.append(device)
        def fail(*args,**kwargs):
            self.assertEqual(events,['unhook','cpu']);raise OSError('unreadable encoder')
        with patch('torch.cuda.empty_cache'),patch('transformers.Qwen3Model.from_pretrained',side_effect=fail):
            with self.assertRaisesRegex(OSError,'unreadable'):z_image.prompt_arguments(pipe,'two',cache)
        self.assertEqual(set(cache),{'one'});self.assertIsNone(pipe.text_encoder)

    def test_sequential_offload_releases_idle_encoder_even_with_ram_headroom(self):
        pipe=self.pipe();cache={};events=[]
        import weakref
        from backend.app.inference.runtime.cpu_weights import remember
        pipe.vae=torch.nn.Linear(2,2)
        encoder=weakref.ref(pipe.text_encoder)
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):remember(pipe)
        pipe.encode_prompt.side_effect=lambda **kwargs:(events.append('encode') or ([torch.ones(2,3)],[]))
        pipe.remove_all_hooks.side_effect=lambda:events.append('unhook')
        pipe.to.side_effect=lambda device:events.append(device)
        def replan(current):
            self.assertIsNone(current.text_encoder)
            events.append('replan');current.memory_policy['mode']='block_offload'
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3),patch('torch.cuda.empty_cache'),patch.object(z_image,'reconfigure_after_encoder_change',side_effect=replan):
            first=z_image.prompt_arguments(pipe,'one',cache)
            self.assertIsNone(pipe.text_encoder)
            again=z_image.prompt_arguments(pipe,'one',cache)
        self.assertEqual(events,['encode','unhook','cpu','replan'])
        self.assertIs(first['prompt_embeds'][0],again['prompt_embeds'][0])
        self.assertTrue(pipe.memory_policy['text_encoder_released'])
        self.assertIsNone(encoder())
        self.assertFalse(hasattr(pipe,'_cpu_weights'))

    def test_reconfiguration_preserves_reuse_and_updates_allocator_baseline(self):
        pipe=self.pipe()
        with patch('backend.app.inference.runtime.memory_policy.configure_pipeline',return_value={'mode':'model_offload','weight_bytes':123}) as configure,patch('backend.app.inference.runtime.memory_policy.available_vram',return_value=(100,200,{})),patch('torch.cuda.memory_reserved',return_value=30):
            z_image.reconfigure_after_encoder_change(pipe)
        self.assertTrue(pipe.memory_policy['pipeline_reused'])
        self.assertEqual(pipe.memory_policy['free_vram_with_allocator_bytes'],130)
        self.assertEqual(pipe.memory_policy['mode'],'model_offload')
        configure.assert_called_once_with(pipe,768,1024,activation_reserve_bytes=z_image.activation_reserve(768,1024))
