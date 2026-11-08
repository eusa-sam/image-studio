import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from backend.app.inference.runtime.execution import enter_phase, pipeline_phases, release
from backend.app.inference.runtime.memory_policy import cached_storage_bytes, cache_allocation_fits


class ExecutionTests(unittest.TestCase):
    def test_generation_uses_shared_budget_and_restores_previous_binding_after_failure(self):
        previous=Mock();first=SimpleNamespace(shared_budget=previous);second=SimpleNamespace()
        pipe=SimpleNamespace(weight_maps=[first,second])
        with patch('backend.app.inference.runtime.memory_policy.cache_budget_fits',return_value=False) as check:
            with self.assertRaisesRegex(ValueError,'cancel'):
                with pipeline_phases(pipe):
                    self.assertFalse(first.shared_budget(cpu=7))
                    self.assertFalse(second.shared_budget(gpu=3,temporary=5))
                    raise ValueError('cancel')
            self.assertEqual(check.call_count,2)
            check.assert_called_with(pipe,gpu=3,temporary=5)
        self.assertIs(first.shared_budget,previous)
        self.assertNotIn('shared_budget',vars(second))

    def test_resumed_request_returns_to_idle_after_generation(self):
        events=[]
        pipe=SimpleNamespace(_inference_phase='suspend',_phase_handler=events.append,
                             transformer=torch.nn.Identity())
        enter_phase(pipe,'idle')
        enter_phase(pipe,'idle')
        with pipeline_phases(pipe):pipe.transformer(torch.ones(1))
        self.assertEqual(events,['idle','condition','denoise','idle'])
        self.assertEqual(pipe._inference_phase,'idle')

    def test_phase_order_and_cleanup_after_failure(self):
        class VAE:
            def decode(self, value): return value + 1
        events=[]
        pipe=SimpleNamespace(text_encoder=torch.nn.Identity(),transformer=torch.nn.Identity(),
                             vae=VAE(),_phase_handler=events.append,_weight_transfer=Mock())
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            with pipeline_phases(pipe):
                pipe.text_encoder(torch.ones(1))
                pipe.transformer(torch.ones(1))
                self.assertEqual(pipe.vae.decode(1),2)
                raise RuntimeError('cancel')
        self.assertEqual(events,['condition','denoise','decode','denoise','idle'])
        self.assertNotIn('decode',vars(pipe.vae))
        self.assertFalse(pipe.transformer._forward_pre_hooks)
        pipe._weight_transfer.close.assert_called_once()

    def test_release_finishes_transfers_before_discarding_caches(self):
        events=[]
        pipe=SimpleNamespace(_weight_transfer=SimpleNamespace(close=lambda:events.append('wait')),
            weight_maps=[SimpleNamespace(clear_cache=lambda:events.append('clear'))],_cpu_weights=[])
        release(pipe)
        self.assertEqual(events,['wait','clear'])
        self.assertNotIn('_cpu_weights',vars(pipe))

    def test_failed_transfer_still_drops_completed_transfer_owners(self):
        mapping=SimpleNamespace(shared_budget=lambda:True,clear_cache=Mock())
        pipe=SimpleNamespace(_weight_transfer=SimpleNamespace(close=Mock(side_effect=ValueError('read failed'))),
                             weight_maps=[mapping],_phase_handler=Mock(),_cpu_weights=[])
        with self.assertRaisesRegex(ValueError,'read failed'):release(pipe)
        mapping.clear_cache.assert_called_once()
        self.assertNotIn('shared_budget',vars(mapping));self.assertNotIn('_phase_handler',vars(pipe))

    def test_budget_counts_shared_storage_once_and_protects_ram(self):
        original=torch.zeros(16)
        derived=torch.zeros(8)
        pipe=SimpleNamespace(_cpu_weights=[(None,None,None,original)],
            weight_maps=[SimpleNamespace(cache={'view':original[:4]},derived_cpu_cache={'layer':(derived,original)})],
            _weight_transfer=SimpleNamespace(cpu_tensors=[original]),
            memory_policy=dict(cpu_cache_budget_bytes=128,resident_budget_bytes=128,activation_reserve_bytes=256))
        self.assertEqual(cached_storage_bytes(pipe,'cpu'),96)
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=16*1024**3):
            self.assertTrue(cache_allocation_fits(pipe,cpu=32))
            self.assertFalse(cache_allocation_fits(pipe,cpu=33))
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=8*1024**3):
            self.assertFalse(cache_allocation_fits(pipe,cpu=1))
        with patch('torch.cuda.mem_get_info',return_value=(320,1024)),patch('torch.cuda.memory_reserved',return_value=64),patch('torch.cuda.memory_allocated',return_value=32):
            self.assertTrue(cache_allocation_fits(pipe,gpu=64,temporary=32))
            self.assertFalse(cache_allocation_fits(pipe,gpu=64,temporary=33))
        with patch('torch.cuda.mem_get_info',return_value=(4096,8192)),patch('torch.cuda.memory_reserved',return_value=0),patch('torch.cuda.memory_allocated',return_value=0):
            self.assertFalse(cache_allocation_fits(pipe,gpu=64,temporary=65))
        with self.assertRaises(ValueError):cache_allocation_fits(pipe,cpu=-1)
