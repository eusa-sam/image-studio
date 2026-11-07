import unittest
from types import SimpleNamespace
from unittest.mock import patch
from backend.app.memory_policy import needs_vram_rebalance

class RebalanceThresholdTests(unittest.TestCase):
    def test_pressure_is_prompt_while_small_recoveries_accumulate(self):
        gb=1024**3;reference=10*gb
        policy=dict(mode='partial',free_vram_bytes=reference-2*gb,free_vram_with_allocator_bytes=reference)
        for difference,expected in [(-gb,True),(-256*1024**2-1,True),(-256*1024**2,False),(0,False),(512*1024**2,False),(gb,False),(gb+1,True),(2*gb,True)]:
            with self.subTest(difference=difference):self.assertEqual(needs_vram_rebalance(policy,reference+difference),expected)
        self.assertEqual(policy['free_vram_with_allocator_bytes'],reference)
        policy['mode']='full'
        self.assertFalse(needs_vram_rebalance(policy,reference+2*gb))
        self.assertTrue(needs_vram_rebalance(policy,reference-gb))

    def test_krea_resume_resets_anchor_after_actual_replacement(self):
        from backend.app.krea2 import resume_pipeline
        gb=1024**3
        pipe=SimpleNamespace(memory_policy=dict(weight_bytes=20*gb,free_vram_with_allocator_bytes=10*gb),
                             weight_maps=[SimpleNamespace(cpu_bytes=gb)])
        with patch('torch.cuda.empty_cache'),patch('backend.app.memory_policy.available_vram',return_value=(12*gb,16*gb,{})),patch('backend.app.memory_policy.available_ram',return_value=32*gb),patch('backend.app.krea2.partial_cache_budgets') as budgets:
            resume_pipeline(pipe,(768,1024))
        budgets.assert_called_once()
        self.assertNotIn('free_vram_with_allocator_bytes',pipe.memory_policy)
        self.assertFalse(needs_vram_rebalance(pipe.memory_policy,12*gb))
        self.assertEqual(pipe.memory_policy['free_vram_with_allocator_bytes'],12*gb)
