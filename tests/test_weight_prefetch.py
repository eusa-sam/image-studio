import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from backend.app.weight_prefetch import prefetch_weights

class WeightPrefetchTests(unittest.TestCase):
    def make_pipe(self,folder):
        model=torch.nn.Sequential(torch.nn.Linear(16,16,bias=False),torch.nn.Linear(16,16,bias=False))
        for module in model:module._old_forward=module.forward
        class Mapping:
            def _stamp(self):return self.path.stat().st_mtime_ns
            def linear_packed(self,name,device):
                if self._stamp()!=self.stamp:raise ValueError('COMPONENT_CHANGED')
                return self.native_cpu_cache.get(name)
        mapping=Mapping();mapping.path=Path(folder)/'weights.safetensors'
        tensors={}
        for index in range(2):
            tensors[f'{index}.weight']=torch.full((16,16),index+1,dtype=torch.int8)
            tensors[f'{index}.scale']=torch.ones(16)
        save_file(tensors,str(mapping.path));mapping.stamp=mapping._stamp()
        mapping.sources={f'{i}.weight':(f'{i}.weight',torch.bfloat16) for i in range(2)}
        mapping.packed_sources={f'{i}.weight':(256,f'{i}.scale',16,16) for i in range(2)}
        mapping.native_cpu_cache={};mapping.packed_cache={};mapping.cpu_budget=1024;mapping.cpu_bytes=0
        return SimpleNamespace(transformer=model,weight_maps=[mapping]),tensors

    @patch('backend.app.weight_prefetch.available_ram',return_value=16*1024**3)
    def test_exact_compressed_cache_and_warm_call_without_executor(self,ram):
        with tempfile.TemporaryDirectory() as folder:
            pipe,tensors=self.make_pipe(folder);mapping=pipe.weight_maps[0]
            with prefetch_weights(pipe):
                for i in range(2):
                    raw,scale=mapping.linear_packed(f'{i}.weight','cpu')
                    self.assertTrue(torch.equal(raw,tensors[f'{i}.weight']))
                    self.assertTrue(torch.equal(scale,tensors[f'{i}.scale']))
            self.assertEqual(mapping.cpu_bytes,640)
            self.assertNotIn('linear_packed',vars(mapping))
            with patch('backend.app.weight_prefetch.ThreadPoolExecutor',side_effect=AssertionError('warm read')):
                with prefetch_weights(pipe):pass

    def test_ram_budget_and_cpu_cache_budget_do_not_preload(self):
        for ram,budget in [(7*1024**3,1024),(16*1024**3,0)]:
            with self.subTest(ram=ram,budget=budget),tempfile.TemporaryDirectory() as folder:
                pipe,_=self.make_pipe(folder);mapping=pipe.weight_maps[0];mapping.cpu_budget=budget
                with patch('backend.app.weight_prefetch.available_ram',return_value=ram),patch('backend.app.weight_prefetch.safe_open',side_effect=AssertionError('read')):
                    with prefetch_weights(pipe):self.assertIsNone(mapping.linear_packed('0.weight','cpu'))
                self.assertEqual(mapping.cpu_bytes,0)

    @patch('backend.app.weight_prefetch.available_ram',return_value=16*1024**3)
    def test_changed_source_and_body_exception_restore_override(self,ram):
        with tempfile.TemporaryDirectory() as folder:
            pipe,_=self.make_pipe(folder);mapping=pipe.weight_maps[0]
            original=mapping.linear_packed;mapping.linear_packed=original
            with patch.object(mapping,'_stamp',return_value=-1),self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):
                with prefetch_weights(pipe):mapping.linear_packed('0.weight','cpu')
            self.assertIs(mapping.linear_packed,original)
            self.assertFalse(mapping.native_cpu_cache)
            with self.assertRaisesRegex(RuntimeError,'body'):
                with prefetch_weights(pipe):raise RuntimeError('body')
            self.assertIs(mapping.linear_packed,original)
            self.assertEqual(mapping.cpu_bytes,0)

    def test_failed_scheduling_restores_override(self):
        with tempfile.TemporaryDirectory() as folder:
            pipe,_=self.make_pipe(folder);mapping=pipe.weight_maps[0]
            with patch('backend.app.weight_prefetch.available_ram',side_effect=RuntimeError('memory probe')),self.assertRaisesRegex(RuntimeError,'memory probe'):
                with prefetch_weights(pipe):pass
            self.assertNotIn('linear_packed',vars(mapping))

    def test_ram_drop_discards_prefetch_before_cache_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            pipe,_=self.make_pipe(folder);mapping=pipe.weight_maps[0]
            with patch('backend.app.weight_prefetch.available_ram',side_effect=[16*1024**3,7*1024**3,7*1024**3]):
                with prefetch_weights(pipe):self.assertIsNone(mapping.linear_packed('0.weight','cpu'))
            self.assertFalse(mapping.native_cpu_cache)
            self.assertEqual(mapping.cpu_bytes,0)

    @patch('backend.app.weight_prefetch.available_ram',return_value=16*1024**3)
    def test_change_during_read_is_not_published(self,ram):
        with tempfile.TemporaryDirectory() as folder:
            pipe,_=self.make_pipe(folder);mapping=pipe.weight_maps[0]
            with patch.object(mapping,'_stamp',side_effect=[mapping.stamp,-1]),self.assertRaisesRegex(ValueError,'during prefetch'):
                with prefetch_weights(pipe):mapping.linear_packed('0.weight','cpu')
            self.assertFalse(mapping.native_cpu_cache)
            self.assertNotIn('linear_packed',vars(mapping))
