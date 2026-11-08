import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from backend.app.inference.quantization.staging import Staging, staging_weights
from backend.app.inference.quantization.quantized_weights import FileWeights
from backend.app.inference.runtime.execution import pipeline_phases


class StagingTests(unittest.TestCase):
    def test_reserved_slot_remains_usable_while_activations_use_reserve(self):
        transfer=Staging.__new__(Staging)
        transfer.mapping=SimpleNamespace(packed_cache={},packed_sources={'weight':(64,'scale',256,1)})
        transfer.capacity=1024;transfer.pending={};transfer.last=None
        transfer.slots=[SimpleNamespace(name=None)];transfer.executor=Mock()
        transfer.pipe=SimpleNamespace(memory_policy={'activation_reserve_bytes':4*1024**3})
        with patch('backend.app.inference.quantization.staging.allocator_free_bytes',return_value=0,create=True),patch('backend.app.inference.quantization.staging.available_ram',return_value=16*1024**3):
            transfer.schedule('weight')
        transfer.executor.submit.assert_called_once_with(transfer.prepare,transfer.slots[0],'weight')
        self.assertIn('weight',transfer.pending)

    def test_budget_rejection_precedes_buffer_allocation(self):
        with self.assertRaises(ValueError):Staging(None,None,128*1024**2+1)
        with patch('backend.app.inference.quantization.staging.cache_allocation_fits',return_value=False),patch('torch.empty') as allocate:
            with self.assertRaises(MemoryError):Staging(None,None,1024)
        allocate.assert_not_called()

    def test_budget_fallback_removes_previous_async_diagnostics(self):
        mapping=SimpleNamespace(packed_sources={'weight':(4096,'scale',256,64)})
        pipe=SimpleNamespace(weight_maps=[mapping],memory_policy={},_staging_statistics={'hits':9})
        with patch('backend.app.inference.quantization.staging.Staging',side_effect=MemoryError('budget')):
            with staging_weights(pipe):self.assertNotIn('_weight_transfer',vars(pipe))
        self.assertIsNone(pipe._staging_statistics)
        self.assertEqual(pipe.memory_policy['tensor_transfer'],'blocking')
        self.assertEqual(pipe.memory_policy['weight_prefetch'],'synchronous')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA staging events')
    def test_two_slots_preserve_order_and_finish_before_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'packed.safetensors';state={}
            for i in range(4):
                state[f'{i}.weight']=torch.full((64,64),i+1).to(torch.float8_e4m3fn)
                state[f'{i}.weight_scale']=torch.tensor(.125)
                state[f'{i}.comfy_quant']=torch.tensor(list(json.dumps({'format':'float8_e4m3fn'}).encode()),dtype=torch.uint8)
            save_file(state,path)
            model=torch.nn.ModuleList([torch.nn.Linear(64,64,bias=False) for _ in range(4)])
            mapping=FileWeights(model,path,lambda x:x);mapping.cpu_budget=1024**2
            pipe=SimpleNamespace(weight_maps=[mapping],memory_policy=dict(cpu_cache_budget_bytes=1024**2,
                                 resident_budget_bytes=1024**2,activation_reserve_bytes=1))
            results=[]
            with pipeline_phases(pipe),staging_weights(pipe),patch('backend.app.inference.quantization.staging.allocator_free_bytes',return_value=0,create=True):
                transfer=pipe._weight_transfer
                self.assertEqual(pipe.memory_policy['weight_prefetch'],'bounded_two_slot_staging')
                self.assertEqual(len(transfer.cpu_tensors),2)
                pointers=[v.data_ptr() for v in transfer.cpu_tensors]
                for _ in range(12):
                    for i in range(4):
                        raw,scale,_=mapping.linear_packed(f'{i}.weight',torch.device('cuda:0'))
                        results.append((raw.float()*scale).sum())
                self.assertEqual(pointers,[v.data_ptr() for v in transfer.cpu_tensors])
            expected=torch.tensor([(i+1)*512 for _ in range(12) for i in range(4)],device='cuda')
            torch.testing.assert_close(torch.stack(results),expected.float(),rtol=0,atol=0)
            self.assertGreater(transfer.hits,0)
            self.assertEqual(transfer.hits+transfer.resident_hits+transfer.synchronous_transfers,48)
            self.assertEqual(transfer.fallbacks,transfer.resident_hits+transfer.synchronous_transfers)
            self.assertTrue(transfer.closed);self.assertFalse(transfer.cpu_tensors);self.assertFalse(transfer.cuda_tensors)
            self.assertNotIn('_weight_transfer',vars(pipe))
            self.assertNotIn('linear_packed',vars(mapping))
            with self.assertRaisesRegex(RuntimeError,'cancel'):
                with pipeline_phases(pipe),staging_weights(pipe):
                    mapping.linear_packed('0.weight',torch.device('cuda:0'))
                    raise RuntimeError('cancel')
            self.assertNotIn('_weight_transfer',vars(pipe))
            with pipeline_phases(pipe),staging_weights(pipe):
                raw,scale,_=mapping.linear_packed('1.weight',torch.device('cuda:0'))
                self.assertEqual(float((raw.float()*scale).sum()),1024.)
