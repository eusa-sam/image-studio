import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from backend.app.memory_policy import residency_plan,available_vram
from backend.app.quantized_weights import FileWeights,load_local_weights


class MemoryPolicyTests(unittest.TestCase):
    def test_validation_device_keeps_int8_on_cpu_and_requires_workspace_headroom(self):
        from backend.app.memory_policy import weight_validation_device
        elements=100_000_000
        required=elements*20+2*1024**3
        self.assertEqual(weight_validation_device(required,elements,False),'cuda:0')
        self.assertEqual(weight_validation_device(required-1,elements,False),'cpu')
        self.assertEqual(weight_validation_device(16*1024**3,elements,True),'cpu')
        self.assertEqual(weight_validation_device(required,elements,True,allow_int8=True),'cuda:0')
        self.assertEqual(weight_validation_device(required-1,elements,True,allow_int8=True),'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_validation_preload_remains_host_owned_and_values_match(self):
        from unittest.mock import patch
        import json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fp8.safetensors'
            quantized=torch.arange(16.).reshape(4,4).to(torch.float8_e4m3fn)
            save_file({'layer.weight':quantized,'layer.weight_scale':torch.tensor(.25),
                'layer.comfy_quant':torch.tensor(list(json.dumps({'format':'float8_e4m3fn'}).encode()),dtype=torch.uint8)},path)
            def create():
                with torch.device('meta'):return torch.nn.ModuleDict({'layer':torch.nn.Linear(4,4,bias=False)})
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3):
                cpu=FileWeights(create(),str(path),lambda state:state,preload_budget=32)
                gpu=FileWeights(create(),str(path),lambda state:state,preload_budget=32,validation_device='cuda:0')
            self.assertEqual(gpu.cpu_bytes,32)
            self.assertEqual(gpu.cache['layer.weight'].device.type,'cpu')
            torch.testing.assert_close(gpu['layer.weight'],cpu['layer.weight'],rtol=0,atol=0)
            self.assertEqual(gpu.cache_bytes,0)

    def test_preload_policy_only_spends_ram_on_int8_restore(self):
        from backend.app.memory_policy import int8_preload_budget
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            for dtype,expected in ((torch.float8_e4m3fn,0),(torch.float32,0),(torch.int8,8*1024**3)):
                save_file({'layer.weight':torch.ones(4,4).to(dtype)},path)
                self.assertEqual(int8_preload_budget(path,32*1024**3),expected)
            self.assertEqual(int8_preload_budget(path,18*1024**3),2*1024**3)
            self.assertEqual(int8_preload_budget(path,12*1024**3),0)
    def test_pipeline_full_model_and_sequential_choices(self):
        from unittest.mock import Mock,patch
        from backend.app.memory_policy import configure_pipeline
        gb=1024**3
        with torch.device('meta'):
            models=[torch.nn.Linear(16384,16384,bias=False) for _ in range(2)]
        pipe=Mock();pipe.components=dict(zip(('a','b'),models))
        with patch('backend.app.memory_policy.available_ram',return_value=32*gb):
            for free,mode,call in [(8*gb,'full','to'),(int(5.5*gb),'model_offload','enable_model_cpu_offload'),(4*gb,'sequential_offload','enable_sequential_cpu_offload')]:
                pipe.reset_mock()
                with patch('backend.app.memory_policy.available_vram',return_value=(free,16*gb,{})):
                    self.assertEqual(configure_pipeline(pipe,1024,1024)['mode'],mode)
                getattr(pipe,call).assert_called_once()

    def test_driver_memory_is_conservative_and_windows_requires_driver_measurement(self):
        from unittest.mock import patch
        import subprocess
        gb=1024**3
        with patch('torch.cuda.mem_get_info',return_value=(15*gb,16*gb)):
            with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,'8192, 16384')):
                self.assertEqual(available_vram()[0],8*gb)
            with patch('subprocess.run',side_effect=OSError),patch('os.name','nt'):
                with self.assertRaisesRegex(RuntimeError,'메모리 조회'):available_vram()
            with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,'0, 0')),patch('os.name','nt'):
                with self.assertRaisesRegex(RuntimeError,'메모리 조회'):available_vram()
            with patch('subprocess.run',side_effect=OSError),patch('os.name','posix'):
                self.assertEqual(available_vram()[2]['driver_query'],'unavailable')
    def test_residency_uses_free_memory_and_resolution_not_card_capacity(self):
        gb=1024**3
        self.assertEqual(residency_plan(14*gb,16*gb,8*gb,1024,1024)['mode'],'full')
        busy=residency_plan(6*gb,16*gb,8*gb,1024,1024)
        self.assertEqual(busy['mode'],'partial')
        self.assertEqual(busy['resident_budget_bytes'],2*gb)
        self.assertEqual(residency_plan(3*gb,16*gb,8*gb,1024,1024)['resident_budget_bytes'],0)
        self.assertEqual(residency_plan(14*gb,16*gb,8*gb,2048,2048)['mode'],'partial')
        self.assertEqual(residency_plan(14*gb,16*gb,8*gb,1024,1024,40*gb)['cpu_cache_budget_bytes'],16*gb)
        self.assertEqual(residency_plan(14*gb,16*gb,8*gb,1024,1024,6*gb)['cpu_cache_budget_bytes'],0)

    def test_file_weights_match_eager_with_peft_and_release_after_forward(self):
        from accelerate import cpu_offload
        from peft import LoraConfig,inject_adapter_in_model
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            save_file({'0.weight':torch.arange(16.).reshape(4,4),'0.bias':torch.ones(4)},path)
            def create():
                with torch.device('meta'):return torch.nn.Sequential(torch.nn.Linear(4,4))
            expected=create();load_local_weights(expected,str(path),lambda state:state,dtype=torch.float32)
            lazy=create();mapping=FileWeights(lazy,str(path),lambda state:state,dtype=torch.float32)
            self.assertEqual(lazy[0].weight.untyped_storage().nbytes(),4)
            for model in (expected,lazy):
                inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,target_modules=['0']),model,adapter_name='test')
                torch.nn.init.constant_(model[0].lora_A['test'].weight,.25)
                torch.nn.init.constant_(model[0].lora_B['test'].weight,.5)
            mapping.bind(lazy)
            cpu_offload(lazy,torch.device('cpu'),state_dict=mapping)
            x=torch.ones(2,4)
            for _ in range(2):
                torch.testing.assert_close(lazy(x),expected(x),rtol=0,atol=0)
                self.assertTrue(lazy[0].base_layer.weight.is_meta)
                self.assertEqual(mapping.cache_bytes,0)
            import os
            stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):lazy(x)

    def test_cpu_cache_is_bounded_and_declines_under_host_pressure(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            save_file({'weight':torch.ones(4,4),'bias':torch.ones(4)},path)
            with torch.device('meta'):model=torch.nn.Linear(4,4)
            mapping=FileWeights(model,str(path),lambda state:state,dtype=torch.float32)
            mapping.cpu_budget=64
            with patch('backend.app.memory_policy.available_ram',return_value=7*1024**3):
                mapping['weight'];self.assertEqual(mapping.cpu_bytes,0)
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3):
                first=mapping['weight'];self.assertIs(mapping['weight'],first)
                mapping['bias'];self.assertEqual(mapping.cpu_bytes,64)

    def test_preload_survives_adapter_rename_without_redecoding(self):
        from unittest.mock import patch
        from peft import LoraConfig,inject_adapter_in_model
        from backend.app.quantized_weights import read_tensor
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            save_file({'0.weight':torch.arange(16.).reshape(4,4),'0.bias':torch.ones(4)},path)
            with torch.device('meta'):model=torch.nn.Sequential(torch.nn.Linear(4,4))
            with patch('backend.app.memory_policy.available_ram',return_value=12*1024**3):
                constrained=FileWeights(model,str(path),lambda state:state,dtype=torch.float32,preload_budget=80)
                self.assertEqual(constrained.cpu_bytes,0)
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3),patch('backend.app.quantized_weights.read_tensor',wraps=read_tensor) as reader:
                mapping=FileWeights(model,str(path),lambda state:state,dtype=torch.float32,preload_budget=80)
                self.assertEqual(reader.call_count,2);self.assertEqual(mapping.cpu_bytes,80)
                inject_adapter_in_model(LoraConfig(r=2,target_modules=['0']),model,adapter_name='test')
                mapping.bind(model)
                self.assertIn('0.base_layer.weight',mapping.cache)
                torch.testing.assert_close(mapping['0.base_layer.weight'],torch.arange(16.).reshape(4,4))
                self.assertEqual(reader.call_count,2)
                mapping.limit_cpu_cache(0)
                self.assertEqual(mapping.cpu_bytes,0);self.assertEqual(mapping.cache,{})
                mapping['0.base_layer.weight'];self.assertEqual(reader.call_count,3)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed to verify residency promotion')
    def test_preloaded_weight_moves_to_gpu_without_duplicate_cpu_cache(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            save_file({'weight':torch.ones(4,4)},path)
            with torch.device('meta'):model=torch.nn.Linear(4,4,bias=False)
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3):
                mapping=FileWeights(model,str(path),lambda state:state,dtype=torch.float32,preload_budget=64)
            mapping.budget=64
            result=mapping['weight']
            self.assertEqual(result.device.type,'cuda')
            self.assertEqual(mapping.cpu_bytes,0);self.assertEqual(mapping.cache_bytes,64)
            self.assertIs(mapping['weight'],result)
            torch.testing.assert_close(result.cpu(),torch.ones(4,4))

    def test_cpu_cache_does_not_keep_source_file_mapped(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            save_file({'weight':torch.ones(4,4)},path)
            with torch.device('meta'):model=torch.nn.Linear(4,4,bias=False)
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3):
                mapping=FileWeights(model,str(path),lambda state:state,dtype=torch.float32,preload_budget=64)
            # Windows refuses to truncate a file while a safetensors mapping still owns it.
            path.write_bytes(b'changed')
            torch.testing.assert_close(mapping.cache['weight'],torch.ones(4,4))
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):mapping['weight']

    def test_converted_owned_tensor_is_not_copied_again(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors'
            save_file({'weight':torch.ones(4,4)},path)
            with torch.device('meta'):model=torch.nn.Linear(4,4,bias=False)
            mapping=FileWeights(model,str(path),lambda state:state,dtype=torch.bfloat16)
            self.assertEqual(mapping.file_views,set())
            owned=torch.ones(4,4,dtype=torch.bfloat16);mapping.cpu_budget=32
            with patch('backend.app.memory_policy.available_ram',return_value=32*1024**3):mapping._cache_cpu('weight',owned)
            self.assertIs(mapping.cache['weight'],owned)
