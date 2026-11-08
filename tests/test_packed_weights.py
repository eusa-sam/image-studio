import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from backend.app.inference.quantization.quantized_weights import FileWeights, load_local_weights

class PackedWeightsTests(unittest.TestCase):
    def test_placeholder_binding_is_single_strict_assignment_after_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.safetensors'
            values={'0.weight':torch.ones(4,4),'1.weight':torch.ones(4,4)}
            save_file(values,path)
            def model():
                with torch.device('meta'):
                    return torch.nn.Sequential(torch.nn.Linear(4,4,bias=False),torch.nn.Linear(4,4,bias=False))
            target=model()
            with patch.object(target,'load_state_dict',wraps=target.load_state_dict) as assign:
                mapping=FileWeights(target,str(path),lambda state:state)
            self.assertEqual(assign.call_count,1)
            self.assertEqual(assign.call_args.kwargs,dict(strict=True,assign=True))
            self.assertEqual(set(assign.call_args.args[0]),set(values))
            self.assertTrue(all(not parameter.is_meta and not parameter.requires_grad for parameter in target.parameters()))
            self.assertEqual(set(mapping),set(values))
            values['1.weight'][0,0]=float('nan');save_file(values,path)
            target=model()
            with patch.object(target,'load_state_dict',wraps=target.load_state_dict) as assign:
                with self.assertRaisesRegex(ValueError,'Nonfinite'):
                    FileWeights(target,str(path),lambda state:state)
                assign.assert_not_called()
            self.assertTrue(all(parameter.is_meta for parameter in target.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_packed_promotion_reuses_host_copy_and_removes_duplicate_budget(self):
        for int8,size in ((False,260),(True,320)):
            with tempfile.TemporaryDirectory() as folder:
                path=self.fixture(folder,int8)
                mapping=FileWeights(self.model(),str(path),lambda x:x);mapping.limit_cpu_cache(size)
                with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                    raw,scale,group=mapping.linear_packed('0.weight','cpu')
                mapping.packed_enabled=True;mapping.packed_int8_enabled=True;mapping.budget=size
                with patch('safetensors.safe_open',side_effect=AssertionError('CPU cache must avoid disk reread')):
                    packed=mapping.linear_packed('0.weight','cuda')
                torch.testing.assert_close(packed[0].cpu().float(),raw.float(),rtol=0,atol=0)
                self.assertEqual(mapping.cpu_bytes,0);self.assertFalse(mapping.native_cpu_cache)
                self.assertEqual(mapping.cache_bytes,size)
                from backend.app.inference.runtime.pipeline_cache import PipelineCache
                from types import SimpleNamespace
                cache=PipelineCache();cache.key='model';cache.positive='woman'
                cache.pipe=SimpleNamespace(memory_policy={'mode':'partial'},weight_maps=[mapping])
                cache.conditioning={'woman':{'prompt_embeds':torch.ones(1,2,device='cuda')}}
                with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                    self.assertTrue(cache.suspend())
                self.assertEqual(mapping.cache_bytes,0);self.assertFalse(mapping.packed_cache)
                self.assertEqual(mapping.cpu_bytes,size)
                self.assertEqual(cache.conditioning['woman']['prompt_embeds'].device.type,'cpu')
                self.assertEqual(cache.key,'model');self.assertTrue(cache.suspended)
                cache.clear();self.assertFalse(cache.suspended)

    def test_int8_bound_validates_without_dense_restore_and_preserves_real_weights(self):
        for group in (16,64,256):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'weights.safetensors'
                raw=torch.randint(-128,128,(16,group),dtype=torch.int8)
                scale=torch.full((16,1),.125)
                metadata={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':group}
                values={'0.weight':raw,'0.weight_scale':scale,'0.comfy_quant':torch.tensor(list(json.dumps(metadata).encode()),dtype=torch.uint8)}
                save_file(values,path)
                with torch.device('meta'):model=torch.nn.Sequential(torch.nn.Linear(group,16,bias=False))
                with patch('backend.app.inference.quantization.quantized_weights.read_tensor',side_effect=AssertionError('bound must avoid dense validation')):
                    mapping=FileWeights(model,str(path),lambda x:x)
                self.assertEqual(mapping.int8_bound_validations,1)
                from backend.app.inference.quantization.quantized_weights import undo_convrot
                torch.testing.assert_close(mapping['0.weight'],(undo_convrot(raw.float(),group)*scale).bfloat16(),rtol=0,atol=0)

    def test_int8_bound_keeps_invalid_metadata_and_overflow_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder,True)
            from safetensors.torch import load_file
            initial=load_file(path)
            for scale in (float('nan'),float('inf'),0.,-1.):
                values=dict(initial);values['0.weight_scale']=torch.full((16,1),scale);save_file(values,path)
                with self.assertRaises(ValueError):FileWeights(self.model(),str(path),lambda x:x)
            for group in (True,0,128):
                values=dict(initial)
                values['0.comfy_quant']=torch.tensor(list(json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':group}).encode()),dtype=torch.uint8)
                save_file(values,path)
                with self.assertRaises(ValueError):FileWeights(self.model(),str(path),lambda x:x)
            values=dict(initial);values['0.weight']=torch.zeros(16,16,dtype=torch.int8);values['0.weight_scale']=torch.full((16,1),1e38);save_file(values,path)
            mapping=FileWeights(self.model(),str(path),lambda x:x)
            self.assertEqual(mapping.int8_bound_validations,0)
            self.assertEqual(mapping['0.weight'].count_nonzero(),0)
            values['0.weight']=torch.full((16,16),-128,dtype=torch.int8);save_file(values,path)
            with self.assertRaises(ValueError):FileWeights(self.model(),str(path),lambda x:x)

    def test_native_cpu_cache_shares_budget_owns_storage_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder,True)
            mapping=FileWeights(self.model(),str(path),lambda x:x)
            mapping.limit_cpu_cache(320)
            with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                raw,scale,group=mapping.linear_packed('0.weight','cpu')
                again=mapping.linear_packed('0.weight','cpu')
            self.assertEqual(group,16);self.assertIs(raw,again[0])
            self.assertEqual(mapping.cpu_bytes,320)
            self.assertEqual(mapping.statistics()['native_cpu_bytes'],320)
            mapping.limit_cpu_cache(319)
            self.assertEqual(mapping.cpu_bytes,0);self.assertFalse(mapping.native_cpu_cache)
            path.write_bytes(b'changed')
            self.assertEqual(tuple(raw.shape),(16,16))
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):mapping.linear_packed('0.weight','cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_free_allocator_blocks_can_back_packed_and_decoded_caches(self):
        for packed in (True,False):
            with tempfile.TemporaryDirectory() as folder:
                path=self.fixture(folder)
                mapping=FileWeights(self.model(),str(path),lambda x:x)
                expected=mapping['0.weight'];mapping.packed_enabled=packed;mapping.budget=512
                with patch('torch.cuda.mem_get_info',return_value=(0,16*1024**3)),patch('torch.cuda.memory_reserved',return_value=4096),patch('torch.cuda.memory_allocated',return_value=0):
                    actual=mapping['0.weight']
                self.assertEqual(actual.device.type,'cuda')
                torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
                self.assertEqual(mapping.cache_bytes,260 if packed else 512)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_transient_fp8_decode_preserves_values_without_retaining_vram(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder)
            mapping=FileWeights(self.model(),str(path),lambda x:x)
            expected=mapping['0.weight']
            mapping.transient_gpu_budget=2560
            value=mapping['0.weight']
            self.assertEqual(value.device.type,'cuda')
            torch.testing.assert_close(value.cpu(),expected,rtol=0,atol=0)
            self.assertEqual(mapping.cache_bytes,0);self.assertFalse(mapping.cache);self.assertFalse(mapping.packed_cache)
            self.assertEqual(mapping.transient_gpu_reads,1)
            mapping.transient_gpu_budget=2559
            self.assertEqual(mapping['0.weight'].device.type,'cpu')

    def test_validated_quantized_cpu_reads_skip_revalidation_and_preserve_values(self):
        from safetensors import safe_open
        from backend.app.inference.quantization.quantized_weights import read_tensor
        for int8 in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                path=self.fixture(folder,int8)
                mapping=FileWeights(self.model(),str(path),lambda x:x)
                with safe_open(str(path),framework='pt',device='cpu') as reader:
                    expected=read_tensor(reader,'0.weight')
                with patch('backend.app.inference.quantization.quantized_weights.read_tensor',side_effect=AssertionError('already validated')):
                    torch.testing.assert_close(mapping['0.weight'],expected,rtol=0,atol=0)
                    torch.testing.assert_close(mapping['0.weight'],expected,rtol=0,atol=0)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):mapping['0.weight']

    def test_lazy_vae_cache_reads_once_and_checks_source_on_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vae.safetensors'
            save_file({'weight':torch.ones(2,2,3,3)},path)
            with torch.device('meta'):model=torch.nn.Conv2d(2,2,3,bias=False)
            mapping=FileWeights(model,str(path),lambda x:x)
            mapping.limit_cpu_cache(2*2*3*3*2)
            from backend.app.inference.quantization.quantized_weights import read_tensor
            with patch('backend.app.inference.quantization.quantized_weights.read_tensor',wraps=read_tensor) as reader, patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                first=mapping['weight']
                for _ in range(12):self.assertIs(mapping['weight'],first)
                self.assertEqual(reader.call_count,1)
            self.assertEqual(mapping.cpu_bytes,72)
            mapping.limit_cpu_cache(0)
            self.assertEqual(mapping.cpu_bytes,0)
            self.assertFalse(mapping.cache)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):mapping['weight']

    def fixture(self,folder,int8=False):
        path=Path(folder)/'weights.safetensors'
        raw=((torch.arange(256.).reshape(16,16)%16)-8).to(torch.int8 if int8 else torch.float8_e4m3fn)
        metadata={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':16} if int8 else {'format':'float8_e4m3fn'}
        save_file({'0.weight':raw,'0.weight_scale':torch.full((16,1),.125) if int8 else torch.tensor(.125),
            '0.comfy_quant':torch.tensor(list(json.dumps(metadata).encode()),dtype=torch.uint8)},path)
        return path
    def model(self):
        with torch.device('meta'):return torch.nn.Sequential(torch.nn.Linear(16,16,bias=False))

    def test_too_small_budget_keeps_existing_cpu_path_without_cuda_query(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder)
            mapping=FileWeights(self.model(),str(path),lambda x:x)
            mapping.packed_enabled=True;mapping.budget=259
            with patch('torch.cuda.mem_get_info',side_effect=AssertionError('no GPU allocation fits')):
                value=mapping['0.weight']
            self.assertEqual(value.device.type,'cpu');self.assertEqual(mapping.packed_cache,{})
            self.assertEqual(mapping.cache_bytes,0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_quantized_cache_uses_packed_bytes_and_preserves_peft_forward(self):
        from accelerate import cpu_offload
        from peft import LoraConfig,inject_adapter_in_model
        for int8 in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                path=self.fixture(folder,int8)
                eager=self.model();load_local_weights(eager,str(path),lambda x:x)
                lazy=self.model()
                with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3):
                    mapping=FileWeights(lazy,str(path),lambda x:x,preload_budget=512)
                for model in (eager,lazy):
                    inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,target_modules=['0']),model,adapter_name='test')
                    torch.nn.init.constant_(model[0].lora_A['test'].weight,.25)
                    torch.nn.init.constant_(model[0].lora_B['test'].weight,.5)
                mapping.bind(lazy);mapping.packed_enabled=True;mapping.budget=320 if int8 else 260;mapping.packed_int8_enabled=int8
                self.assertIn('0.base_layer.weight',mapping.packed_sources)
                eager.cuda();cpu_offload(lazy,torch.device('cuda:0'),state_dict=mapping)
                x=torch.ones((2,16),device='cuda',dtype=torch.bfloat16)
                with torch.inference_mode():
                    expected=eager(x)
                    for _ in range(3):torch.testing.assert_close(lazy(x),expected,rtol=0,atol=0)
                self.assertEqual(mapping.cache_bytes,320 if int8 else 260);self.assertEqual(mapping.cpu_bytes,0)
                self.assertEqual(mapping.statistics()['packed_int8_tensors' if int8 else 'packed_fp8_tensors'],1)
                self.assertGreaterEqual(mapping.statistics()['packed_hits'],2)
                self.assertTrue(lazy[0].base_layer.weight.is_meta)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):mapping['0.base_layer.weight']

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_insufficient_temporary_vram_falls_back_without_packed_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder)
            mapping=FileWeights(self.model(),str(path),lambda x:x)
            mapping.packed_enabled=True;mapping.budget=260;mapping.reserve=4*1024**3
            with patch('torch.cuda.mem_get_info',return_value=(mapping.reserve,16*1024**3)),patch('torch.cuda.memory_reserved',return_value=0),patch('torch.cuda.memory_allocated',return_value=0):
                value=mapping['0.weight']
            self.assertEqual(value.device.type,'cpu');self.assertEqual(mapping.packed_cache,{})

    def test_int8_gpu_path_remains_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder,True)
            mapping=FileWeights(self.model(),str(path),lambda x:x)
            mapping.packed_enabled=True;mapping.budget=320
            with patch('torch.cuda.mem_get_info',side_effect=AssertionError('INT8 GPU not enabled')):
                value=mapping['0.weight']
            self.assertEqual(value.device.type,'cpu');self.assertEqual(mapping.packed_cache,{})
