import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import accelerate.hooks as hooks
from backend.app.offload import reuse_allocator,prefer_cudnn
from backend.app.worker import run


class OffloadTests(unittest.TestCase):
    def test_attention_priority_restored_on_failure_and_fallbacks_enabled(self):
        from torch.nn.attention import SDPBackend
        original=torch._C._get_sdp_priority_order()
        with self.assertRaisesRegex(ValueError,'failed'):
            with prefer_cudnn():
                self.assertEqual(torch._C._get_sdp_priority_order()[0],int(SDPBackend.CUDNN_ATTENTION))
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertTrue(torch.backends.cuda.mem_efficient_sdp_enabled())
                raise ValueError('failed')
        self.assertEqual(torch._C._get_sdp_priority_order(),original)

    def test_scope_preserves_validation_and_restores_after_failure(self):
        original=hooks.set_module_tensor_to_device
        with self.assertRaisesRegex(ValueError,'shape'):
            with reuse_allocator():
                with torch.device('meta'):model=torch.nn.Linear(2,2,bias=False)
                hooks.set_module_tensor_to_device(model,'weight','cpu',value=torch.ones(3,3))
        self.assertIs(hooks.set_module_tensor_to_device,original)

    def test_only_krea_worker_uses_policy_and_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'request.json'
            for runner,enabled in [('krea2_txt2img',True),('sdxl_txt2img',False)]:
                path.write_text(json.dumps({'plan':{'runner':runner}}))
                with patch.object(hooks,'set_module_tensor_to_device') as setter:
                    def execute(request,cache=None):
                        self.assertEqual(request,path)
                        self.assertIsNone(cache)
                        hooks.set_module_tensor_to_device('module','weight','cuda',value='value',tied_params_map={})
                        return 7
                    with patch('backend.app.worker._run',side_effect=execute):self.assertEqual(run(path),7)
                    expected=dict(value='value',tied_params_map={})
                    if enabled:expected['clear_cache']=False
                    setter.assert_called_once_with('module','weight','cuda',**expected)
                    self.assertIs(hooks.set_module_tensor_to_device,setter)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_forward_keeps_values_and_offloads_without_flushing(self):
        from accelerate import cpu_offload
        model=torch.nn.Linear(16,16,bias=False).to(dtype=torch.bfloat16)
        model.requires_grad_(False)
        inputs=torch.ones(2,16,device='cuda',dtype=torch.bfloat16)
        expected=torch.nn.functional.linear(inputs,model.weight.cuda())
        cpu_offload(model,execution_device=torch.device('cuda:0'))
        with patch('accelerate.utils.modeling.clear_device_cache') as clear:
            with reuse_allocator(),torch.inference_mode():
                for _ in range(3):torch.testing.assert_close(model(inputs),expected,rtol=0,atol=0)
                self.assertTrue(model.weight.is_meta)
            clear.assert_not_called()

