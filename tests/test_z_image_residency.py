import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch
from backend.app import z_image


class PartialResidencyTests(unittest.TestCase):
    def pipe(self):
        transformer=torch.nn.Linear(2,2)
        transformer.dtype=torch.float32
        transformer._no_split_modules=['Block']
        return SimpleNamespace(text_encoder=None,transformer=transformer,vae=torch.nn.Linear(2,2),
                               remove_all_hooks=Mock(),to=Mock())

    def policy(self,**changes):
        return dict(mode='sequential_offload',resident_budget_bytes=1024,free_ram_bytes=16*1024**3,**changes)

    def test_current_budget_reserves_vae_and_keeps_blocks_together(self):
        pipe=self.pipe();mapping={'layers.0':0,'layers.1':'cpu'}
        events=[]
        pipe.remove_all_hooks.side_effect=lambda:events.append('unhook')
        pipe.to.side_effect=lambda device:events.append(device)
        with patch('backend.app.memory_policy.configure_pipeline',return_value=self.policy()),patch('accelerate.infer_auto_device_map',return_value=mapping) as infer,patch('accelerate.dispatch_model',side_effect=lambda *a,**k:events.append('dispatch')) as dispatch,patch('accelerate.cpu_offload') as offload:
            result=z_image.configure_pipeline(pipe,768,1024)
        infer.assert_called_once_with(pipe.transformer,max_memory={0:1000,'cpu':8*1024**3},no_split_module_classes=['Block'],dtype=torch.float32)
        self.assertEqual(events,['unhook','cpu','dispatch'])
        dispatch.assert_called_once_with(pipe.transformer,mapping,main_device='cuda:0')
        offload.assert_called_once_with(pipe.vae,execution_device=torch.device('cuda:0'))
        self.assertEqual(result['mode'],'block_offload');self.assertEqual(result['device_map'],mapping)

    def test_no_mixed_fit_keeps_existing_hooks_and_never_uses_disk(self):
        for mapping in ({'':'cpu'},{'':0},{'a':0,'b':'disk'}):
            pipe=self.pipe()
            with patch('backend.app.memory_policy.configure_pipeline',return_value=self.policy()),patch('accelerate.infer_auto_device_map',return_value=mapping),patch('accelerate.dispatch_model') as dispatch:
                self.assertEqual(z_image.configure_pipeline(pipe,768,1024)['mode'],'sequential_offload')
            pipe.remove_all_hooks.assert_not_called();dispatch.assert_not_called()
        for changes,encoder in [({'free_ram_bytes':8*1024**3},None),({'resident_budget_bytes':24},None),({'mode':'full'},None),({},object())]:
            pipe=self.pipe();pipe.text_encoder=encoder;policy=self.policy();policy.update(changes)
            with patch('backend.app.memory_policy.configure_pipeline',return_value=policy),patch('accelerate.infer_auto_device_map') as infer:
                self.assertIs(z_image.configure_pipeline(pipe,768,1024),policy)
            infer.assert_not_called();pipe.remove_all_hooks.assert_not_called()

    def test_dispatch_failure_propagates_and_replan_removes_stale_mapping(self):
        pipe=self.pipe()
        with patch('backend.app.memory_policy.configure_pipeline',return_value=self.policy()),patch('accelerate.infer_auto_device_map',return_value={'a':0,'b':'cpu'}),patch('accelerate.dispatch_model',side_effect=RuntimeError('allocation failed')):
            with self.assertRaisesRegex(RuntimeError,'allocation failed'):z_image.configure_pipeline(pipe,768,1024)
        pipe._generation_size=(768,1024)
        pipe.memory_policy=dict(family='zimage',pipeline_reused=True,device_map={'a':0},transformer_budget_bytes=1000)
        with patch.object(z_image,'configure_pipeline',return_value={'mode':'full'}),patch('backend.app.memory_policy.available_vram',return_value=(100,200,{})),patch('torch.cuda.memory_reserved',return_value=0):
            z_image.reconfigure_after_encoder_change(pipe)
        self.assertNotIn('device_map',pipe.memory_policy)
        self.assertNotIn('transformer_budget_bytes',pipe.memory_policy)
        self.assertTrue(pipe.memory_policy['pipeline_reused'])
