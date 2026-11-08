import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from backend.app.inference.runtime.denoiser_residency import retain_denoiser, prepare_encoder


class DenoiserResidencyTests(unittest.TestCase):
    def setUp(self):
        guard=patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=32*1024**3)
        guard.start();self.addCleanup(guard.stop)

    def pipe(self):
        return SimpleNamespace(transformer=torch.nn.Linear(2,2),vae=torch.nn.Linear(2,2),text_encoder=torch.nn.Linear(2,2),
            memory_policy=dict(mode='model_offload',resident_budget_bytes=1024,activation_reserve_bytes=512,family='flux2'),remove_all_hooks=Mock(),to=Mock())

    def test_retains_only_when_budget_fits_and_restores_encoder_even_on_error(self):
        pipe=self.pipe();encoder=pipe.text_encoder
        with patch('backend.app.inference.runtime.memory_policy.configure_pipeline',side_effect=ValueError('failed')):
            with self.assertRaisesRegex(ValueError,'failed'):retain_denoiser(pipe,768,1024)
        self.assertIs(pipe.text_encoder,encoder)
        pipe=self.pipe();pipe.memory_policy['resident_budget_bytes']=1
        with patch('backend.app.inference.runtime.memory_policy.configure_pipeline') as configure:retain_denoiser(pipe,768,1024)
        configure.assert_not_called();pipe.to.assert_not_called()
        pipe=self.pipe()
        with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=0):
            retain_denoiser(pipe,768,1024);pipe.to.assert_not_called()
            pipe.memory_policy['mode']='denoiser_resident'
            with self.assertRaisesRegex(RuntimeError,'RAM'):prepare_encoder(pipe,768,1024,Mock())

    def test_rechecks_capacity_and_keeps_original_reserve(self):
        for mode in ('full','sequential_offload'):
            pipe=self.pipe();encoder=pipe.text_encoder;policy=pipe.memory_policy
            def configure(current,width,height,**kwargs):
                self.assertEqual(kwargs,dict(activation_reserve_bytes=512))
                return dict(mode=mode,free_vram_bytes=1536,resident_budget_bytes=1024,activation_reserve_bytes=512)
            with patch('backend.app.inference.runtime.memory_policy.configure_pipeline',side_effect=configure) as place,patch('backend.app.inference.runtime.memory_policy.available_vram',return_value=(1024,2048,{})),patch('torch.cuda.memory_reserved',return_value=128):
                retain_denoiser(pipe,768,1024)
            self.assertIs(pipe.text_encoder,encoder);self.assertIs(pipe.memory_policy,policy)
            self.assertEqual(policy['family'],'flux2');self.assertEqual(policy['free_vram_with_allocator_bytes'],1152)
            self.assertEqual(place.call_count,1 if mode=='full' else 2)
            self.assertEqual(policy['mode'],'denoiser_resident' if mode=='full' else mode)

    def test_new_prompt_restores_encoder_placement_and_preserves_policy_reference(self):
        pipe=self.pipe();policy=pipe.memory_policy;policy.update(mode='denoiser_resident',pipeline_reused=True)
        def resume(current,width,height):current.memory_policy=dict(mode='model_offload',family='flux2')
        resume=Mock(side_effect=resume)
        prepare_encoder(pipe,768,1024,resume)
        resume.assert_called_once_with(pipe,768,1024);pipe.to.assert_called_once_with('cpu')
        self.assertIs(pipe.memory_policy,policy);self.assertEqual(policy['mode'],'model_offload')
        self.assertTrue(policy['pipeline_reused'])
        prepare_encoder(pipe,768,1024,resume);self.assertEqual(resume.call_count,1)
