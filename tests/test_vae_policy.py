import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
import torch
from backend.app.inference.runtime.vae_policy import adaptive_decode, full_decode_fits


class VaePolicyTests(unittest.TestCase):
    def test_pressure_and_large_images_keep_tiling(self):
        vae=torch.nn.Linear(1,1)
        with patch('backend.app.inference.runtime.memory_policy.available_vram',side_effect=RuntimeError('driver query unavailable')):
            self.assertFalse(full_decode_fits(vae,torch.zeros(1,4,1,128,128)))
        with patch('backend.app.inference.runtime.memory_policy.available_vram',return_value=(10*1024**3,16*1024**3,{})),patch('torch.cuda.memory_reserved',return_value=0),patch('torch.cuda.memory_allocated',return_value=0):
            self.assertFalse(full_decode_fits(vae,torch.zeros(1,4,96,128)))
            self.assertFalse(full_decode_fits(vae,torch.zeros(1,4,192,128)))
            self.assertTrue(full_decode_fits(vae,torch.zeros(1,4,1,128,128)))
            self.assertFalse(full_decode_fits(vae,torch.zeros(1,4,1,96,128)))
        with patch('backend.app.inference.runtime.memory_policy.available_vram',return_value=(1024,16*1024**3,{})),patch('torch.cuda.memory_reserved',return_value=0),patch('torch.cuda.memory_allocated',return_value=0):
            self.assertFalse(full_decode_fits(vae,torch.zeros(1,4,1,128,128)))

    def test_restores_policy_and_method_after_failure(self):
        decode=Mock(side_effect=RuntimeError('cancel'));vae=SimpleNamespace(use_tiling=True,decode=decode)
        with patch('backend.app.inference.runtime.vae_policy.full_decode_fits',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'cancel'):
                with adaptive_decode(vae):vae.decode(torch.zeros(1))
        self.assertTrue(vae.use_tiling);self.assertIs(vae.decode,decode)
