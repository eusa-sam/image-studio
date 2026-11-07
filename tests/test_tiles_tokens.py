from tokenizer_fixture import local_tokenizers
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from spandrel import ModelTiling
from fastapi.testclient import TestClient
from backend.app.worker import tiled_upscale
from backend.app.prompt_tokens import count_tokens, tokenizers
from backend.app.main import create_app


class TilesTokensTests(unittest.TestCase):
    def test_tiling_matches_local_filter_at_seams_and_preserves_alpha(self):
        class Model:
            tiling = ModelTiling.SUPPORTED
            def __call__(self, tensor):
                return torch.nn.functional.avg_pool2d(tensor,3,1,1).repeat_interleave(2,-1).repeat_interleave(2,-2)
        pixels=np.random.default_rng(42).integers(0,256,(17,23,4),dtype=np.uint8)
        image=Image.fromarray(pixels)
        with patch.object(torch.Tensor,'cuda',lambda t:t):
            tiled=tiled_upscale(image,Model(),2,tile_size=8,halo=2)
            full=tiled_upscale(image,Model(),2,tile_size=32,halo=2)
        np.testing.assert_array_equal(np.asarray(tiled),np.asarray(full))
        np.testing.assert_array_equal(np.asarray(tiled.getchannel('A')),np.asarray(image.getchannel('A').resize((46,34),Image.Resampling.LANCZOS)))

    def test_unsafe_tiling_and_invalid_output_rejected(self):
        class Model:
            tiling=ModelTiling.DISCOURAGED
            def __call__(self,t): return t*float('nan')
        with patch.object(torch.Tensor,'cuda',lambda t:t):
            with self.assertRaisesRegex(ValueError,'권장'):
                tiled_upscale(Image.new('RGB',(9,9)),Model(),1,tile_size=8)
            with self.assertRaisesRegex(ValueError,'픽셀'):
                tiled_upscale(Image.new('RGB',(8,8)),Model(),1)

    @local_tokenizers()
    def test_actual_clip_counts_match_untruncated_ids_and_boundary(self):
        for tokenizer in tokenizers():
            for text in ('', '여성 한 명, blue dress', 'cat '*75, 'cat '*76):
                report=count_tokens(text,'', [tokenizer])['encoders'][0]
                self.assertEqual(report['positive'],len(tokenizer(text,truncation=False)['input_ids'])-2)
                self.assertEqual(report['truncated'],report['positive']>75)
        self.assertFalse(count_tokens('cat '*75, '')['truncated'])
        self.assertTrue(count_tokens('', 'cat '*76)['truncated'])

    def test_token_api_bounds_and_local_request(self):
        with tempfile.TemporaryDirectory() as root, TestClient(create_app(root),base_url='http://127.0.0.1:8765') as client, local_tokenizers():
            response=client.post('/api/prompt-tokens',json={'positive':'cat '*76})
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.json()['truncated'])
            self.assertEqual(len(response.json()['encoders']),2)
            self.assertEqual(client.post('/api/prompt-tokens',json={'positive':'x'*16001}).status_code,422)
