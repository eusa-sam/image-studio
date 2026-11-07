import hashlib,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from backend.app.pipeline_cache import PipelineCache
from backend.app.worker import run


class Loaded(Exception):pass


class KreaStartupTests(unittest.TestCase):
    def test_verified_encoder_overlaps_checkpoint_and_failures_do_not_load_denoiser(self):
        for failure in (None,'component','checkpoint'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);config=root/'configs/krea2';config.mkdir(parents=True)
                (config/'manifest.json').write_bytes(b'{}')
                plan=dict(runner='krea2_txt2img',settings=dict(width=768,height=1024),krea2={'sha256':'checked'},config_sha256=hashlib.sha256(b'{}').hexdigest())
                request=root/'request.json';request.write_text(json.dumps(dict(plan=plan,positive='original')))
                ready=threading.Event();prepared=(object(),object(),object(),{'prompt_embeds':'value'},'cpu')
                resolved={'clip':root/'clip','vae':root/'vae'}
                def components(*args):
                    if failure=='component':raise ValueError('BAD_COMPONENT')
                    return resolved,[]
                def verify(plan,cache,future):
                    result=future.result()
                    if not ready.wait(3):raise AssertionError('Encoder did not overlap checkpoint verification')
                    if failure=='checkpoint':raise ValueError('BAD_CHECKPOINT')
                    return root/'model','checked',*result
                def prepare(*args):
                    ready.set();return prepared
                cache=PipelineCache()
                with patch('backend.app.worker.MODEL_ROOT',root),patch('backend.app.worker.verify_components',side_effect=components),patch('backend.app.worker.verify_inputs',side_effect=verify),patch('backend.app.krea2.prepare_encoder',side_effect=prepare) as encoder,patch('backend.app.krea2.load_pipeline',side_effect=Loaded) as loader,patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.empty_cache'):
                    with self.assertRaises(Loaded if failure is None else ValueError):run(request,cache)
                self.assertIsNone(cache.pipe)
                self.assertFalse(cache.conditioning)
                if failure=='component':encoder.assert_not_called()
                else:encoder.assert_called_once_with(root/'clip',config,'original')
                if failure is None:
                    self.assertIs(loader.call_args.kwargs['prepared_encoder'],prepared)
                else:loader.assert_not_called()

    def test_large_gpu_keeps_original_full_loading_path(self):
        from backend.app.krea2 import prepare_encoder
        with patch('backend.app.memory_policy.available_vram',return_value=(40*1024**3,48*1024**3,{})),patch('backend.app.krea2.verify_config') as config:
            self.assertIsNone(prepare_encoder(Path('unused'),Path('unused'),'prompt'))
            config.assert_not_called()
