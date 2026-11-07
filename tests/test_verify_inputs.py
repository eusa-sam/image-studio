import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.models import fingerprint
from backend.app.worker import verify_inputs


class VerifyInputsTests(unittest.TestCase):
    def test_worker_imports_overlap_verification_and_failure_prevents_model_initialization(self):
        import builtins,json
        from backend.app.worker import _run
        started=threading.Event();importing=threading.Event()
        original=builtins.__import__
        def verify(*args):
            started.set()
            if not importing.wait(3):raise AssertionError('Library import did not overlap verification')
            raise ValueError('INVALID_MODEL')
        def imports(name,*args,**kwargs):
            if name=='torch':
                if not started.wait(3):raise AssertionError('Verification did not start')
                importing.set()
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'request.json';path.write_text(json.dumps(dict(plan={'settings':{}})))
            with patch('backend.app.worker.verify_inputs',side_effect=verify),patch('torch.cuda.reset_peak_memory_stats') as gpu,patch('builtins.__import__',side_effect=imports):
                with self.assertRaisesRegex(ValueError,'INVALID_MODEL'):_run(path)
            gpu.assert_not_called()

    def test_overlap_keeps_hashes_paths_and_component_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'model';checkpoint.write_bytes(b'model')
            component=root/'encoder';component.write_bytes(b'encoder')
            row=dict(path='encoder',fingerprint=fingerprint(component),profile=dict(sha256=hashlib.sha256(b'encoder').hexdigest()))
            plan=dict(model_path='model',fingerprint=fingerprint(checkpoint),runner='krea2_txt2img',components={'clip':row},loras=[])
            entered={name:threading.Event() for name in ('model','encoder')}
            class Cache:
                def file_hash(self,path):
                    entered[path.name].set()
                    other='encoder' if path.name=='model' else 'model'
                    if not entered[other].wait(3):raise AssertionError('Independent verification did not overlap')
                    return hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('backend.app.worker.MODEL_ROOT',root):
                path,sha,resolved,loras=verify_inputs(plan,Cache())
                self.assertEqual(path,checkpoint);self.assertEqual(sha,hashlib.sha256(b'model').hexdigest())
                self.assertEqual(resolved,{'clip':component});self.assertEqual(loras,[])
                row['profile']['sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'COMPONENT_HASH_MISMATCH'):verify_inputs(plan,Cache())
                row['fingerprint']={}
                with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):verify_inputs(plan)

    def test_invalid_model_path_rejected_before_component_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('backend.app.worker.MODEL_ROOT',root),patch('backend.app.worker.component_path') as component:
                with self.assertRaisesRegex(ValueError,'MODEL_CHANGED'):
                    verify_inputs(dict(model_path='../outside',fingerprint={},components={'clip':{}}))
                component.assert_not_called()

    def test_external_component_future_is_consumed_once_and_propagates_failure(self):
        from concurrent.futures import Future
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'model';checkpoint.write_bytes(b'model')
            plan=dict(model_path='model',fingerprint=fingerprint(checkpoint),runner='krea2_txt2img')
            future=Future();future.set_result(({'clip':root/'already-verified'},[]))
            with patch('backend.app.worker.MODEL_ROOT',root),patch('backend.app.worker.verify_components') as components:
                result=verify_inputs(plan,verified_components=future)
                self.assertEqual(result[2],{'clip':root/'already-verified'})
                components.assert_not_called()
                failed=Future();failed.set_exception(ValueError('COMPONENT_FAILED'))
                with self.assertRaisesRegex(ValueError,'COMPONENT_FAILED'):verify_inputs(plan,verified_components=failed)
