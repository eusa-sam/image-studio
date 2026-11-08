import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from backend.app.inference.quantization.quantized_weights import open_weights, weight_readers


class WeightReaderTests(unittest.TestCase):
    def test_reuses_within_scope_and_closes_before_next_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors';save_file({'weight':torch.arange(8)},str(path))
            original=hashlib.sha256(path.read_bytes()).hexdigest();events=[]
            class Tracked:
                def __init__(self,*args,**kwargs):self.reader=safe_open(*args,**kwargs);events.append('open')
                def __enter__(self):return self.reader.__enter__()
                def __exit__(self,*args):events.append('close');return self.reader.__exit__(*args)
            with patch('safetensors.safe_open',side_effect=Tracked):
                with weight_readers():
                    with open_weights(path) as a:values=a.get_tensor('weight')
                    with open_weights(path.parent/'.'/path.name) as b:self.assertIs(a,b)
                    self.assertEqual(events,['open'])
                self.assertEqual(events,['open','close'])
                torch.testing.assert_close(values,torch.arange(8))
                with weight_readers():
                    with open_weights(path) as c:self.assertIsNot(c,a)
                self.assertEqual(events,['open','close','open','close'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original)

    def test_nested_scope_and_exception_preserve_outer_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors';save_file({'weight':torch.ones(2)},str(path))
            with patch('safetensors.safe_open',wraps=safe_open) as opening:
                with weight_readers():
                    with open_weights(path) as outer:pass
                    with self.assertRaisesRegex(ValueError,'failure'):
                        with weight_readers():
                            with open_weights(path) as inner:self.assertIs(outer,inner)
                            raise ValueError('failure')
                    with open_weights(path) as restored:self.assertIs(restored,outer)
                with open_weights(path) as outside:self.assertIsNot(outside,outer)
                self.assertEqual(opening.call_count,2)

    def test_pipeline_failure_releases_scope(self):
        from backend.app.worker import pipeline_output
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors';save_file({'weight':torch.ones(2)},str(path));seen=[]
            def pipeline(**arguments):
                with open_weights(path) as reader:seen.append(reader)
                raise ValueError('failed inference')
            with self.assertRaisesRegex(ValueError,'failed inference'):pipeline_output(pipeline,prompt='test')
            with open_weights(path) as reader:self.assertIsNot(reader,seen[0])

    def test_validation_and_demand_read_share_one_reader(self):
        from backend.app.inference.quantization.quantized_weights import FileWeights
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.safetensors';model=torch.nn.Linear(2,2)
            original=model.weight.detach().clone();save_file(model.state_dict(),str(path))
            with patch('safetensors.safe_open',wraps=safe_open) as opening,weight_readers():
                weights=FileWeights(model,str(path),lambda state:state)
                actual=weights['weight']
                torch.testing.assert_close(actual,original.to(torch.bfloat16),rtol=0,atol=0)
                self.assertEqual(opening.call_count,1)
