import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from backend.app.models import fingerprint, preflight
from backend.app.worker import component_path, upscale
from backend.app.storage import SessionStore
from backend.app.jobs import JobQueue, execution_plan


class ComponentTests(unittest.TestCase):
    def test_img2img_source_is_scoped_and_hashed(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store:
            session=store.create_session('source'); other=store.create_session('other')
            job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
            image=io.BytesIO(); Image.new('RGB',(32,32),'blue').save(image,format='PNG')
            store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':image.getvalue()})
            base={'runner':'sdxl_img2img','settings':{'source_job':job}}
            with patch('backend.app.jobs.preflight',side_effect=lambda settings:dict(base)):
                plan=execution_plan(store,session,{})
                self.assertEqual(plan['source']['sha256'],hashlib.sha256(image.getvalue()).hexdigest())
                self.assertEqual(plan['source']['size'],[32,32])
                with self.assertRaises(FileNotFoundError):
                    execution_plan(store,other,{})

    def test_img2img_requires_source_and_nonzero_denoising_steps(self):
        with self.assertRaisesRegex(ValueError,'선택'):
            preflight({'model':'x','flow':'img2img'})
        with self.assertRaisesRegex(ValueError,'Steps'):
            preflight({'model':'x','flow':'img2img','source_job':str(uuid4()),'steps':1,'strength':0.1})

    def test_component_hash_and_path_are_checked_before_load(self):
        with tempfile.TemporaryDirectory() as directory, patch('backend.app.worker.MODEL_ROOT',Path(directory)):
            path=Path(directory)/'model.pth'; path.write_bytes(b'weights')
            component=dict(path=path.name,fingerprint=fingerprint(path),profile={'sha256':hashlib.sha256(b'weights').hexdigest()})
            self.assertEqual(component_path(component),path)
            component['profile']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):
                component_path(component)
            component['path']='../outside.pth'
            with self.assertRaisesRegex(ValueError,'CHANGED'):
                component_path(component)

    def test_preflight_rejects_wrong_component_role_and_accepts_1024_upscale(self):
        with tempfile.TemporaryDirectory() as directory, patch('backend.app.models.MODEL_ROOT',Path(directory)):
            for relative in ('model_index.json','scheduler/scheduler_config.json','unet/config.json','vae/config.json',
                'text_encoder/config.json','text_encoder_2/config.json','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer_2/vocab.json','tokenizer_2/merges.txt'):
                path=Path(directory)/'configs/sdxl'/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{}')
            inventory=[dict(id='image',available=True,path='checkpoint/image',fingerprint={}),
                dict(id='up',component='upscaler',path='upscale_models/up',fingerprint={},profile={'scale':2})]
            with patch('backend.app.models.scan',return_value=inventory):
                with self.assertRaisesRegex(ValueError,'vae'):
                    preflight({'model':'image','vae':'up'})
                self.assertEqual(preflight({'model':'image','upscaler':'up'})['settings']['width'],1024)
                plan=preflight({'model':'image','upscaler':'up','width':512,'height':512})
                self.assertEqual(plan['components']['upscaler']['profile']['scale'],2)

    def test_upscale_rgb_normalization_scale_and_finite_output_on_cpu(self):
        import torch
        import numpy as np
        from PIL import Image
        class Descriptor:
            input_channels=3; output_channels=3; scale=2
            invalid=False
            def eval(self): return self
            def cuda(self): return self
            def __call__(self,tensor):
                output=tensor.repeat_interleave(2,-1).repeat_interleave(2,-2)
                return output * float('nan') if self.invalid else output
        model=Descriptor()
        pixels=np.arange(4*4*3,dtype=np.uint8).reshape(4,4,3)
        with patch('spandrel.ImageModelDescriptor',Descriptor), patch('spandrel.ModelLoader') as loader, \
             patch('torch.load',return_value={}), patch.object(torch.Tensor,'cuda',lambda tensor:tensor):
            loader.return_value.load_from_state_dict.return_value=model
            result=upscale(Image.fromarray(pixels),Path('fake.pth'),2)
            np.testing.assert_array_equal(np.asarray(result),pixels.repeat(2,0).repeat(2,1))
            with self.assertRaisesRegex(ValueError,'채널/배율'):
                upscale(Image.fromarray(pixels),Path('fake.pth'),4)
            model.invalid=True
            with self.assertRaisesRegex(ValueError,'픽셀 오류'):
                upscale(Image.fromarray(pixels),Path('fake.pth'),2)

    def test_postprocess_failure_keeps_original_downloadable(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, SessionStore(directory) as store, patch.object(JobQueue,'_loop'):
            session=store.create_session('partial')
            receipt=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)
            job=receipt['job_id']; plan={'test':True,'runner':'sdxl_txt2img','settings':{}}
            queue=JobQueue(store)
            # Queue initialization marks unresolved prior work interrupted; explicitly prepare this test attempt.
            store.transition(session,job,('INTERRUPTED',),'QUEUED',details={'plan':plan})
            work=Path(directory)/'runtime/jobs'/job; work.mkdir(parents=True)
            Image.new('RGB',(8,8),'red').save(work/'original.png')
            (work/'report.json').write_text(json.dumps({'schema_version':1,'stages':[{'name':'upscale','status':'FAILED'}]}))
            (work/'worker-done.json').write_text(json.dumps({'code':1,'cpu_cached':False}))
            (work/'worker.log').write_text('upscale failed')
            process=Mock(); process.wait.return_value=1; process.stdin=io.BytesIO()
            with patch('backend.app.jobs.preflight',return_value=plan), patch('backend.app.jobs.Ollama.image_ready'), \
                 patch('backend.app.jobs.subprocess.Popen',return_value=process):
                queue._execute(session,job)
            state=store.jobs(session)[0]
            self.assertEqual(state['status'],'PARTIAL_FAILED')
            self.assertTrue(state['has_original'])
            self.assertTrue(store.artifact(session,job,'original.png').is_file())
            with self.assertRaises(FileNotFoundError):
                store.artifact(session,job,'image.png')
            queue.close()
