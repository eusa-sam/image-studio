import io,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from PIL import Image
from backend.app.models import preflight
from backend.app.jobs import execution_plan
from backend.app.storage import SessionStore
from backend.app.references import save_reference


class AdapterPlanTests(unittest.TestCase):
    def test_model_component_and_reference_selection(self):
        with tempfile.TemporaryDirectory() as directory,patch('backend.app.models.MODEL_ROOT',Path(directory)):
            root=Path(directory)
            for name in ('model_index.json','scheduler/scheduler_config.json','unet/config.json','vae/config.json','text_encoder/config.json','text_encoder_2/config.json','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer_2/vocab.json','tokenizer_2/merges.txt'):
                p=root/'configs/sdxl'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            config=root/'configs/ip_adapter/clip_vision_h/config.json';config.parent.mkdir(parents=True);config.write_text('{}')
            rows=[dict(id='image',available=True,family='sdxl_candidate',path='checkpoint/image',fingerprint={})]
            rows += [dict(id=key,component=key,path=key+'/model',fingerprint={},profile={'family':'sdxl'}) for key in ('ip_adapter','clip_vision')]
            settings=dict(model='image',ip_adapter='ip_adapter',clip_vision='clip_vision',reference_job=str(uuid4()))
            with patch('backend.app.models.scan',return_value=rows):
                plan=preflight(settings);self.assertIn('ip_adapter',plan['components']);self.assertIn('vision_config_sha256',plan)
                for change in ({'clip_vision':None},{'reference_job':None},{'reference_attachment':str(uuid4())},{'reference_job':'bad'}, {'ip_adapter':None},{'clip_vision':'ip_adapter'},{'ip_adapter_scale':float('nan')}):
                    with self.subTest(change=change),self.assertRaises(ValueError):preflight(settings|change)
                rows[0]['family']='krea2'
                with self.assertRaisesRegex(ValueError,'SDXL'):preflight(settings)

    def test_reference_sources_are_distinct_scoped_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory,SessionStore(directory) as store:
            session=store.create_session('references');other=store.create_session('other')
            buffer=io.BytesIO();Image.new('RGB',(1024,1024),'blue').save(buffer,format='PNG');data=buffer.getvalue()
            job=store.save_prompt(session,str(uuid4()),expected_revision=0,positive='x',negative='',settings={},request_id=str(uuid4()),generate=True)['job_id']
            store.transition(session,job,('AWAITING_VALIDATION',),'SUCCEEDED',artifacts={'image.png':data})
            reference=save_reference(store,session,'palette.png',data)
            for runner,adapter,flow in [('sdxl_txt2img','adapter','txt2img'),('flux2_reference_edit',None,'reference_edit')]:
                for key,value,kind in [('reference_job',job,'generated_output'),('reference_attachment',reference['id'],'user_reference')]:
                    settings=dict(ip_adapter=adapter,flow=flow,reference_job=None,reference_attachment=None);settings[key]=value
                    with patch('backend.app.jobs.preflight',return_value={'runner':runner,'settings':settings}):
                        plan=execution_plan(store,session,settings);self.assertEqual(plan['reference']['kind'],kind)
                        self.assertEqual(plan['reference']['id'],value);self.assertNotIn('source',plan)
                        if key=='reference_attachment':
                            if flow=='reference_edit':
                                self.assertEqual(plan['reference']['sha256'],hashlib.sha256(data).hexdigest())
                                self.assertEqual(plan['reference']['representation'],'original')
                            else:self.assertNotEqual(plan['reference']['sha256'],hashlib.sha256(data).hexdigest())
                        with self.assertRaises((ValueError,FileNotFoundError)):execution_plan(store,other,settings)
            preview=Path(directory)/'sessions'/session/'references'/reference['id']/'preview.png';preview.write_bytes(b'tampered')
            with patch('backend.app.jobs.preflight',return_value={'runner':'sdxl_txt2img','settings':settings}):
                with self.assertRaisesRegex(ValueError,'해시'):execution_plan(store,session,settings)
