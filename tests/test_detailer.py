import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from backend.app.inference.postprocess.detailer import decode, refine_regions, select_regions
from backend.app.models import preflight,scan
from backend.app.jobs import original_artifacts


class DetailerTests(unittest.TestCase):
    def test_krea_detail_conditioning_describes_the_crop_instead_of_a_whole_scene(self):
        from backend.app.inference.postprocess.detailer import detail_prompt
        scene='a woman in a field of flowers'
        self.assertIn('single hand',detail_prompt('hand',scene,'krea2'))
        self.assertNotIn(scene,detail_prompt('hand',scene,'krea2'))
        self.assertEqual(detail_prompt('face',scene,'krea2'),'detailed face, natural eyes, '+scene)
        self.assertEqual(detail_prompt('hand',scene,'sdxl'),'natural hand, anatomically correct fingers, '+scene)
        with self.assertRaises(ValueError):detail_prompt('other',scene,'krea2')

    def test_region_cap_order_and_seed_wrap_are_deterministic(self):
        regions=[{'confidence':.4,'target':'face'},{'confidence':.9,'target':'hand'},{'confidence':.8,'target':'face'}]
        selected=select_regions(regions,2,2**32-1)
        self.assertEqual([r['confidence'] for r in selected],[.9,.8])
        self.assertEqual([r['seed'] for r in selected],[0,1])
        self.assertNotIn('seed',regions[0])

    def test_letterbox_nms_clipping_and_invalid_output(self):
        output=np.zeros((1,5,8400),dtype=np.float32)
        output[0,:,0]=[320,320,160,160,.9]
        output[0,:,1]=[321,321,160,160,.8]
        output[0,:,2]=[1,161,40,40,.7]
        boxes=decode(output,(200,100),(640,320),(0,160),.35)
        self.assertEqual(boxes[0]['box'],[75,25,125,75]);self.assertEqual(len(boxes),1)
        output[0,0,0]=np.nan
        with self.assertRaises(ValueError):decode(output,(200,100),(640,320),(0,160),.35)

    def test_empty_and_overlapping_regions_preserve_every_pixel_outside_mask(self):
        image=Image.fromarray(np.random.default_rng(3).integers(0,255,(80,100,3),dtype=np.uint8))
        output,mask,records=refine_regions(image,[],lambda *args:self.fail('no detections must skip regeneration'))
        np.testing.assert_array_equal(output,image);self.assertFalse(np.asarray(mask).any());self.assertEqual(records,[])
        regions=[{'box':[0,0,35,40],'target':'face'},{'box':[20,20,70,65],'target':'hand'}]
        seen=[]
        def regenerate(crop,index,region):
            seen.append(np.asarray(crop).copy())
            return Image.new('RGB',(512,512),'red' if index==0 else 'blue')
        output,mask,records=refine_regions(image,regions,regenerate,resolution=512)
        outside=np.asarray(mask)==0
        np.testing.assert_array_equal(np.asarray(image)[outside],np.asarray(output)[outside])
        self.assertFalse(np.array_equal(image,output));self.assertEqual(len(records),2)
        self.assertTrue(any((seen[1][:,:,0]>200).flat))

    def test_profile_target_and_denoising_steps_are_checked(self):
        with patch('backend.app.models.scan',return_value=[dict(id='base',available=True,path='base',fingerprint={}),
            dict(id='hand',component='detector',profile={'target':'hand'},path='hand',fingerprint={})]), patch('pathlib.Path.is_file',return_value=True):
            with self.assertRaisesRegex(ValueError,'대상'):
                preflight({'model':'base','face_detector':'hand'})
            with self.assertRaisesRegex(ValueError,'Steps'):
                preflight({'model':'base','hand_detector':'hand','detail_steps':4,'detail_strength':0.1})
            self.assertIn('hand_detector',preflight({'model':'base','hand_detector':'hand'})['components'])

    def test_partial_detail_png_does_not_hide_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);Image.new('RGB',(8,8),'red').save(root/'original.png')
            Image.new('RGB',(8,8),'blue').save(root/'detail-0-before.png')
            (root/'detail-0-after.png').write_bytes(b'partial')
            (root/'report.json').write_text(json.dumps({'stages':[{'status':'SUCCEEDED'},{'status':'RUNNING'}]}))
            artifacts=original_artifacts(root,'CANCELLED')
            self.assertEqual([s['status'] for s in json.loads(artifacts['report.json'])['stages']],['SUCCEEDED','CANCELLED'])
            self.assertIn('original.png',artifacts);self.assertIn('detail-0-before.png',artifacts)
            self.assertNotIn('detail-0-after.png',artifacts)

    def test_invalid_or_unreadable_report_does_not_hide_valid_original(self):
        for content in ('{broken','[]','null','{"stages":null}','{"stages":[null]}','{"stages":"bad"}','directory'):
            for status in (None,'FAILED','CANCELLED','INTERRUPTED'):
                with self.subTest(content=content,status=status),tempfile.TemporaryDirectory() as folder:
                    root=Path(folder)
                    Image.new('RGB',(8,8),'red').save(root/'original.png')
                    Image.new('RGB',(8,8),'blue').save(root/'detail-0-before.png')
                    source=(root/'original.png').read_bytes()
                    report=root/'report.json'
                    if content=='directory':report.mkdir()
                    else:report.write_text(content,encoding='utf-8')
                    artifacts=original_artifacts(root,status)
                    self.assertEqual(artifacts['original.png'],source)
                    self.assertIn('detail-0-before.png',artifacts)
                    recovered=json.loads(artifacts['report.json'])
                    self.assertEqual(recovered['status'],'INTERRUPTED_AFTER_GENERATION')
                    self.assertTrue(recovered['report_error'])
                    if content=='directory':self.assertTrue(report.is_dir())
                    else:self.assertEqual(report.read_text(encoding='utf-8'),content)
                    self.assertEqual((root/'original.png').read_bytes(),source)
