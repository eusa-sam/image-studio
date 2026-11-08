import unittest

from backend.app.worker import generation_resolution
from backend.app.inference.runtime.pipeline_cache import PipelineCache


class GenerationResolutionTests(unittest.TestCase):
    def test_disabled_detail_setting_does_not_change_model_cache_identity(self):
        plan={'runner':'krea2_txt2img','model_path':'same'}
        settings={'width':768,'height':512,'detail_resolution':512}
        first=PipelineCache.identity(plan,generation_resolution(settings,{}))
        settings['detail_resolution']=1024
        self.assertEqual(generation_resolution(settings,{'upscaler':'selected'}),(768,512))
        self.assertEqual(first,PipelineCache.identity(plan,generation_resolution(settings,{})))

    def test_each_enabled_detector_reserves_crop_without_shrinking_image(self):
        settings={'width':1024,'height':512,'detail_resolution':768}
        for resolved in ({'face_detector':'face'},{'hand_detector':'hand'},
                         {'face_detector':'face','hand_detector':'hand'}):
            with self.subTest(resolved=resolved):
                self.assertEqual(generation_resolution(settings,resolved),(1024,768))
                self.assertEqual(settings,{'width':1024,'height':512,'detail_resolution':768})
