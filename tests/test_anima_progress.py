import unittest
from types import SimpleNamespace

from diffusers import AnimaModularPipeline
from diffusers.modular_pipelines.anima.denoise import AnimaDenoiseStep, AnimaDenoiseLoopWrapper
from backend.app.anima import progress_reporting
from backend.app.worker import pipeline_output


class AnimaProgressTests(unittest.TestCase):
    def test_real_text_and_image_loops_report_even_when_console_bar_disabled(self):
        pipe=AnimaModularPipeline()
        pending=[pipe._blocks];loops=[]
        while pending:
            block=pending.pop();pending.extend(block.sub_blocks.values())
            if isinstance(block,AnimaDenoiseLoopWrapper):
                block.set_progress_bar_config(disable=True);loops.append(block)
        self.assertEqual(len(loops),2)
        updates=[]
        with progress_reporting(pipe,lambda step,total:updates.append((step,total))):
            for loop in loops:
                with loop.progress_bar(total=3) as bar:bar.update();bar.update(2)
        self.assertEqual(updates,[(0,3),(1,3),(3,3)]*2)
        self.assertTrue(all('progress_bar' not in vars(loop) for loop in loops))

    def test_shared_loop_nested_scope_and_exception_restore_previous_override(self):
        loop=AnimaDenoiseStep();loop.set_progress_bar_config(disable=True)
        original=loop.progress_bar;loop.progress_bar=original
        pipe=SimpleNamespace(_blocks=SimpleNamespace(sub_blocks={'first':loop,'shared':loop}))
        outer=[];inner=[]
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with progress_reporting(pipe,lambda *args:outer.append(args)):
                with progress_reporting(pipe,lambda *args:inner.append(args)):
                    with loop.progress_bar(total=2) as bar:bar.update()
                raise RuntimeError('stop')
        self.assertIs(loop.progress_bar,original)
        self.assertEqual(inner,[(0,2),(1,2)])
        self.assertEqual(outer,[(0,2),(1,2)])

    def test_persistent_pipeline_does_not_keep_previous_job_callback(self):
        loop=AnimaDenoiseStep();loop.set_progress_bar_config(disable=True)
        class Pipe:
            memory_policy={'family':'anima'}
            _blocks=loop
            def __call__(self,**arguments):
                self.arguments=arguments
                with loop.progress_bar(total=2) as bar:bar.update();bar.update()
                return 'image'
        pipe=Pipe();first=[];second=[]
        for updates in (first,second):
            self.assertEqual(pipeline_output(pipe,on_step=lambda *args:updates.append(args),prompt='same'),'image')
            self.assertEqual(pipe.arguments,{'prompt':'same'})
        self.assertEqual(first,[(0,2),(1,2),(2,2)])
        self.assertEqual(first,second)
        self.assertNotIn('progress_bar',vars(loop))
