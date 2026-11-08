import hashlib
from unittest.mock import patch
import copy,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from backend.app.inference.runtime.pipeline_cache import PipelineCache
from backend.app.models import ROOT


class PipelineCacheTests(unittest.TestCase):
    def test_krea_strength_change_waits_then_discards_only_derived_weights(self):
        from types import SimpleNamespace
        events=[];original=object()
        mapping=SimpleNamespace(derived_identity='old',derived_cpu_cache={'old':object()},
            derived_gpu_cache={'old':object()},transfer_following={'old':'next'},cache={'base':original})
        cache=PipelineCache();cache.key=cache.identity(dict(runner='krea2_txt2img'),(1024,1024))
        cache.pipe=SimpleNamespace(memory_policy=dict(family='krea2',mode='partial'),weight_maps=[mapping],
            _weight_transfer=SimpleNamespace(close=lambda:events.append(bool(mapping.derived_gpu_cache))))
        cache.positive='text';cache.conditioning={'text':{'saved':object()}};cache.lora_weights=(1.,)
        self.assertIs(cache.get(cache.key,'text',replace_loras=[(dict(weight=.5),'adapter')]),cache.pipe)
        self.assertEqual(events,[True]);self.assertFalse(mapping.derived_cpu_cache);self.assertFalse(mapping.derived_gpu_cache)
        self.assertIs(mapping.cache['base'],original);self.assertIn('text',cache.conditioning)

    def test_failed_cleanup_cannot_keep_a_reusable_pipeline(self):
        cache=PipelineCache();cache.pipe=object();cache.key='old';cache.conditioning={'old':1}
        with patch('backend.app.inference.runtime.execution.release',side_effect=ValueError('pending transfer')):
            with self.assertRaisesRegex(ValueError,'pending transfer'):cache.clear()
        self.assertIsNone(cache.pipe);self.assertIsNone(cache.key);self.assertFalse(cache.conditioning)

    def test_krea_adapter_replacement_preserves_pipeline_and_text_conditions(self):
        from types import SimpleNamespace
        cache=PipelineCache();plan=dict(runner='krea2_txt2img',loras=[dict(path='old',weight=1)])
        cache.key=cache.identity(plan,(1024,1024))
        cache.pipe=SimpleNamespace(memory_policy=dict(family='krea2',mode='partial'))
        cache.positive='text';cache.conditioning={'text':{'saved':object()}}
        key=cache.identity(dict(plan,loras=[dict(path='new',weight=.5)]),(1024,1024))
        selections=[(dict(weight=.5),'new')];pipe=cache.pipe
        with patch('backend.app.inference.models.krea2_lora.replace_loras') as replace:
            self.assertIs(cache.get(key,'text',replace_loras=selections),pipe)
        replace.assert_called_once_with(pipe,selections)
        self.assertIn('text',cache.conditioning)
        self.assertEqual(cache.lora_weights,(.5,))

    def test_hash_reuses_only_unchanged_file_and_rejects_racing_change(self):
        from unittest.mock import patch
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weight';path.write_bytes(b'first')
            cache=PipelineCache()
            with patch('hashlib.file_digest',wraps=hashlib.file_digest) as digest:
                self.assertEqual(cache.file_hash(path),hashlib.sha256(b'first').hexdigest())
                cache.file_hash(path);self.assertEqual(digest.call_count,1)
                path.write_bytes(b'changed');cache.file_hash(path);self.assertEqual(digest.call_count,2)
            self.assertEqual([row['reused'] for row in cache.hash_checks],[False,True,False])
            self.assertEqual([row['bytes'] for row in cache.hash_checks],[5,5,7])
            self.assertTrue(all(row['seconds']>=0 and row['cpu_seconds']>=0 for row in cache.hash_checks))
            cache.verified.clear()
            original=hashlib.file_digest
            def racing(stream,algorithm):
                result=original(stream,algorithm);path.write_bytes(b'changed again');return result
            with patch('hashlib.file_digest',side_effect=racing),self.assertRaises((ValueError,OSError)):
                cache.file_hash(path)
            self.assertEqual(len(cache.hash_checks),3)

    def test_worker_resets_only_per_request_hash_measurements(self):
        from unittest.mock import patch
        from backend.app.worker import run
        with tempfile.TemporaryDirectory() as folder:
            for runner in ('flux2_txt2img','zimage_txt2img','sdxl_txt2img','sd1_img2img','sd2_inpaint'):
                path=Path(folder)/'request.json';path.write_text(json.dumps({'plan':{'runner':runner}}))
                cache=PipelineCache();cache.hash_checks.append({'previous':True});cache.verified['old']='retained'
                with patch('backend.app.worker._run') as execute:
                    run(path,cache)
                execute.assert_called_once_with(path,cache)
                self.assertEqual(cache.hash_checks,[])
                self.assertEqual(cache.verified,{'old':'retained'})

    def test_identity_ignores_sampling_but_invalidates_model_components_and_lora(self):
        plan=dict(model_path='a',fingerprint={'bytes':1},components={'clip':{'path':'b'}},loras=[{'scale':1}],settings={'seed':1})
        key=PipelineCache.identity(plan,(768,1024))
        changed=copy.deepcopy(plan);changed['settings']['seed']=2
        self.assertEqual(key,PipelineCache.identity(changed,(768,1024)))
        changed['components']['upscaler']={'path':'upscaler'}
        self.assertEqual(key,PipelineCache.identity(changed,(768,1024)))
        for field,value in [('model_path','c'),('components',{}),('loras',[{'scale':.5}]),('fingerprint',{'bytes':2})]:
            changed=copy.deepcopy(plan);changed[field]=value
            self.assertNotEqual(key,PipelineCache.identity(changed,(768,1024)))
        self.assertNotEqual(key,PipelineCache.identity(plan,(1024,1024)))

    def test_prompt_cache_is_bounded_and_pipeline_changes_release_references(self):
        cache=PipelineCache();cache.key='a';cache.pipe=object();cache.positive='one';cache.conditioning['one']=1
        pipe=cache.get('a','one');self.assertEqual(cache.conditioning,{'one':1})
        self.assertIs(cache.get('a','two'),pipe);self.assertEqual(cache.conditioning,{})
        self.assertIsNone(cache.get('b','two'));self.assertIsNone(cache.key)

    def test_sd_identity_separates_prediction_and_dedicated_inpaint_but_not_reference_strength(self):
        plan=dict(runner='sdxl_txt2img', settings=dict(prediction_type='epsilon',ip_adapter_scale=.7))
        key=PipelineCache.identity(plan,(768,1024))
        changed=copy.deepcopy(plan);changed['settings']['ip_adapter_scale']=.2
        self.assertEqual(key,PipelineCache.identity(changed,(768,1024)))
        for field,value in [('inpaint_channels',9),('vision_config_sha256','changed')]:
            changed=copy.deepcopy(plan);changed[field]=value
            self.assertNotEqual(key,PipelineCache.identity(changed,(768,1024)))
        changed=copy.deepcopy(plan);changed['settings']['prediction_type']='v_prediction'
        self.assertNotEqual(key,PipelineCache.identity(changed,(768,1024)))

    def test_sd_retention_keeps_models_only_with_ram_headroom(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        for family in ('sd1','sd2','sdxl'):
            with self.subTest(family=family):
                pipe=SimpleNamespace(memory_policy=dict(family=family,mode='full'),
                    components={'unet':torch.nn.Linear(2,2)},remove_all_hooks=Mock(),to=Mock())
                cache=PipelineCache();cache.pipe=pipe;cache.key='same'
                with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=16*1024**3),patch('torch.cuda.empty_cache'):
                    self.assertTrue(cache.suspend())
                self.assertIs(cache.pipe,pipe);self.assertTrue(cache.suspended)
                pipe.to.assert_called_once_with('cpu');pipe.remove_all_hooks.assert_called_once()
                with patch('backend.app.inference.runtime.memory_policy.available_ram',return_value=0),patch('torch.cuda.empty_cache'):
                    self.assertFalse(cache.suspend())
                self.assertIsNone(cache.pipe);self.assertIsNone(cache.key)

    def test_lora_strength_reuses_only_identical_ordered_sources(self):
        for runner in ('sdxl_txt2img','sd1_img2img','sd2_inpaint','flux2_txt2img','flux2_reference_edit','zimage_txt2img'):
            plan=dict(runner=runner,loras=[dict(path='a',weight=0,fingerprint='first'),dict(path='b',weight=.7)])
            key=PipelineCache.identity(plan,(768,1024))
            changed=copy.deepcopy(plan);changed['loras'][0]['weight']=1
            self.assertEqual(key,PipelineCache.identity(changed,(768,1024)))
            self.assertEqual(plan['loras'][0]['weight'],0)
            for rows in (list(reversed(plan['loras'])),plan['loras'][:1],[],[dict(path='a',weight=0,fingerprint='changed'),plan['loras'][1]]):
                changed=copy.deepcopy(plan);changed['loras']=rows
                self.assertNotEqual(key,PipelineCache.identity(changed,(768,1024)))
        plan['runner']='krea2_txt2img';changed=copy.deepcopy(plan);changed['loras'][0]['weight']=1
        self.assertEqual(PipelineCache.identity(plan,(768,1024)),PipelineCache.identity(changed,(768,1024)))

    def test_service_reuses_cache_in_custom_store_and_exits_on_parent_eof(self):
        runtime=ROOT/'data/runtime/test-custom-store/runtime';(runtime/'jobs').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as service, tempfile.TemporaryDirectory(dir=runtime/'jobs') as first, tempfile.TemporaryDirectory(dir=runtime/'jobs') as second:
            code="""
import json,sys
from pathlib import Path
from backend.app import worker
from backend.app.inference.runtime.pipeline_cache import PipelineCache
PipelineCache.suspend=lambda self:True
def fake(path,cache):
    cache.pipe=(cache.pipe or 0)+1
    cache.suspended=cache.pipe%2==0
    Path(path).with_name('result.json').write_text(json.dumps({'calls':cache.pipe,'receipts':str(cache.receipt_root)}))
worker.run=fake
from backend.app.worker_service import serve
serve(sys.argv[1],sys.argv[2])
"""
            process=subprocess.Popen([sys.executable,'-c',code,service,str(runtime.parent)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                for index,folder in enumerate([first,second],1):
                    request=Path(folder)/'request.json';request.write_text('{}')
                    process.stdin.write((json.dumps(str(request))+'\n').encode());process.stdin.flush()
                    deadline=time.monotonic()+20
                    while not request.with_name('worker-done.json').exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                    self.assertTrue(request.with_name('worker-done.json').exists())
                    self.assertEqual(json.loads(request.with_name('worker-done.json').read_text())['cpu_cached'],index%2==0)
                    self.assertEqual(json.loads(request.with_name('result.json').read_text())['calls'],index)
                    self.assertEqual(Path(json.loads(request.with_name('result.json').read_text())['receipts']),runtime.parent/'indexes/model-verification')
                    token='a'*32
                    acknowledgement=Path(service)/('suspend-'+token+'.json')
                    if acknowledgement.exists():acknowledgement.unlink()
                    process.stdin.write((json.dumps(dict(command='suspend',token=token))+'\n').encode());process.stdin.flush()
                    deadline=time.monotonic()+20
                    while not acknowledgement.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                    self.assertEqual(json.loads(acknowledgement.read_text()),{'code':0,'cached':True})
                process.stdin.close();self.assertEqual(process.wait(timeout=10),0)
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=10)
                if not process.stdin.closed:process.stdin.close()
                process.stderr.close()


class FileChangeStampTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').name=='nt','Windows file journal')
    def test_recent_large_file_without_journal_cannot_authorize_receipt(self):
        from backend.app.inference.runtime.pipeline_cache import file_stamp
        import time
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'x'*(1024*1024+1))
            now=time.time_ns()
            with patch('backend.app.inference.runtime.pipeline_cache.time.time_ns',return_value=now+3_000_000_000):
                settled=file_stamp(path)
            self.assertIsNotNone(settled)
            if settled[-1]==0:self.assertIsNone(file_stamp(path))

    def test_rapid_edits_with_restored_mtime_never_reuse_an_old_digest(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'0000')
            original=path.stat();cache=PipelineCache(Path(directory)/'receipts')
            for index in range(100):
                value=f'{index:04d}'.encode();path.write_bytes(value)
                os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
                self.assertEqual(cache.file_hash(path),hashlib.sha256(value).hexdigest())
                self.assertEqual(PipelineCache(cache.receipt_root).file_hash(path),hashlib.sha256(value).hexdigest())

    def test_same_size_edit_with_restored_mtime_invalidates_cached_hash(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'first')
            original=path.stat();cache=PipelineCache()
            self.assertEqual(cache.file_hash(path),hashlib.sha256(b'first').hexdigest())
            path.write_bytes(b'other')
            os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
            self.assertEqual(path.stat().st_size,original.st_size)
            self.assertEqual(path.stat().st_mtime_ns,original.st_mtime_ns)
            self.assertEqual(cache.file_hash(path),hashlib.sha256(b'other').hexdigest())
            self.assertFalse(cache.hash_checks[-1]['reused'])
            cache.file_hash(path)
            self.assertTrue(cache.hash_checks[-1]['reused'])

    def test_unavailable_stamp_never_reuses_or_retains_a_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'first');cache=PipelineCache()
            cache.file_hash(path)
            with patch('backend.app.inference.runtime.pipeline_cache.file_stamp',return_value=None),patch('hashlib.file_digest',wraps=hashlib.file_digest) as digest:
                cache.file_hash(path);cache.file_hash(path)
                self.assertEqual(digest.call_count,2)
                self.assertNotIn(path.resolve(),cache.verified)
                self.assertTrue(all(not row['reused'] for row in cache.hash_checks[-2:]))

    def test_change_stamp_mutation_during_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'first');cache=PipelineCache()
            with patch('backend.app.inference.runtime.pipeline_cache.file_stamp',side_effect=[(1,),(2,)]):
                with self.assertRaisesRegex(ValueError,'COMPONENT_CHANGED'):cache.file_hash(path)
            self.assertFalse(cache.verified)

    def test_same_size_mtime_replacement_invalidates_by_file_identity(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'first');cache=PipelineCache()
            cache.file_hash(path);original=path.stat()
            replacement=Path(directory)/'new';replacement.write_bytes(b'other')
            os.utime(replacement,ns=(original.st_atime_ns,original.st_mtime_ns))
            path.rename(Path(directory)/'backup');replacement.rename(path)
            self.assertEqual(cache.file_hash(path),hashlib.sha256(b'other').hexdigest())
            self.assertFalse(cache.hash_checks[-1]['reused'])
