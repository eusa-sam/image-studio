import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from backend.app.pipeline_cache import PipelineCache
from backend.app.verification_receipts import location,read_receipt,hash_reader


@unittest.skipUnless(os.name=='nt','Windows file identity receipts')
class VerificationReceiptTests(unittest.TestCase):
    def test_new_process_reuses_verified_digest_without_reading_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'first');receipts=root/'receipts'
            expected=PipelineCache(receipts).file_hash(model)
            code="from pathlib import Path;from unittest.mock import patch;from backend.app.pipeline_cache import PipelineCache;import sys;cache=PipelineCache(Path(sys.argv[2]));guard=patch('hashlib.file_digest',side_effect=AssertionError('unexpected source read'));guard.start();print(cache.file_hash(Path(sys.argv[1])));assert cache.hash_checks[-1]['source']=='receipt'"
            result=subprocess.run([sys.executable,'-c',code,str(model),str(receipts)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.splitlines()[-1],expected)

    def test_edit_with_restored_mtime_and_corrupt_receipts_force_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'first');receipts=root/'receipts'
            PipelineCache(receipts).file_hash(model);original=model.stat()
            model.write_bytes(b'other');os.utime(model,ns=(original.st_atime_ns,original.st_mtime_ns))
            cache=PipelineCache(receipts)
            self.assertEqual(cache.file_hash(model),hashlib.sha256(b'other').hexdigest())
            self.assertEqual(cache.hash_checks[-1]['source'],'read')
            target=location(receipts,model.resolve());valid=target.read_bytes()
            changed=json.loads(valid);changed['sha256']='0'*64
            wrong_path=json.loads(valid);wrong_path['path']='elsewhere'
            for bad in (b'{',b'x'*8193,json.dumps(changed).encode(),json.dumps(wrong_path).encode()):
                target.write_bytes(bad)
                cache=PipelineCache(receipts)
                self.assertEqual(cache.file_hash(model),hashlib.sha256(b'other').hexdigest())
                self.assertEqual(cache.hash_checks[-1]['source'],'read')

    def test_cache_disk_failure_preserves_digest_and_leaves_no_partial_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'first');receipts=root/'receipts'
            with patch('backend.app.verification_receipts.os.replace',side_effect=OSError('disk full')):
                self.assertEqual(PipelineCache(receipts).file_hash(model),hashlib.sha256(b'first').hexdigest())
            self.assertFalse(list(receipts.rglob('*.tmp')))
            self.assertFalse(list(receipts.rglob('*.json')))

    def test_validation_lock_blocks_write_and_existing_writer_blocks_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'first');receipts=root/'receipts'
            PipelineCache(receipts).file_hash(model)
            def checked(*args):
                with self.assertRaises(OSError):model.write_bytes(b'other')
                return read_receipt(*args)
            with patch('backend.app.verification_receipts.read_receipt',side_effect=checked):
                self.assertEqual(PipelineCache(receipts).file_hash(model),hashlib.sha256(b'first').hexdigest())
            with model.open('r+b') as writer:
                with self.assertRaises(OSError):PipelineCache(receipts).file_hash(model)
            self.assertEqual(model.read_bytes(),b'first')

    def test_unavailable_stamp_does_not_read_or_publish_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'first');receipts=root/'receipts'
            PipelineCache(receipts).file_hash(model)
            with patch('backend.app.pipeline_cache.file_stamp',return_value=None),patch('hashlib.file_digest',wraps=hashlib.file_digest) as digest:
                cache=PipelineCache(receipts);cache.file_hash(model);cache.file_hash(model)
                self.assertEqual(digest.call_count,2)
                self.assertTrue(all(row['source']=='read' for row in cache.hash_checks))

    def test_long_local_model_path_preserves_hash_and_receipt_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root.joinpath(*(['nested-'+'x'*45]*6));folder.mkdir(parents=True)
            model=folder/'model.safetensors';model.write_bytes(b'first')
            self.assertGreater(len(str(model)),260)
            receipts=root/'receipts';expected=hashlib.sha256(b'first').hexdigest()
            self.assertEqual(PipelineCache(receipts).file_hash(model),expected)
            cache=PipelineCache(receipts);self.assertEqual(cache.file_hash(model),expected)
            self.assertEqual(cache.hash_checks[-1]['source'],'receipt')
