import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from backend.app.backup import create_backup, restore_backup
from backend.app.storage import SessionStore


class BackupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows directory handle semantics')
    def test_locked_publish_keeps_complete_staging_and_retry_preserves_backup(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        rename = Path.rename
        def locked_rename(path, destination):
            # A reader without FILE_SHARE_DELETE prevents publishing the directory.
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x02000000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                return rename(path, destination)
            finally:
                kernel.CloseHandle(handle)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with SessionStore(root/'source') as store:
                session = store.create_session('locked restore')
                backup = store.root/'backups'/create_backup(store)['id']
                manifest = json.loads((backup/'manifest.json').read_text(encoding='utf-8'))
                originals = {row['path']: (backup/'files'/row['path']).read_bytes() for row in manifest['files']}
                destination = root/'restored'
                with patch.object(Path, 'rename', locked_rename):
                    with self.assertRaises(PermissionError) as failure:
                        restore_backup(backup, destination)
                self.assertIn(failure.exception.winerror, (5, 32))
                self.assertFalse(destination.exists())
                staging, = root.glob('restored.restore-*')
                restore_backup(backup, destination)
                for relative, content in originals.items():
                    for base in (store.root, backup/'files', staging, destination):
                        self.assertEqual((base/relative).read_bytes(), content)
                with SessionStore(destination) as restored:
                    self.assertEqual(restored.sessions(), store.sessions())

    def test_terminated_restore_keeps_unpublished_staging_and_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with SessionStore(root/'source') as store:
                store.create_session('survive process termination')
                backup=store.root/'backups'/create_backup(store)['id']
                destination=root/'interrupted'
                marker=root/'copied'
                script='''
import os, sys, time
from pathlib import Path
from unittest.mock import patch
from backend.app.backup import restore_backup
import shutil
copy=shutil.copyfileobj
def pause_after_copy(src,dst):
    copy(src,dst)
    dst.flush()
    os.fsync(dst.fileno())
    Path(sys.argv[3]).write_text('copied',encoding='utf-8')
    time.sleep(60)
with patch('backend.app.backup.shutil.copyfileobj',side_effect=pause_after_copy):
    restore_backup(sys.argv[1],sys.argv[2])
'''
                child=subprocess.Popen([sys.executable,'-c',script,str(backup),str(destination),str(marker)],
                                       stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                try:
                    deadline=time.monotonic()+15
                    while not marker.exists() and child.poll() is None and time.monotonic()<deadline:
                        time.sleep(.05)
                    self.assertTrue(marker.exists(),'restore child did not reach durable partial copy')
                    self.assertIsNone(child.poll())
                finally:
                    if child.poll() is None:child.terminate()
                    stdout,stderr=child.communicate(timeout=10)
                self.assertNotEqual(child.returncode,0,(stdout,stderr))
                self.assertFalse(destination.exists())
                staging=list(root.glob('interrupted.restore-*'))
                self.assertEqual(len(staging),1)
                manifest=json.loads((backup/'manifest.json').read_text(encoding='utf-8'))
                first=manifest['files'][0]['path']
                self.assertEqual((staging[0]/first).read_bytes(),(backup/'files'/first).read_bytes())
                restored=restore_backup(backup,destination)
                for row in manifest['files']:
                    self.assertEqual((restored/row['path']).read_bytes(),(store.root/row['path']).read_bytes())
                self.assertTrue(staging[0].is_dir())

    def test_restore_late_failures_preserve_source_and_allow_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with SessionStore(root/'source') as store:
                session=store.create_session('preserved')
                store.save_prompt(session,str(uuid4()),expected_revision=0,positive='original prompt',
                                  negative='',settings={'seed':42},request_id=str(uuid4()))
                backup=store.root/'backups'/create_backup(store)['id']
                manifest=json.loads((backup/'manifest.json').read_text(encoding='utf-8'))
                originals={row['path']:(backup/'files'/row['path']).read_bytes() for row in manifest['files']}
                self.assertGreater(len(originals),1)
                copy=shutil.copyfileobj
                for failure in ('disk_full','changed_copy','destination_appeared'):
                    with self.subTest(failure=failure):
                        destination=root/failure
                        calls=0
                        def fail_late(src,dst):
                            nonlocal calls
                            calls+=1
                            copy(src,dst)
                            if calls==2:
                                if failure=='disk_full':raise OSError('disk full after partial restore')
                                if failure=='changed_copy':dst.write(b'changed')
                                if failure=='destination_appeared':
                                    destination.mkdir()
                                    (destination/'keep.txt').write_bytes(b'existing user data')
                        expected=OSError if failure=='disk_full' else ValueError
                        with patch('backend.app.backup.shutil.copyfileobj',side_effect=fail_late):
                            with self.assertRaises(expected):restore_backup(backup,destination)
                        self.assertGreaterEqual(calls,2)
                        staging=list(root.glob(failure+'.restore-*'))
                        self.assertEqual(len(staging),1)
                        first=manifest['files'][0]['path']
                        self.assertEqual((staging[0]/first).read_bytes(),originals[first])
                        if failure=='destination_appeared':
                            self.assertEqual(list(destination.iterdir()),[destination/'keep.txt'])
                            self.assertEqual((destination/'keep.txt').read_bytes(),b'existing user data')
                        else:self.assertFalse(destination.exists())
                        restored=restore_backup(backup,root/(failure+'-retry'))
                        for relative,content in originals.items():
                            self.assertEqual((backup/'files'/relative).read_bytes(),content)
                            self.assertEqual((store.root/relative).read_bytes(),content)
                            self.assertEqual((restored/relative).read_bytes(),content)

    def test_unreadable_source_directory_never_publishes_partial_backup(self):
        with tempfile.TemporaryDirectory() as temporary,SessionStore(temporary) as store:
            session=store.create_session('must survive')
            source=store.root/'sessions'/session/'session.json'
            original=source.read_bytes()
            scandir=os.scandir
            for blocked in (store.root/'sessions',source.parent):
                def fail(path):
                    if Path(path)==blocked:raise PermissionError('source directory unreadable')
                    return scandir(path)
                with patch('backend.app.backup.os.scandir',side_effect=fail):
                    with self.assertRaisesRegex(PermissionError,'unreadable'):create_backup(store)
                self.assertEqual(list((store.root/'backups').glob('*/manifest.json')),[])
                self.assertEqual(source.read_bytes(),original)
            # Absent optional domains are valid; retry must include every source file.
            result=create_backup(store)
            manifest=json.loads((store.root/'backups'/result['id']/'manifest.json').read_text(encoding='utf-8'))
            self.assertIn(source.relative_to(store.root).as_posix(),[row['path'] for row in manifest['files']])

    def test_roundtrip_and_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with SessionStore(root/'source') as store:
                session=store.create_session('백업 테스트')
                prompt=str(uuid4())
                store.save_prompt(session,prompt,expected_revision=0,positive='한글\n여성 한 명',negative='',settings={'seed':42},request_id=str(uuid4()))
                result=create_backup(store)
                backup=store.root/'backups'/result['id']
                self.assertFalse(result['model_weights'])
                destination=restore_backup(backup,root/'restored')
                with SessionStore(destination) as restored:
                    self.assertEqual(restored.read_prompt(session,prompt),store.read_prompt(session,prompt))
                    self.assertEqual(restored.sessions(),store.sessions())
                with self.assertRaises(ValueError):restore_backup(backup,destination)
                self.assertFalse((destination/'transactions').exists())

    def test_corruption_version_paths_and_interruption_never_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with SessionStore(root/'source') as store:
                store.create_session('safe')
                backup=store.root/'backups'/create_backup(store)['id']
                manifest_path=backup/'manifest.json'
                original=manifest_path.read_text(encoding='utf-8')
                manifest=json.loads(original)
                for mutation in ('version','path','hash','duplicate'):
                    row=json.loads(original)
                    if mutation=='version':row['schema_version']=99
                    if mutation=='path':row['files'][0]['path']='../escape'
                    if mutation=='hash':row['files'][0]['sha256']='0'*64
                    if mutation=='duplicate':row['files'].append(row['files'][0])
                    manifest_path.write_text(json.dumps(row),encoding='utf-8')
                    with self.assertRaises(ValueError):restore_backup(backup,root/'bad')
                    self.assertFalse((root/'bad').exists())
                manifest_path.write_text(original,encoding='utf-8')
                with patch('backend.app.backup.shutil.copyfileobj',side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):restore_backup(backup,root/'interrupted')
                    with self.assertRaises(OSError):create_backup(store)
                self.assertFalse((root/'interrupted').exists())
                self.assertTrue(list(root.glob('interrupted.restore-*')))
                self.assertEqual(len(list((store.root/'backups').glob('*/manifest.json'))),1)
                self.assertEqual(store.sessions()[0]['title'],'safe')
                restore_backup(backup,root/'retry')

