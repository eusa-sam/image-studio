"""Disposable, per-model hash receipts for local Windows files."""
import json
import os
import re
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from .storage import encode,digest


def location(root,path):
    root=Path(root).resolve()
    key=digest(str(path).encode('utf-8'))
    target=root/key[:2]/(key+'.json')
    if not target.resolve().is_relative_to(root):raise ValueError('Receipt path escapes root')
    return target


def read_receipt(root,path,stamp):
    if root is None or os.name!='nt' or stamp is None:return None
    try:
        with location(root,path).open('rb') as stream:raw=stream.read(8193)
        if len(raw)>8192:return None
        row=json.loads(raw)
        if not isinstance(row,dict) or set(row)!={'schema_version','path','stamp','sha256','checksum'}:return None
        checksum=row.pop('checksum')
        if row['schema_version']!=1 or type(row['schema_version']) is not int or row['path']!=str(path):return None
        if row['stamp']!=list(stamp) or [type(v) for v in row['stamp']]!=[type(v) for v in stamp]:return None
        if not isinstance(row['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',row['sha256']):return None
        if digest(encode(row))!=checksum:return None
        return row['sha256']
    except (OSError,ValueError,TypeError):return None


def write_receipt(root,path,stamp,sha256):
    if root is None or os.name!='nt' or stamp is None:return
    temporary=None
    try:
        target=location(root,path);target.parent.mkdir(parents=True,exist_ok=True)
        row=dict(schema_version=1,path=str(path),stamp=list(stamp),sha256=sha256)
        row['checksum']=digest(encode(row))
        temporary=target.with_name(target.name+'.'+str(uuid4())+'.tmp')
        with temporary.open('xb') as stream:
            stream.write(encode(row));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
    except (OSError,ValueError):
        pass  # Receipts are optional; a verified image job must survive cache disk failure.
    finally:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass


@contextmanager
def hash_reader(path):
    if os.name!='nt':
        with path.open('rb') as stream:yield stream
        return
    import ctypes as c
    import msvcrt
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True)
    api.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
    api.CreateFileW.restype=w.HANDLE
    api.CloseHandle.argtypes=[w.HANDLE]
    api.CloseHandle.restype=w.BOOL
    # Deny writes and replacement while checking a receipt or reading its source.
    name=str(path)
    if len(name)>=260 and not name.startswith('\\\\?\\'):
        name='\\\\?\\UNC\\'+name[2:] if name.startswith('\\\\') else '\\\\?\\'+name
    handle=api.CreateFileW(name,0x80000000,1,None,3,0x08000000,None)
    if handle==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
    try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:
        api.CloseHandle(handle);raise
    with os.fdopen(descriptor,'rb') as stream:yield stream
