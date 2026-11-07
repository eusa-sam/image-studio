"""Verify revision-pinned local bundles against their committed source manifests."""
import hashlib
import json
import re
from pathlib import Path


def verify_bundle(folder, specification, label, check_hashes=True, cache=None):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'download_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest,dict):raise ValueError(f'Invalid {label} manifest')
    if any(manifest.get(key)!=specification[key] for key in ('repository','revision')):
        raise ValueError(f'Unsupported {label} bundle revision')
    if not isinstance(manifest.get('files'),list):raise ValueError(f'Invalid {label} manifest files')
    expected={row['path']:row for row in specification['files']}
    seen=set()
    for row in manifest['files']:
        if not isinstance(row,dict) or not isinstance(row.get('path'),str):
            raise ValueError(f'Invalid {label} manifest file')
        name=row['path'];path=(folder/name).resolve()
        if name in seen or name not in expected or not path.is_relative_to(folder):
            raise ValueError(f'Invalid or duplicate {label} bundle path')
        seen.add(name)
        spec=expected[name]
        if not path.is_file() or path.stat().st_size!=spec['size']:
            raise ValueError(f'Missing or incomplete {label} file: '+name)
        if not isinstance(row.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',row['sha256']):
            raise ValueError(f'Invalid {label} SHA256')
        if spec['sha256'] and row['sha256']!=spec['sha256']:
            raise ValueError(f'{label} source hash mismatch: '+name)
        if check_hashes:
            if spec['sha256'] and cache is not None:
                if cache.file_hash(path)!=row['sha256']:raise ValueError(f'{label} file hash mismatch: '+name)
                continue
            sha=hashlib.sha256()
            blob=hashlib.sha1(f"blob {spec['size']}\0".encode())
            with path.open('rb') as stream:
                while block:=stream.read(8*1024*1024):
                    sha.update(block)
                    if not spec['sha256']:blob.update(block)
            if sha.hexdigest()!=row['sha256'] or (not spec['sha256'] and blob.hexdigest()!=spec['blob_id']):
                raise ValueError(f'{label} file hash mismatch: '+name)
    if seen!=set(expected):raise ValueError(f'Incomplete {label} bundle')
    return manifest


def bundle_fingerprint(folder, specification, label):
    folder=Path(folder)
    manifest=verify_bundle(folder,specification,label,check_hashes=False)
    files={}
    for name in ['download_manifest.json']+[row['path'] for row in manifest['files']]:
        stat=(folder/name).stat()
        files[name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return dict(files=files)


