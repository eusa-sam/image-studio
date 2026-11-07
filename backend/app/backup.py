"""Consistent file snapshots; restoration never overwrites an existing data root."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from datetime import datetime, timezone
from uuid import uuid4

from .storage import encode

DOMAINS = ('sessions', 'settings', 'presets', 'models')


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or parts.as_posix()!=relative or parts.parts[0] not in DOMAINS or '..' in parts.parts or '\\' in relative or ':' in relative:
        raise ValueError('Invalid backup path')
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Backup path escapes root')
    for parent in (path, *path.parents):
        if parent.is_symlink() or parent.is_junction():
            raise ValueError('Backup links are not supported')
        if parent == root:break
    return path


def create_backup(store):
    with store._mutex:
        store._ready()
        backup_id = str(uuid4())
        base = store._path('backups/' + backup_id)
        base.mkdir(parents=True)
        rows = []
        def unreadable(error):
            raise error
        # Source originals stay locked until the completion manifest is durable.
        for domain in DOMAINS:
            source = safe_path(store.root, domain)
            try:source.stat()
            except FileNotFoundError:continue
            for directory, dirs, files in os.walk(source, followlinks=False, onerror=unreadable):
                for name in dirs + files:
                    path = Path(directory) / name
                    relative = path.relative_to(store.root).as_posix()
                    safe_path(store.root, relative)
                for name in sorted(files):
                    path = Path(directory) / name
                    relative = path.relative_to(store.root).as_posix()
                    target = safe_path(base / 'files', relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('rb') as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush();os.fsync(dst.fileno())
                    rows.append(dict(path=relative, bytes=target.stat().st_size, sha256=checksum(target)))
        manifest = dict(schema_version=1, id=backup_id, created_at=datetime.now(timezone.utc).isoformat(),
                        model_weights=False, files=sorted(rows, key=lambda row:row['path']))
        store._replace('backups/' + backup_id + '/manifest.json', encode(manifest))
        return dict(id=backup_id, files=len(rows), bytes=sum(row['bytes'] for row in rows), model_weights=False)


def restore_backup(backup, destination):
    backup, destination = Path(backup).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Restore requires a new destination directory')
    manifest = json.loads((backup / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), list):
        raise ValueError('Unsupported backup manifest')
    seen = set()
    for row in manifest['files']:
        source = safe_path(backup / 'files', row['path'])
        key = row['path'].casefold()
        if key in seen or not source.is_file() or source.stat().st_size != row['bytes'] or checksum(source) != row['sha256']:
            raise ValueError('Duplicate or corrupt backup file')
        seen.add(key)
    staging = destination.with_name(destination.name + '.restore-' + str(uuid4()))
    staging.mkdir(parents=True)
    # Failed staging is retained for diagnosis; it is never exposed as a restored root.
    for row in manifest['files']:
        source = safe_path(backup / 'files', row['path'])
        target = safe_path(staging, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush();os.fsync(dst.fileno())
        if checksum(target) != row['sha256']:
            raise ValueError('Backup changed during restore')
    if destination.exists():raise ValueError('Restore destination appeared during copy')
    staging.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restore a backup into a new, unused data directory.')
    parser.add_argument('backup');parser.add_argument('destination')
    args = parser.parse_args()
    print(restore_backup(args.backup, args.destination))
