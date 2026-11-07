"""Backfill legacy artifact hashes only from matching committed transaction evidence."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .backup import checksum, create_backup
from .storage import SessionStore, ARTIFACT_NAMES, encode, identifier


def migrate(store, *, apply=False):
    with store._mutex:
        store._ready()
        evidence = {}
        for marker in store._path('transactions').glob('*/*/COMMITTED.json'):
            base = marker.parent.relative_to(store.root).as_posix()
            manifest = store._read_json(base + '/manifest.json')
            for index, row in enumerate(manifest['files']):
                if '/artifacts/' not in row['path']:
                    continue
                store._path(row['path'])
                payload = store._path(base + '/payload/' + str(index))
                if checksum(payload) != row['after']:
                    raise ValueError('Corrupt artifact transaction evidence: ' + base)
                evidence.setdefault(row['path'], set()).add(row['after'])
        report = dict(schema_version=1, id=str(uuid4()), apply=apply,
                      created_at=datetime.now(timezone.utc).isoformat(), eligible=[], skipped=[], errors=[], migrated=[])
        changes = []
        for path in sorted(store._path('sessions').glob('*/*/jobs/*/state.json')):
            relative = path.relative_to(store.root).as_posix()
            try:
                state = store._read_json(relative)
                if state['id'] != identifier(path.parent.name):
                    raise ValueError('Job identity mismatch')
                if 'artifact_hashes' in state:
                    report['skipped'].append(relative)
                    continue
                if state['status'] not in ('SUCCEEDED', 'PARTIAL_FAILED', 'CANCELLED', 'INTERRUPTED'):
                    report['skipped'].append(relative)
                    continue
                names = sorted(p.name for p in (path.parent/'artifacts').iterdir()) if (path.parent/'artifacts').exists() else []
                if not names:
                    report['skipped'].append(relative)
                    continue
                if set(state.get('artifacts', [])) - set(names):
                    raise ValueError('Recorded artifact is missing')
                hashes = {}
                for name in names:
                    if name not in ARTIFACT_NAMES:
                        raise ValueError('Unknown artifact name: ' + name)
                    artifact = path.parent/'artifacts'/name
                    key = artifact.relative_to(store.root).as_posix()
                    expected = evidence.get(key, set())
                    if len(expected) != 1 or checksum(store._path(key)) not in expected:
                        raise ValueError('Missing, conflicting or mismatched commit evidence: ' + name)
                    hashes[name] = next(iter(expected))
                if type(state['revision']) is not int or state['revision'] < 1:
                    raise ValueError('Invalid job revision')
                state.update(artifact_hashes=hashes, revision=state['revision']+1,
                             updated_at=report['created_at'])
                changes.append((relative, state))
                report['eligible'].append(relative)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report['errors'].append(dict(path=relative, error=str(exc)))
        if apply and changes:
            report['backup'] = create_backup(store)
            progress = 'migrations/artifact-hashes/' + report['id'] + '.json'
            store._replace(progress, encode(report))
            for relative, state in changes:
                # Each state is an independent recoverable commit; reruns skip completed records.
                store._commit({relative: encode(state)})
                report['migrated'].append(relative)
                store._replace(progress, encode(report))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--apply', action='store_true', help='Back up first, then migrate verified legacy states.')
    args = parser.parse_args()
    with SessionStore(args.root) as store:
        print(json.dumps(migrate(store, apply=args.apply), ensure_ascii=False, indent=2))
