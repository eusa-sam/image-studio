"""Disposable month-sharded Workspace summaries; session.json is authoritative."""
from collections import defaultdict
import json
import re

from .storage import digest, encode


def workspaces(store):
    with store._mutex:
        store._ready()
        sources=[]
        for folder in sorted(store._path('sessions').glob('*/*')):
            if not folder.is_dir():continue
            relative=(folder/'session.json').relative_to(store.root).as_posix()
            try:
                stat=store._path(relative).stat()
                sources.append((relative,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino))
            except (ValueError,OSError) as error:sources.append((relative,str(error)))
        signature=digest(encode(sources))
        try:
            head=store._read_json('indexes/workspaces/head.json')
            checksum=head.pop('sha256')
            if digest(encode(head))!=checksum:raise ValueError('Corrupt Workspace index header')
            if head['signature']!=signature or not isinstance(head['pages'],list) or not isinstance(head['errors'],list):
                raise ValueError('Stale Workspace index')
            rows=[]
            for page in head['pages']:
                if not re.fullmatch(r'indexes/workspaces/\d{4}-(?:0[1-9]|1[0-2])/page-\d{6}\.json',page['path']):
                    raise ValueError('Invalid Workspace page')
                content=store._path(page['path']).read_bytes()
                if digest(content)!=page['sha256']:raise ValueError('Corrupt Workspace page')
                data=json.loads(content)
                if data['schema_version']!=1 or not isinstance(data['items'],list) or len(data['items'])>200:
                    raise ValueError('Invalid Workspace page')
                rows.extend(data['items'])
            for row in rows:
                if not isinstance(row,dict) or not isinstance(row.get('ref'),str):raise ValueError('Invalid Workspace summary')
                store._session_path(row['ref'])
                if row['id']!=row['ref'].split('/')[1] or not isinstance(row['title'],str) or not isinstance(row['created_at'],str) or type(row['revision']) is not int:
                    raise ValueError('Invalid Workspace summary')
            store.read_errors['sessions']=head['errors']
            store.read_errors.pop('workspace_index',None)
            return sorted(rows,key=lambda row:row['ref'])
        except (ValueError,OSError,KeyError,TypeError):
            rows=store.sessions(tolerate_errors=True)
        months=defaultdict(list)
        for row in rows:months[row['ref'].split('/')[0]].append(row)
        try:
            pages=[]
            for month,items in sorted(months.items()):
                for offset in range(0,len(items),200):
                    path=f'indexes/workspaces/{month}/page-{offset//200+1:06d}.json'
                    content=encode(dict(schema_version=1,items=items[offset:offset+200]))
                    store._replace(path,content)
                    pages.append(dict(path=path,sha256=digest(content)))
            head=dict(schema_version=1,signature=signature,pages=pages,errors=store.read_errors['sessions'])
            head['sha256']=digest(encode(head))
            store._replace('indexes/workspaces/head.json',encode(head))
            store.read_errors.pop('workspace_index',None)
        except (OSError,ValueError) as error:
            # The disposable index must not hide healthy originals when the disk is full.
            store.read_errors['workspace_index']=[dict(path='indexes/workspaces',error=str(error))]
        return rows
