"""Disposable Workspace message summaries; bodies remain in individual turn files."""
import json

from .storage import digest,encode,identifier


def page(store,session,*,limit=50,before=None):
    if type(limit) is not int or not 1<=limit<=200:raise ValueError('대화 조회는 1~200개입니다.')
    if before is not None:identifier(before)
    with store._mutex:
        store._ready()
        base=store._session_path(session)
        store._read_json(base+'/session.json')
        scope=base+'/messages'
        index='indexes/messages/'+session
        paths=sorted(p/'meta.json' for p in store._path(scope).glob('*') if p.is_dir())
        sources=[]
        for path in paths:
            relative=path.relative_to(store.root).as_posix()
            try:
                stat=store._path(relative).stat()
                sources.append((relative,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino))
            except (OSError,ValueError) as error:sources.append((relative,str(error)))
        signature=digest(encode(sources))
        def summary(path):
            record=store._read_json(path.relative_to(store.root).as_posix())
            if record['id']!=identifier(path.parent.name) or not isinstance(record['created_at'],str) or record['status'] not in ('PENDING','SUCCEEDED','FAILED','CANCELLED'):
                raise ValueError('Invalid chat metadata')
            return {key:record[key] for key in ('id','created_at','status')}
        try:
            head=store._read_json(index+'/head.json');checksum=head.pop('sha256')
            if digest(encode(head))!=checksum or head['signature']!=signature or not isinstance(head['pages'],list) or not isinstance(head['errors'],list):raise ValueError('Stale message index')
            rows=[]
            for number,entry in enumerate(head['pages'],1):
                if entry['path']!=f'{index}/page-{number:06d}.json':raise ValueError('Invalid message index page')
                content=store._path(entry['path']).read_bytes()
                if digest(content)!=entry['sha256']:raise ValueError('Corrupt message index page')
                data=json.loads(content)
                if data['schema_version']!=1 or not isinstance(data['items'],list) or not 1<=len(data['items'])<=200:raise ValueError('Invalid message index items')
                rows.extend(data['items'])
            for row in rows:
                identifier(row['id'])
                if not isinstance(row['created_at'],str) or row['status'] not in ('PENDING','SUCCEEDED','FAILED','CANCELLED'):raise ValueError('Invalid message summary')
            if len({row['id'] for row in rows})!=len(rows) or rows!=sorted(rows,key=lambda row:(row['created_at'],row['id'])):raise ValueError('Invalid message order')
            store.read_errors[scope]=head['errors']
            store.read_errors.pop(index,None)
        except (OSError,ValueError,KeyError,TypeError):
            rows=store._list_records(scope,paths,summary,True)
            rows.sort(key=lambda row:(row['created_at'],row['id']))
            try:
                pages=[]
                for offset in range(0,len(rows),200):
                    path=f'{index}/page-{offset//200+1:06d}.json'
                    content=encode(dict(schema_version=1,items=rows[offset:offset+200]))
                    store._replace(path,content);pages.append(dict(path=path,sha256=digest(content)))
                head=dict(schema_version=1,signature=signature,pages=pages,errors=store.read_errors[scope])
                head['sha256']=digest(encode(head));store._replace(index+'/head.json',encode(head))
                store.read_errors.pop(index,None)
            except (OSError,ValueError) as error:store.read_errors[index]=[dict(path=index,error=str(error))]
        end=len(rows)
        if before is not None:
            end=next((i for i,row in enumerate(rows) if row['id']==before),-1)
            if end<0:raise ValueError('이 Workspace에서 이전 대화 기준을 찾을 수 없습니다.')
        start=max(0,end-limit)
        selected=rows[start:end]
        items=[]
        errors=list(store.read_errors.get(scope,[]))
        for row in selected:
            try:items.append(store._chat(session,row['id'],include_application=True))
            except (OSError,ValueError,KeyError,TypeError) as error:errors.append(dict(path=scope+'/'+row['id'],error=str(error)))
        store.read_errors[scope]=errors
        return dict(items=items,before=selected[0]['id'] if start else None,has_more=start>0)
