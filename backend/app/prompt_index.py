"""Disposable per-document previews; selected documents always use verified originals."""
from .storage import digest,encode,identifier


def summaries(store,session):
    with store._mutex:
        store._ready()
        base=store._session_path(session)
        store._read_json(base+'/session.json')
        scope=base+'/prompts';index='indexes/prompts/'+session
        rows=[];errors=[];cache_errors=[]
        for folder in sorted(store._path(scope).glob('*')):
            if not folder.is_dir():continue
            relative=folder.relative_to(store.root).as_posix()
            try:
                key=identifier(folder.name)
                head=store._read_json(relative+'/head.json')
                if head.get('id')!=key or type(head.get('revision')) is not int or head['revision']<1:
                    raise ValueError('Invalid prompt head')
                version=relative+f'/revisions/{head["revision"]:06d}'
                sources=[]
                # Preview validity depends on its source; opening still verifies every body file.
                for name in ('meta.json','positive.txt'):
                    stat=store._path(version+'/'+name).stat()
                    sources.append((name,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino))
                signature=digest(encode([head,sources]));path=index+'/'+key+'.json'
                try:
                    cached=store._read_json(path);checksum=cached.pop('sha256')
                    row=cached['item']
                    if digest(encode(cached))!=checksum or cached['schema_version']!=1 or cached['signature']!=signature:
                        raise ValueError('Stale prompt summary')
                    if row['id']!=key or row['revision']!=head['revision'] or not isinstance(row['title'],str) or len(row['title'])>60:
                        raise ValueError('Invalid prompt summary')
                except (OSError,ValueError,KeyError,TypeError):
                    document=store.read_prompt(session,key)
                    row=dict(id=key,revision=document['revision'],title=document['positive'][:60])
                    cached=dict(schema_version=1,signature=signature,item=row)
                    cached['sha256']=digest(encode(cached))
                    try:store._replace(path,encode(cached))
                    except (OSError,ValueError) as error:cache_errors.append(dict(path=path,error=str(error)))
                rows.append(row)
            except (OSError,ValueError,KeyError,TypeError) as error:
                errors.append(dict(path=relative,error=str(error)))
        store.read_errors[scope]=errors
        if cache_errors:store.read_errors[index]=cache_errors
        else:store.read_errors.pop(index,None)
        return rows
