"""Conversation metadata; existing message files stay in their original locations."""
from datetime import datetime, timezone
from uuid import uuid4, uuid5, NAMESPACE_URL

from .storage import ConflictError, encode, identifier


def default_id(session):
    return str(uuid5(NAMESPACE_URL, 'image-studio:conversation:' + session))


def read(store, session, conversation_id=None):
    base = store._session_path(session)
    workspace = store._read_json(base + '/session.json')
    cid = identifier(conversation_id or default_id(session))
    path = base + '/conversations/' + cid + '/meta.json'
    if cid == default_id(session) and not store._path(path).exists():
        return dict(schema_version=1,id=cid, title='기존 대화', revision=0, archived=False,
                    created_at=workspace['created_at'], updated_at=workspace['created_at'], automatic_title=False)
    row = store._read_json(path)
    if row['id'] != cid or not isinstance(row['title'], str) or type(row['archived']) is not bool:
        raise ValueError('잘못된 채팅 메타데이터입니다.')
    return row


def listing(store, session, include_archived=False):
    with store._mutex:
        store._ready()
        base = store._session_path(session)
        scope = base + '/conversations'
        rows = store._list_records(scope, store._path(scope).glob('*/meta.json'),
                                   lambda p: read(store, session, p.parent.name), True)
        if not any(row['id'] == default_id(session) for row in rows):
            rows.append(read(store, session))
        # Only metadata is needed to order the list, never message bodies.
        for path in store._path(base + '/messages').glob('*/meta.json'):
            try:
                message = store._read_json(path.relative_to(store.root).as_posix())
                cid = message.get('conversation_id') or default_id(session)
                for row in rows:
                    if row['id'] == cid:
                        row['updated_at'] = max(row['updated_at'], message.get('finished_at') or message['created_at'])
            except (OSError, ValueError, KeyError, TypeError):
                continue  # Message listing owns corruption diagnostics.
        return sorted((r for r in rows if include_archived or not r['archived']),
                      key=lambda r: (r['updated_at'], r['id']), reverse=True)


def create(store, session):
    with store._mutex:
        store._ready()
        read(store, session)
        now = datetime.now(timezone.utc).isoformat()
        row = dict(schema_version=1,id=str(uuid4()), title='새 채팅', revision=1, archived=False,
                   created_at=now, updated_at=now, automatic_title=True)
        store._commit({store._session_path(session) + '/conversations/' + row['id'] + '/meta.json': encode(row)})
        return row


def update(store, session, cid, revision, title, archived):
    title = title.strip()
    if not title or len(title) > 200 or type(archived) is not bool:
        raise ValueError('채팅 이름은 1~200자입니다.')
    with store._mutex:
        store._ready()
        row = read(store, session, cid)
        if row['revision'] != revision:
            raise ConflictError('다른 창에서 채팅이 변경됐습니다. 목록을 새로 불러오세요.')
        if archived and any(t['status'] == 'PENDING' for t in store.chats(session, conversation_id=cid)):
            raise ConflictError('응답을 완료하거나 중단한 뒤 보관하세요.')
        row.update(title=title, archived=archived, revision=revision + 1, automatic_title=False)
        store._commit({store._session_path(session) + '/conversations/' + cid + '/meta.json': encode(row)})
        return row


def begin_changes(store, session, payload):
    row = read(store, session, payload.get('conversation_id'))
    if row['archived']:
        raise ConflictError('보관한 채팅은 복원한 뒤 메시지를 보내세요.')
    if not row.get('automatic_title'):
        return {}
    row.update(title=payload['message'].strip().split('\n')[0][:60] or '새 채팅', automatic_title=False,
               revision=row['revision'] + 1, updated_at=datetime.now(timezone.utc).isoformat())
    return {store._session_path(session) + '/conversations/' + row['id'] + '/meta.json': encode(row)}
