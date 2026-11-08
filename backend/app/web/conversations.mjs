export function setupConversations({api,getSession,onSelect,doc=document,storage=globalThis.localStorage}) {
  const $=id=>doc.getElementById(id);
  let workspace=null,selected=null,rows=[],epoch=0,busy=false,ready=false;
  const saved=session=>{try{return storage?.getItem('image-maker.conversation.'+session);}catch{return null;}};
  function render() {
    const row=rows.find(r=>r.id===selected);
    $('conversation-current').textContent=row?.title||'채팅 선택';
    $('conversation-title').value=row?.title||'';
    $('conversation-archive').textContent=row?.archived?'다시 표시':'목록에서 숨기기';
    $('conversation-list').replaceChildren(...rows.map(item=>{
      const button=doc.createElement('button');button.type='button';
      button.textContent=item.title+(item.archived?' · 숨김':'');
      button.setAttribute('aria-pressed',String(item.id===selected));button.disabled=busy;
      button.onclick=async()=>{selected=item.id;remember();render();$('conversation-menu').open=false;try{await onSelect();}catch(error){$('conversation-status').textContent=error.message;}};return button;
    }));
    for(const id of ['conversation-new','conversation-rename','conversation-archive','conversation-archived'])$(id).disabled=busy||!ready;
  }
  function remember(){try{storage?.setItem('image-maker.conversation.'+workspace,selected);}catch{}}
  async function load(preferred) {
    const session=getSession(),version=++epoch;
    if(workspace!==session){workspace=session;selected=saved(session);ready=false;rows=[];render();}
    const result=await api('/api/conversations?session='+encodeURIComponent(session)+'&include_archived='+$('conversation-archived').checked);
    if(version!==epoch||session!==getSession())return false;
    rows=result;
    selected=[preferred,selected,rows[0]?.id].find(id=>rows.some(row=>row.id===id))||null;
    ready=true;remember();render();return true;
  }
  async function mutate(kind) {
    if(busy||!ready||workspace!==getSession())return;
    const session=workspace,row=rows.find(r=>r.id===selected),title=$('conversation-title').value;busy=true;render();
    try {
      const result=kind==='new'?await api('/api/conversations',{session}):await api('/api/conversations/update',{
        session,conversation_id:row.id,expected_revision:row.revision,
        title:kind==='rename'?title:row.title,archived:kind==='archive'?!row.archived:row.archived});
      if(session!==getSession())return;
      if(result.archived)$('conversation-archived').checked=true;
      await load(result.id);if(kind==='new')$('conversation-menu').open=false;await onSelect();$('conversation-status').textContent=kind==='new'?'새 채팅을 만들었습니다.':kind==='archive'?(result.archived?'목록에서 숨겼습니다. 대화 내용은 유지됩니다.':'채팅을 다시 표시합니다.'):'저장했습니다.';
    } catch(error){if(session===getSession())$('conversation-status').textContent=error.message;}
    finally{busy=false;render();}
  }
  doc.addEventListener('pointerdown',event=>{const menu=$('conversation-menu');if(menu.open&&!menu.contains(event.target))menu.open=false;});
  $('conversation-menu').addEventListener('keydown',event=>{if(event.key==='Escape'){$('conversation-menu').open=false;$('conversation-menu').querySelector('summary').focus();}});
  $('conversation-new').onclick=()=>mutate('new');
  $('conversation-rename').onclick=()=>mutate('rename');
  $('conversation-archive').onclick=()=>mutate('archive');
  $('conversation-archived').onchange=async()=>{try{await load();await onSelect();}catch(error){$('conversation-status').textContent=error.message;}};
  return {load,id:()=>workspace===getSession()?selected:null,ready:()=>ready&&workspace===getSession()&&!!selected,
    archived:()=>!!rows.find(row=>row.id===selected)?.archived};
}
