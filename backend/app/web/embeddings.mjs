import {componentCompatibility,renderComponentSelect} from './pipeline.mjs?v=20261107-ui32';

export function setupEmbeddings({doc=document,inventory,checkpoint,isBusy,changed,copy=text=>navigator.clipboard.writeText(text)}) {
  const $=id=>doc.getElementById(id);let selected=[];
  function refresh() {
    const models=inventory(),base=checkpoint();
    renderComponentSelect($('embedding'),'embedding',base,models,'임베딩 추가…');
    $('embedding').disabled=isBusy()||selected.length>=4;
    for(const option of $('embedding').options)if(selected.includes(option.value))option.disabled=true;
    $('embedding-list').replaceChildren(...selected.map(id=>{
      const model=models.find(m=>m.id===id),reason=componentCompatibility(model,base,'embedding');
      const row=doc.createElement('div');row.className='embedding-item';
      const info=doc.createElement('div'),name=doc.createElement('strong'),tokens=doc.createElement('code');
      name.textContent=model?.name||'누락 임베딩 · '+id;tokens.textContent=(model?.profile?.tokens||[]).join(' ');info.append(name,tokens);
      if(reason){const error=doc.createElement('small');error.className='chat-error';error.textContent=reason;info.append(error);}
      const copyButton=doc.createElement('button');copyButton.type='button';copyButton.textContent='토큰 복사';copyButton.disabled=!tokens.textContent||isBusy();
      copyButton.onclick=async()=>{try{await copy(tokens.textContent);$('embedding-status').textContent='토큰을 복사했습니다. 원하는 프롬프트 위치에 붙여 넣으세요.';}catch{$('embedding-status').textContent='복사하지 못했습니다. 표시된 토큰을 직접 선택해 복사하세요.';}};
      const remove=doc.createElement('button');remove.type='button';remove.textContent='제거';remove.disabled=isBusy();remove.setAttribute('aria-label',(model?.name||id)+' 임베딩 제거');
      remove.onclick=()=>{if(isBusy())return;selected=selected.filter(value=>value!==id);refresh();changed();};
      row.append(info,copyButton,remove);return row;
    }));
    $('embedding-count').textContent=selected.length?`${selected.length}/4개 선택`:'선택 안 함';
  }
  function add(id) {
    if(isBusy())return false;
    const reason=componentCompatibility(inventory().find(m=>m.id===id),checkpoint(),'embedding');
    if(reason||selected.length>=4){$('embedding-status').textContent=reason||'임베딩은 최대 4개입니다.';return false;}
    if(selected.includes(id))return false;
    selected.push(id);$('embedding').value='';$('embedding-status').textContent='필요한 토큰을 프롬프트에 사용하세요. 원문은 자동 수정하지 않습니다.';refresh();changed();return true;
  }
  $('embedding').onchange=()=>{const id=$('embedding').value;$('embedding').value='';if(id)add(id);};
  return {get:()=>[...selected],restore:ids=>{selected=[...(ids||[])];$('embedding-status').textContent='';refresh();},refresh,add};
}
