export function renderReply(doc,container,text) {
  const node=(tag,value)=>{const element=doc.createElement(tag);element.textContent=value;return element;};
  const inline=(element,value)=>{
    for(const part of value.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g)) {
      const code=part.startsWith('`')&&part.endsWith('`')&&part.length>2;
      const bold=part.startsWith('**')&&part.endsWith('**')&&part.length>4;
      element.append(node(code?'code':bold?'strong':'span',code?part.slice(1,-1):bold?part.slice(2,-2):part));
    }
  };
  const blocks=[];let paragraph=[],list=null,fence=null,code=[];
  const flush=()=>{if(paragraph.length){const p=node('p','');inline(p,paragraph.join('\n'));blocks.push(p);paragraph=[];}list=null;};
  for(const line of text.replace(/\r\n/g,'\n').split('\n')) {
    if(/^\s*```/.test(line)) {
      if(fence!==null){const pre=node('pre','');pre.append(node('code',code.join('\n')));blocks.push(pre);fence=null;code=[];}
      else {flush();fence=true;}
      continue;
    }
    if(fence!==null){code.push(line);continue;}
    if(!line.trim()){flush();continue;}
    const heading=line.match(/^#{1,6}\s+(.+)$/),item=line.match(/^\s*(?:([-*+])|(\d+)[.)])\s+(.+)$/);
    if(heading){flush();const h=node('h4','');inline(h,heading[1]);blocks.push(h);}
    else if(item){
      const tag=item[1]?'ul':'ol';
      if(!list||list.tag!==tag){flush();list={tag,element:node(tag,'')};if(item[2])list.element.setAttribute('start',item[2]);blocks.push(list.element);}
      const li=node('li','');inline(li,item[3]);list.element.append(li);
    } else {if(list)flush();paragraph.push(line);}
  }
  if(fence!==null){const pre=node('pre','');pre.append(node('code',code.join('\n')));blocks.push(pre);}
  flush();container.replaceChildren(...blocks);
}

export function sameDraft(a, b) {
  const excluded=['embeddings','source_artifact','reference_artifact'];
  const keys=Object.keys(a.settings).filter(key=>!excluded.includes(key));
  return a.session === b.session && a.prompt_id === b.prompt_id &&
    a.expected_revision === b.expected_revision && a.positive === b.positive && a.negative === b.negative &&
    JSON.stringify(a.settings.embeddings||[])===JSON.stringify(b.settings.embeddings||[]) &&
    ['source_artifact','reference_artifact'].every(key=>(a.settings[key]||'image.png')===(b.settings[key]||'image.png')) &&
    keys.length === Object.keys(b.settings).filter(key=>!excluded.includes(key)).length &&
    keys.every(key => Object.hasOwn(b.settings,key) && (key === 'loras' ? JSON.stringify(a.settings[key].map(l => [l.model,l.weight])) === JSON.stringify(b.settings[key].map(l => [l.model,l.weight])) : a.settings[key] === b.settings[key]));
}

export function preferredModel(models, selected) {
  return models.find(m=>m.name===selected)?.name || models.find(m=>m.size>=4e9 && m.size<=10e9)?.name || models[0]?.name || '';
}

export function tokenOptions(output,context) {
  const result={num_predict:output.trim()===''?null:Number(output),num_ctx:context.trim()===''?null:Number(context)};
  for(const [key,value] of Object.entries(result)) {
    if(value!==null && (!Number.isSafeInteger(value) || (key==='num_predict'?value!==-1&&value<1:value<1)))throw Error('출력 토큰은 -1 또는 양의 정수, 컨텍스트는 양의 정수를 입력하세요. 비워두면 모델 기본값입니다.');
  }
  return result;
}

const advancedFields=['temperature','top_p','top_k','min_p','repeat_penalty','seed'];
export function chatOptions(fields,extra='') {
  const result=extra.trim()?JSON.parse(extra):{};
  if(!result || typeof result!=='object' || Array.isArray(result))throw Error('추가 옵션은 JSON 객체로 입력하세요.');
  for(const key of advancedFields) {
    const value=String(fields[key]??'').trim();
    if(!value)continue;
    if(Object.hasOwn(result,key))throw Error(key+'는 입력칸과 추가 JSON 중 한 곳에만 지정하세요.');
    const number=Number(value);
    if(!Number.isFinite(number) || (['top_k','seed'].includes(key)&&!Number.isSafeInteger(number)))throw Error(key+' 값을 확인하세요.');
    result[key]=number;
  }
  return result;
}

export async function proposalSettings(turn,model,api) {
  const settings={...turn.context.settings};
  if(model && !['anima','krea2','flux2','zimage'].includes(model.family) && settings.prompt_mode==='native') {
    const family=model.family.startsWith('sd1')?'sd1':model.family.startsWith('sd2')?'sd2':'sdxl';
    const body={family,mode:'native',positive:turn.positive,negative:turn.negative};
    if(settings.embeddings?.length)body.embeddings=[...settings.embeddings];
    if((await api('/api/prompt-tokens',body)).truncated) {
      await api('/api/prompt-tokens',{...body,mode:'weighted'});
      settings.prompt_mode='weighted';
    }
  }
  return settings;
}

export async function streamChat(body, onEvent, fetcher=fetch) {
  const response=await fetcher('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if (!response.ok) {const error=await response.json();throw new Error(typeof error.detail==='string' ? error.detail : JSON.stringify(error.detail));}
  const reader=response.body.getReader(), decoder=new TextDecoder();
  let buffer='', final=null;
  try {
    while(true) {
      const {value,done}=await reader.read();
      buffer+=decoder.decode(value,{stream:!done});
      const lines=buffer.split('\n');buffer=lines.pop();
      for(const line of lines) {
        if(!line.trim()) continue;
        const event=JSON.parse(line);
        if(event.type==='error') throw new Error(event.text);
        if(event.type==='done') final=event.turn;
        onEvent(event);
      }
      if(done) break;
    }
    if(!final) throw new Error('연결이 끊겼습니다. 대화 기록을 확인한 뒤 다시 보내세요.');
    return final;
  } finally {await reader.cancel().catch(()=>{});reader.releaseLock();}
}

export function validateAttachments(files, count) {
  if(files.length+count>4)throw Error('참고 이미지는 최대 4개입니다. 기존 첨부를 빼고 다시 추가하세요.');
  for(const file of files) {
    if(!['image/png','image/jpeg','image/webp'].includes(file.type))throw Error('PNG, JPG, WebP 이미지 파일을 선택하세요.');
    if(file.size>12*1024*1024)throw Error('파일당 최대 12MiB입니다.');
  }
}

export function bindReferenceInput(panel, input, picker, button, upload) {
  let depth=0;
  const clear=()=>{depth=0;panel.classList.remove('is-dragging');};
  button.onclick=()=>picker.click();
  picker.onchange=()=>{const files=[...picker.files];picker.value='';upload(files);};
  const hasFiles=event=>Array.from(event.dataTransfer?.types||[]).includes('Files');
  panel.addEventListener('dragenter',event=>{if(hasFiles(event)){event.preventDefault();depth++;panel.classList.add('is-dragging');}});
  panel.addEventListener('dragover',event=>{if(hasFiles(event)){event.preventDefault();event.dataTransfer.dropEffect='copy';}});
  panel.addEventListener('dragleave',event=>{if(hasFiles(event)&&--depth<=0)clear();});
  panel.addEventListener('drop',event=>{if(hasFiles(event)){event.preventDefault();upload([...event.dataTransfer.files]);}clear();});
  input.addEventListener('paste',event=>{const files=[...(event.clipboardData?.files||[])];if(files.length){event.preventDefault();upload(files);}});
}

export async function chatPage(api,session,before=null) {
  try {
    const result=await api('/api/chat/page?session='+encodeURIComponent(session)+'&limit=50'+(before?'&before='+encodeURIComponent(before):''));
    return Array.isArray(result)?{items:result,before:null,has_more:false}:result;
  } catch(error) {
    if(error.status!==404 || before)throw error;
    return {items:await api('/api/chat?session='+encodeURIComponent(session)),before:null,has_more:false};
  }
}

export function reasoningPanel(doc,text='',pending=false) {
  const element=doc.createElement('details'),summary=doc.createElement('summary'),content=doc.createElement('div');
  element.className='chat-reasoning';content.className='reasoning-text';
  element.append(summary,content);
  const update=(value,writing)=>{element.hidden=!value;content.textContent=value;summary.textContent=writing?'생각하는 중 · 과정 보기':'생각 과정 보기';};
  update(text,pending);return {element,update};
}

export function promptComparison(doc, before, after, label) {
  const element=doc.createElement('details'),summary=doc.createElement('summary');
  element.className='prompt-comparison';
  const known=typeof before==='string',changed=known && before!==after;
  summary.textContent=label+' · '+(!known?'제안 원문':changed?'변경 있음':'변경 없음');
  element.append(summary);
  const grid=doc.createElement('div');grid.className='prompt-comparison-grid';
  let start=0,end=0;
  // Compare code points so highlighting never splits an emoji surrogate pair.
  const old=Array.from(before||''),next=Array.from(after);
  if(changed) {
    while(start<Math.min(old.length,next.length) && old[start]===next[start])start++;
    while(end<Math.min(old.length,next.length)-start && old[old.length-1-end]===next[next.length-1-end])end++;
  }
  for(const [title,chars,tag] of changed?[['요청 당시',old,'del'],['제안',next,'ins']]:[['제안',next,null]]) {
    const section=doc.createElement('section'),heading=doc.createElement('h4'),text=doc.createElement('pre');
    heading.textContent=title;
    if(tag) {
      const prefix=doc.createElement('span'),mark=doc.createElement(tag),suffix=doc.createElement('span');
      prefix.textContent=chars.slice(0,start).join('');mark.textContent=chars.slice(start,chars.length-end).join('');suffix.textContent=end?chars.slice(-end).join(''):'';
      text.append(prefix,mark,suffix);
    } else text.textContent=after;
    if(!chars.length){text.textContent='(비어 있음)';text.className='prompt-empty';}
    section.append(heading,text);grid.append(section);
  }
  element.append(grid);return element;
}

export function setupChat({api, getContext, apply, isWorkspaceReady=()=>true, doc=document, transport=streamChat}) {
  const $=id=>doc.getElementById(id);
  let active=null, visibleSession=null, historyReady=false, historyEpoch=0, modelEpoch=0, modelLoading=false, uploading=false, modelVision=false;
  const drafts=new Map(),attachments=new Map();
  let historyRows=[], historyCursor=null, olderLoading=false;
  let renderedTurns=new Map();
  const selectedReferences=()=>attachments.get(getContext().session)||[];
  const status=text=>{$('llm-status').textContent=text;};
  const nearBottom=()=>$('chat-history').scrollHeight-$('chat-history').scrollTop-$('chat-history').clientHeight<90;
  const bottom=()=>{$('chat-history').scrollTop=$('chat-history').scrollHeight;};
  function controls() {
    $('chat-older').hidden=!historyCursor;
    $('chat-older').disabled=olderLoading || !!active || !historyReady;
    $('chat-older').textContent=olderLoading?'이전 대화를 불러오는 중…':'이전 대화 더 보기';
    $('chat-send').disabled=!isWorkspaceReady() || !historyReady || visibleSession!==getContext().session || uploading || (selectedReferences().length>0&&!modelVision) || modelLoading || !!active || !$('llm-model').value || !$('chat-input').value.trim();
    $('chat-stop').hidden=!active || active.body.session!==getContext().session;
    $('chat-stop').disabled=!!active?.stopping;
    $('llm-model').disabled=!!active;$('llm-device').disabled=!!active;$('llm-think').disabled=!!active || modelLoading;$('chat-image').disabled=!!active;
    $('chat-upload').disabled=!!active || uploading;
    $('chat-attach').disabled=!!active || uploading;
    $('refresh-llm').disabled=!!active || modelLoading;
    $('llm-output').disabled=!!active;$('llm-context').disabled=!!active;
    for(const key of [...advancedFields,'options'])$('llm-'+key).disabled=!!active;
    $('chat-model-name').textContent=$('llm-model').value || '모델 연결 중';
  }
  function imagePreview(image) {
    const link=doc.createElement('a');link.className='chat-image-preview';link.href=image.src;
    link.target='_blank';link.rel='noopener';link.title=image.alt+' · 미리보기 새 탭에서 열기';
    link.setAttribute('aria-label',link.title);link.append(image);return link;
  }
  function referenceCard(reference,session,label) {
    const figure=doc.createElement('figure'),image=doc.createElement('img'),caption=doc.createElement('figcaption');
    image.src='/api/references/'+reference.id+'?session='+encodeURIComponent(session);image.alt=label+' · '+reference.filename;image.loading='lazy';
    caption.textContent=label+' · '+reference.filename;figure.append(imagePreview(image),caption);return figure;
  }
  function renderAttachments() {
    const session=getContext().session;
    $('chat-attachments').replaceChildren(...selectedReferences().map((reference,index)=>{
      const card=referenceCard(reference,session,'사용자 참고 '+(index+1));
      const remove=doc.createElement('button');remove.type='button';remove.textContent='이번 대화에서 빼기';remove.disabled=!!active||uploading;
      remove.onclick=()=>{if(active||uploading)return;attachments.set(session,selectedReferences().filter(r=>r.id!==reference.id));renderAttachments();controls();};card.append(remove);return card;
    }));
  }
  async function upload(files) {
    const session=getContext().session;
    if(!files.length)return;
    if(!session||active||uploading){$('chat-upload-status').textContent='진행 중인 작업이 끝난 뒤 이미지를 첨부하세요.';return;}
    try{validateAttachments(files,selectedReferences().length);}catch(error){$('chat-upload-status').textContent=error.message;return;}
    uploading=true;controls();renderAttachments();
    $('chat-upload-status').textContent='참고 이미지를 첨부하고 있어요…';
    let completed=0;
    try {
      for(const file of files) {
        if(file.size>12*1024*1024)throw Error('파일당 최대12MiB입니다.');
        const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(Error('파일을 읽을 수 없습니다.'));reader.readAsDataURL(file);});
        const reference=await api('/api/references',{session,filename:file.name,data});
        attachments.set(session,[...(attachments.get(session)||[]),reference]);
        completed++;
      }
      if(getContext().session===session)$('chat-upload-status').textContent=modelVision?'첨부 완료 · 직접 빼기 전까지 다음 대화에서도 참고합니다.':'첨부 완료 · 위 모델 설정에서 이미지 지원 모델을 선택하세요.';
    } catch(error){if(getContext().session===session)$('chat-upload-status').textContent=`${files[completed].name} 첨부 실패: ${error.message} · ${completed}개 첨부됨, ${files.length-completed}개 미첨부. 미첨부 파일만 다시 선택하세요.`;}
    finally{uploading=false;renderAttachments();controls();}
  }
  bindReferenceInput($('chat-history').parentElement,$('chat-input'),$('chat-upload'),$('chat-attach'),upload);
  function resize() {const input=$('chat-input');input.style.height='auto';input.style.height=Math.min(input.scrollHeight,160)+'px';}
  function bubble(role,text) {
    const element=doc.createElement('div');element.className='chat-bubble chat-'+role;
    const who=doc.createElement('span');who.className='chat-author';who.textContent=role==='user' ? '나' : '이미지 도우미';
    const content=doc.createElement('div');content.className='chat-text';
    if(role==='assistant')renderReply(doc,content,text);else content.textContent=text;
    element.append(who,content);return {element,content};
  }
  function createTurn(turn) {
    const exchange=doc.createElement('article');exchange.className='chat-exchange';
    const user=bubble('user',turn.message), assistant=bubble('assistant',turn.reply || (turn.status==='PENDING' ? '생각하고 있어요…' : turn.error || '응답이 없습니다.'));
    const reasoning=reasoningPanel(doc,turn.reasoning,turn.status==='PENDING');
    assistant.element.replaceChildren(assistant.element.children[0],reasoning.element,assistant.content);
    exchange.append(user.element,assistant.element);
    if(turn.image_context){const figure=doc.createElement('figure'),image=doc.createElement('img'),caption=doc.createElement('figcaption');figure.className='generated-reference';image.src='/api/jobs/'+turn.image_context.job_id+'/image.png?session='+encodeURIComponent(turn.context?.session||getContext().session);image.alt='이전 생성 결과';image.loading='lazy';caption.textContent='앱 생성 결과 · '+turn.image_context.job_id.slice(0,8);figure.append(imagePreview(image),caption);user.element.append(figure);}
    for(const [index,reference] of (turn.user_references||[]).entries()) {
      const card=referenceCard(reference,turn.context?.session||getContext().session,'사용자 참고 '+(index+1));
      const reuse=doc.createElement('button');reuse.type='button';reuse.textContent='다시 참고하기';reuse.onclick=()=>{const session=getContext().session;if(active||uploading||session!==turn.context?.session)return;const selected=selectedReferences();if(selected.some(r=>r.id===reference.id))return;if(selected.length>=4){status('참고 이미지는 최대4개입니다.');return;}attachments.set(session,[...selected,reference]);renderAttachments();controls();};card.append(reuse);user.element.append(card);
    }
    if(turn.created_at) {const time=doc.createElement('time');time.className='chat-time';time.dateTime=turn.created_at;time.textContent=new Date(turn.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});user.element.append(time);}
    if(turn.status==='PENDING') {assistant.element.classList.add('is-writing');assistant.content.setAttribute('aria-label','응답 작성 중');}
    if(turn.status==='SUCCEEDED' && turn.memory) {
      const details=doc.createElement('details'),summary=doc.createElement('summary'),text=doc.createElement('p');
      details.className='chat-memory';summary.textContent='기억한 내용';text.textContent=turn.memory;
      details.append(summary,text);assistant.element.append(details);
    }
    if(turn.status==='SUCCEEDED' && turn.update) {
      const card=doc.createElement('div');card.className='chat-proposal';
      const title=doc.createElement('h4'),note=doc.createElement('p');title.textContent='✦ 프롬프트 수정 제안';
      note.className='proposal-context';note.textContent='요청 당시 원문과 비교합니다. 현재 작업실에 반영하려면 아래 버튼을 누르세요.';
      if(turn.application)note.textContent=`반영 완료 · revision ${turn.application.revision}`+(turn.application.job_id?' · 생성 요청 접수':'')+' · 이후 수정한 작업실 내용은 유지됩니다.';
      else if(turn.application_error)note.textContent='반영 기록을 확인하지 못했습니다. 작업실의 저장된 내용을 확인하세요: '+turn.application_error;
      card.append(title,note,promptComparison(doc,turn.context?.positive,turn.positive||'','Positive'),promptComparison(doc,turn.context?.negative,turn.negative||'','Negative'));
      const actions=[];
      for(const generate of turn.application||turn.application_error?[]:[false,true]) {
        const button=doc.createElement('button');button.type='button';button.textContent=generate ? '반영하고 생성' : '프롬프트 반영';
        actions.push(button);
        button.onclick=async()=>{button.disabled=true;try {await applyTurn(turn,generate);actions.forEach(action=>action.disabled=true);button.textContent='반영 완료';} catch(e){status(e.message);button.disabled=false;}};card.append(button);
      }
      assistant.element.append(card);
    }
    if(['FAILED','CANCELLED'].includes(turn.status)) {
      assistant.element.children[0].textContent+=' · '+(turn.status==='FAILED'?'응답 실패':'응답 중단');
      if(turn.reply){const note=doc.createElement('p');note.className='chat-error';note.textContent=turn.error || '응답 중단';assistant.element.append(note);}
      const session=turn.context?.session||getContext().session;
      const retry=doc.createElement('button');retry.type='button';retry.textContent='이 메시지 다시 작성';
      retry.onclick=()=>{
        if(active||uploading||!historyReady||!isWorkspaceReady()||session!==getContext().session||session!==visibleSession)return;
        const input=$('chat-input');
        if(input.value && input.value!==turn.message){status('작성 중인 메시지를 먼저 보내거나 비운 뒤 다시 작성하세요.');return;}
        input.value=turn.message;drafts.set(session,turn.message);resize();controls();input.focus();
      };
      assistant.element.append(retry);
    }
    if(active && turn.id===active.body.request_id) {active.replyNode=assistant.content;active.reasoningPanel=reasoning;active.bubble=assistant.element;}
    return exchange;
  }
  function renderTurn(turn) {
    const signature=JSON.stringify(turn),previous=renderedTurns.get(turn.id);
    if(previous?.signature===signature)return previous.node;
    const node=createTurn(turn),reasoning=previous?.node.querySelector('.chat-reasoning');
    if(reasoning)node.querySelector('.chat-reasoning').open=reasoning.open;
    renderedTurns.set(turn.id,{signature,node});return node;
  }
  function renderHistory(turns) {
    const panel=$('chat-history'),nodes=turns.map(renderTurn);
    if(panel.children.length!==nodes.length || nodes.some((node,index)=>panel.children[index]!==node))panel.replaceChildren(...nodes);
    renderedTurns=new Map(turns.map(turn=>[turn.id,renderedTurns.get(turn.id)]));
  }
  function switchWorkspace() {
    const session=getContext().session, changed=visibleSession!==session;
    if(changed) {
      if(visibleSession)drafts.set(visibleSession,$('chat-input').value);
      visibleSession=session;historyReady=false;historyEpoch++;historyRows=[];historyCursor=null;olderLoading=false;renderedTurns.clear();
      $('chat-input').value=drafts.get(session)||'';resize();renderAttachments();
      $('chat-history').replaceChildren();$('chat-upload-status').textContent='';
      status(active ? (active.body.session===session?'응답을 이어 받고 있어요…':'다른 Workspace에서 응답 중입니다. 완료 후 대화할 수 있어요.'):'대화 기록을 불러오는 중…');
    }
    controls();return changed;
  }
  async function refresh() {
    const context=getContext();if(!context.session)return;
    const changed=switchWorkspace();
    const epoch=++historyEpoch, stick=changed || nearBottom();
    olderLoading=false;historyReady=false;controls();
    let page;
    try {page=await chatPage(api,context.session);}
    catch(error){
      if(epoch!==historyEpoch || getContext().session!==context.session)return;
      throw error;
    }
    if(epoch!==historyEpoch || getContext().session!==context.session)return;
    olderLoading=false;
    if(!historyRows.length)historyCursor=page.has_more?page.before:null;
    const history=[...new Map([...historyRows,...page.items].map(turn=>[turn.id,turn])).values()];
    if(active?.body.session===context.session) {
      const index=history.findIndex(t=>t.id===active.body.request_id);
      const current={id:active.body.request_id,message:active.body.message,reply:active.reply,reasoning:active.reasoning,status:'PENDING',user_references:active.references,context:active.body};
      if(index<0)history.push(current);else if(history[index].status==='PENDING')history[index]=current;
    }
    historyRows=history;
    renderHistory(history);
    if(!history.length) {
      const empty=doc.createElement('div');empty.className='chat-welcome';
      const title=doc.createElement('h3');title.textContent='어떤 이미지를 만들어 볼까요?';
      const description=doc.createElement('p');description.textContent='아이디어부터 이야기해도 좋아요. 함께 장면을 정하고 프롬프트를 다듬어 봐요.';empty.append(title,description);
      for(const text of ['여성 한 명이 나오는 일러스트를 같이 구상해 줘','현재 프롬프트에서 개선할 부분을 알려줘']) {
        const button=doc.createElement('button');button.type='button';button.textContent=text;button.onclick=()=>{$('chat-input').value=text;resize();controls();$('chat-input').focus();};empty.append(button);
      }
      $('chat-history').append(empty);
    }
    historyReady=true;if(stick)bottom();controls();
    if(changed&&!active)status('이어서 이야기해 주세요.');
  }
  $('chat-older').onclick=async()=>{
    if(olderLoading || active || !historyReady || !historyCursor)return;
    const session=getContext().session,epoch=historyEpoch,cursor=historyCursor;
    olderLoading=true;controls();
    try {
      const page=await chatPage(api,session,cursor);
      if(epoch!==historyEpoch || session!==getContext().session)return;
      const panel=$('chat-history'),height=panel.scrollHeight,top=panel.scrollTop;
      historyRows=[...new Map([...page.items,...historyRows].map(turn=>[turn.id,turn])).values()];
      historyCursor=page.has_more?page.before:null;
      renderHistory(historyRows);
      panel.scrollTop=top+panel.scrollHeight-height;
      status(page.items.length?'이전 대화를 불러왔어요.': '읽을 수 있는 이전 대화가 없습니다. 저장 기록 안내를 확인하세요.');
    } catch(error){if(epoch===historyEpoch && session===getContext().session)status(error.message);}
    finally{if(epoch===historyEpoch && session===getContext().session){olderLoading=false;controls();}}
  };
  async function applyTurn(turn,generate) {
    if(!sameDraft(turn.context,getContext()))throw new Error('프롬프트가 그 뒤에 변경됐어요. 현재 내용을 기준으로 다시 다듬어 달라고 요청해 주세요.');
    const warning=await apply(turn,generate);
    status(warning || (generate ? '프롬프트를 반영하고 이미지 생성을 요청했어요.' : '프롬프트를 작업실에 반영했어요.'));
  }
  async function modelInfo() {
    const name=$('llm-model').value,epoch=++modelEpoch;
    if(!name){modelLoading=false;modelVision=false;$('llm-think').replaceChildren();$('llm-capabilities').textContent='';controls();return;}
    modelLoading=true;modelVision=false;controls();
    for(const [id,key] of [['llm-output','num_predict'],['llm-context','num_ctx']]) {
      let value='';try{value=localStorage.getItem('image-maker.'+key+'.'+name)||'';}catch{}
      $(id).value=value;
    }
    let advanced={};try{advanced=JSON.parse(localStorage.getItem('image-maker.options.'+name)||'{}');}catch{}
    if(!advanced || typeof advanced!=='object' || Array.isArray(advanced))advanced={};
    for(const key of [...advancedFields,'options'])$('llm-'+key).value=advanced[key]??'';
    try {
      const info=await api('/api/llm/model-info?name='+encodeURIComponent(name));
      if(epoch!==modelEpoch)return;
      modelVision=info.capabilities.includes('vision');
      const values=info.thinking?.values || [];
      $('llm-think').replaceChildren(...[null,...values].map(value=>{
        const option=doc.createElement('option');option.value=JSON.stringify(value);
        option.textContent=value===null ? '모델 기본값'+(info.thinking?.default!=null ? ' ('+JSON.stringify(info.thinking.default)+')':'') : value===false?'끄기':value===true?'켜기':value;
        return option;
      }));
      let selected='null';try {selected=localStorage.getItem('image-maker.think.'+name)||'null';}catch{}
      $('llm-think').value=[...$('llm-think').options].some(o=>o.value===selected)?selected:'null';
      $('llm-capabilities').textContent=(info.capabilities.includes('vision')?'이미지 내용을 함께 볼 수 있는 모델':'이미지 미지원 · 생성 프롬프트만 전달')+' · reasoning '+(values.length?values.map(v=>v===false?'끄기':v===true?'켜기':v).join(' / '):'설정 정보 없음');
    }catch(error){if(epoch===modelEpoch){status(error.message);$('llm-think').replaceChildren();}}
    finally {if(epoch===modelEpoch){modelLoading=false;controls();}}
  }
  async function connect() {
    if(active || modelLoading)return;
    const epoch=++modelEpoch;
    modelLoading=true;controls();
    try {
      const result=await api('/api/llm/models');
      if(epoch!==modelEpoch)return;
      let selected=$('llm-model').value;
      try {selected ||= localStorage.getItem('image-maker.llm') || '';} catch {}
      $('llm-model').replaceChildren(...result.models.map(m=>{const option=doc.createElement('option');option.value=m.name;option.textContent=m.name;return option;}));
      $('llm-model').value=preferredModel(result.models,selected);
      $('llm-connection').textContent=result.error ? '연결 실패' : '로컬 연결';
      $('chat-settings').open=!!result.error || !result.models.length;
      status(result.error || (result.models.length ? '편하게 이야기해 주세요.' : 'Ollama에 설치된 대화 모델이 없습니다.'));
    } catch(e) {if(epoch!==modelEpoch)return;status(e.message);$('chat-settings').open=true;}
    await modelInfo();controls();
  }
  $('refresh-llm').onclick=connect;
  for(const key of [...advancedFields,'options'])$('llm-'+key).onchange=()=>{
    try{localStorage.setItem('image-maker.options.'+$('llm-model').value,JSON.stringify(Object.fromEntries([...advancedFields,'options'].map(name=>[name,$('llm-'+name).value]))));}catch{}
  };
  $('llm-model').onchange=()=>{try{localStorage.setItem('image-maker.llm',$('llm-model').value);}catch{}return modelInfo();};
  $('llm-think').onchange=()=>{try{localStorage.setItem('image-maker.think.'+$('llm-model').value,$('llm-think').value);}catch{}};
  for(const [id,key] of [['llm-output','num_predict'],['llm-context','num_ctx']])$(id).onchange=()=>{try{localStorage.setItem('image-maker.'+key+'.'+$('llm-model').value,$(id).value);}catch{}};
  $('chat-input').oninput=()=>{drafts.set(visibleSession,$('chat-input').value);resize();controls();};
  $('chat-input').onkeydown=event=>{if(event.key==='Enter' && !event.shiftKey && !event.isComposing && event.keyCode!==229){event.preventDefault();if(!$('chat-send').disabled)$('chat-form').requestSubmit();}};
  $('chat-stop').onclick=async()=>{
    const run=active;if(!run)return;run.stopping=true;controls();status('응답을 중단하고 모델 메모리를 정리하고 있어요…');
    try {const result=await api(`/api/chat/${run.body.request_id}/cancel?session=${encodeURIComponent(run.body.session)}`,{});if(!result.cancelled){run.stopping=false;controls();}}catch(e){run.stopping=false;controls();if(active===run && getContext().session===run.body.session)status(e.message);}
  };
  $('chat-form').onsubmit=async event=>{
    event.preventDefault();if(!isWorkspaceReady() || !historyReady || visibleSession!==getContext().session || uploading || (selectedReferences().length>0&&!modelVision) || modelLoading || active || !$('llm-model').value || !$('chat-input').value.trim())return;
    const action=$('chat-action').value;
    const image=$('chat-image').value;
    let limits;try{limits={...tokenOptions($('llm-output').value,$('llm-context').value),options:chatOptions(Object.fromEntries(advancedFields.map(key=>[key,$('llm-'+key).value])),$('llm-options').value)};}catch(error){status(error.message);$('chat-settings').open=true;return;}
    const body={...getContext(),...limits,attachments:selectedReferences().map(r=>r.id),think:JSON.parse($('llm-think').value||'null'),image_mode:['latest','none'].includes(image)?image:'selected',image_job:['latest','none'].includes(image)?null:image,model:$('llm-model').value,device:$('llm-device').value,message:$('chat-input').value.trim(),request_id:crypto.randomUUID()};
    const run={body,reply:'',reasoning:'',stopping:false,references:[...selectedReferences()]};active=run;renderAttachments();
    $('chat-input').value='';drafts.set(body.session,'');resize();controls();
    if($('chat-history').querySelector('.chat-welcome'))$('chat-history').replaceChildren();
    $('chat-history').append(renderTurn({id:body.request_id,message:body.message,status:'PENDING',created_at:new Date().toISOString(),user_references:selectedReferences(),context:getContext()}));bottom();status('응답 준비 중…');
    try {
      const turn=await transport(body,event=>{
        if(event.type==='reply')run.reply=event.text;
        if(event.type==='reasoning')run.reasoning+=event.text;
        if(getContext().session!==body.session)return;
        if(event.type==='reasoning'){const stick=nearBottom();run.reasoningPanel?.update(run.reasoning,true);status('생각하는 중…');if(stick)bottom();}
        if(event.type==='status')status(event.text);
        if(event.type==='reply' && run.replyNode?.isConnected){const stick=nearBottom();run.reasoningPanel?.update(run.reasoning,false);renderReply(doc,run.replyNode,event.text);status('답변하고 있어요…');if(stick)bottom();}
      });
      if(turn.status==='FAILED')throw new Error(turn.error || '응답을 완료하지 못했습니다. 설정을 확인하고 다시 보내세요.');
      if(turn.status==='SUCCEEDED' && turn.update && action!=='suggest')await applyTurn(turn,action==='generate');
      else if(getContext().session===body.session)status(turn.status==='SUCCEEDED' ? (turn.update ? '아래 제안을 확인하고 반영할 수 있어요.' : '이어서 이야기해 주세요.') : turn.error || '응답 중단');
    } catch(e) {
      if(!drafts.get(body.session)) {
        drafts.set(body.session,body.message);
        if(visibleSession===body.session){$('chat-input').value=body.message;resize();}
      }
      if(getContext().session===body.session)status(e.message);
    }
    finally {active=null;controls();renderAttachments();await refresh().catch(e=>status(e.message));if(getContext().session===body.session)$('chat-input').focus();}
  };
  return {refresh,connect,switchWorkspace};
}
