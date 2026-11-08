import test from 'node:test';
import assert from 'node:assert/strict';
import {setupChat,chatPage} from '../backend/app/web/chat.mjs';

// A small DOM fixture exercises the controller; it does not verify browser layout.
function fixture() {
  class Element {
    constructor(){this.children=[];this.value='';this.style={};this.scrollHeight=100;this.scrollTop=0;this.clientHeight=100;this.classList={add(){},remove(){}};}
    append(...nodes){this.children.push(...nodes);for(const node of nodes)node.parentElement=this;}
    replaceChildren(...nodes){this.children=[];this.append(...nodes);}
    setAttribute(){}
    addEventListener(){}
    focus(){}
    querySelector(selector){for(const child of this.children){if(child.className===selector.slice(1))return child;const found=child.querySelector(selector);if(found)return found;}return null;}
    get options(){return this.children;}
  }
  const elements=new Map();
  const doc={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();}};
  const $=id=>doc.getElementById(id);
  $('chat-history').parentElement=new Element();$('llm-model').value='vision';$('chat-image').value='none';$('chat-action').value='suggest';
  let context={session:'A',settings:{}},ready=true, apiImpl=async()=>[],applyImpl=async()=>{},transport=async()=>{throw Error('offline');};
  const chat=setupChat({doc,getContext:()=>context,isWorkspaceReady:()=>ready,apply:(...args)=>applyImpl(...args),api:(...args)=>apiImpl(...args),transport:(...args)=>transport(...args)});
  return {chat,$,setReady:value=>ready=value,setSession:session=>{context={session,settings:{}};},setContext:value=>context=value,apply:fn=>applyImpl=fn,api:fn=>apiImpl=fn,transport:fn=>transport=fn};
}
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};

test('conversations in one workspace isolate drafts, history, and late streamed replies',async()=>{
 const f=fixture(),first={session:'A',conversation_id:'first',settings:{}},second={...first,conversation_id:'second'};
 f.setContext(first);await f.chat.refresh();f.$('chat-input').value='draft one';
 f.setContext(second);f.chat.switchWorkspace();assert.equal(f.$('chat-input').value,'');await f.chat.refresh();
 f.$('chat-input').value='draft two';f.setContext(first);f.chat.switchWorkspace();await f.chat.refresh();
 assert.equal(f.$('chat-input').value,'draft one');
 const pending=deferred();let emit,request;
 f.transport((body,callback)=>{request=body;emit=callback;return pending.promise;});
 const sending=f.$('chat-form').onsubmit({preventDefault(){}});
 assert.equal(request.conversation_id,'first');
 f.setContext(second);f.chat.switchWorkspace();await f.chat.refresh();
 emit({type:'reply',text:'from first'});assert.equal(f.$('chat-input').value,'draft two');
 pending.reject(Error('first failed'));await sending;
 assert.equal(f.$('chat-input').value,'draft two');assert.notEqual(f.$('llm-status').textContent,'first failed');
 f.setContext(first);f.chat.switchWorkspace();await f.chat.refresh();assert.equal(f.$('chat-input').value,'draft one');
});

test('conversation pages always send their ID and never fall back to all workspace history',async()=>{
 let path;
 const api=async url=>{path=url;const error=Error('missing');error.status=404;throw error;};
 await assert.rejects(()=>chatPage(api,'workspace',null,'conversation'),/missing/);
 assert.match(path,/conversation_id=conversation/);
});

test('failed model connection does not leave the model summary loading',async()=>{
 const f=fixture();f.$('llm-model').value='';
 f.api(async()=>({models:[],error:'connection refused'}));await f.chat.connect();
 assert.equal(f.$('chat-model-name').textContent,'대화 모델을 연결해 주세요');
 assert.equal(f.$('llm-status').textContent,'connection refused');
});

test('reasoning uses the server values without aliases and falls back to default when absent',async()=>{
 const f=fixture();await f.chat.refresh();
 f.api(async()=>({capabilities:['completion','thinking'],thinking:{values:[false,true,'low','medium','high'],default:'high',labels:{high:'xhigh'}}}));
 await f.$('llm-model').onchange();
 const option=f.$('llm-think').children.find(row=>row.textContent==='high');
 assert.equal(option.value,'"high"');assert.match(f.$('llm-think').children[0].textContent,/high/);
 assert.equal(f.$('llm-think').children.some(row=>row.textContent==='xhigh'),false);
 f.$('llm-think').value=option.value;f.$('chat-input').value='test';let body;
 f.transport(async request=>{body=request;return {status:'SUCCEEDED'};});
 f.api(async()=>[]);await f.$('chat-form').onsubmit({preventDefault(){}});
 assert.equal(body.think,'high');
 f.api(async()=>({capabilities:['completion','thinking']}));await f.$('llm-model').onchange();
 assert.equal(f.$('llm-think').children.length,1);assert.equal(f.$('llm-think').value,'null');
 assert.match(f.$('llm-capabilities').textContent,/서버가 선택지를 제공하지 않아/);
});

test('explicit workspace switch clears loading after the latest history request only',async()=>{
 const f=fixture();await f.chat.refresh();
 f.setSession('B');f.chat.switchWorkspace();
 assert.equal(f.$('llm-status').textContent,'대화 기록을 불러오는 중…');
 const stale=deferred(),latest=deferred();let calls=0;
 f.api(()=>++calls===1?stale.promise:latest.promise);
 const first=f.chat.refresh(),second=f.chat.refresh();
 stale.resolve([]);await first;
 assert.equal(f.$('llm-status').textContent,'대화 기록을 불러오는 중…');
 latest.resolve([]);await second;
 assert.equal(f.$('llm-status').textContent,'이어서 이야기해 주세요.');
 f.$('llm-status').textContent='연결 실패';f.api(async()=>[]);await f.chat.refresh();
 assert.equal(f.$('llm-status').textContent,'연결 실패');
});

test('chat images open scoped previews without replacing the conversation',async()=>{
 const f=fixture(),session='2026-11/workspace A';f.setSession(session);
 const turn={id:'images',message:'compare',reply:'done',status:'SUCCEEDED',context:{session},
  image_context:{job_id:'generated-job'},user_references:[{id:'user-image',filename:'reference.png'}]};
 f.api(async()=>[turn]);await f.chat.refresh();
 const user=f.$('chat-history').children[0].children[0];
 const generated=user.querySelector('.generated-reference'),reference=user.children.find(node=>node.children.some(child=>child.textContent==='사용자 참고 1 · reference.png'));
 for(const [card,url] of [[generated,'/api/jobs/generated-job/image.png'],[reference,'/api/references/user-image']]){
  const link=card.children[0];assert.equal(link.className,'chat-image-preview');
  assert.equal(link.href,url+'?session='+encodeURIComponent(session));
  assert.equal(link.children[0].src,link.href);assert.equal(link.target,'_blank');assert.equal(link.rel,'noopener');
  assert.match(link.title,/새 탭/);
 }
 await reference.children.at(-1).onclick();
 const pending=f.$('chat-attachments').children[0];assert.equal(pending.children[0].href,reference.children[0].href);
 assert.equal(pending.children.at(-1).textContent,'이번 대화에서 빼기');
 await pending.children.at(-1).onclick();assert.equal(f.$('chat-attachments').children.length,0);
 f.setSession('B');f.chat.switchWorkspace();
 assert.match(generated.children[0].href,/workspace%20A/); // Existing links retain their originating Workspace.
});

test('manual proposals reject changed drafts and preserve retry after a failed save',async()=>{
 for(const generate of [false,true]){
  const f=fixture(),context={session:'A',prompt_id:'p',expected_revision:1,positive:'original',negative:'keep\nnegative',settings:{}};
  const turn={id:'proposal',message:'edit',reply:'proposal',status:'SUCCEEDED',update:true,positive:'revised',negative:context.negative,context};
  f.setContext(context);f.api(async()=>[turn]);await f.chat.refresh();
  const button=f.$('chat-history').querySelector('.chat-proposal').children.at(generate?-1:-2);
  let calls=0;f.apply(async(actual,requested)=>{calls++;assert.equal(actual,turn);assert.equal(requested,generate);if(calls===1)throw Error('save unavailable');});
  for(const change of [{positive:'manual edit'},{negative:'manual negative'},{expected_revision:2},{settings:{model:'different'}},{session:'B'}]){
   f.setContext({...context,...change});await button.onclick();
   assert.equal(calls,0);assert.equal(button.disabled,false);assert.match(f.$('llm-status').textContent,/변경/);
  }
  f.setContext(context);await button.onclick();
  assert.equal(calls,1);assert.equal(button.disabled,false);assert.equal(f.$('llm-status').textContent,'save unavailable');
  await button.onclick();assert.equal(calls,2);assert.equal(button.disabled,true);assert.equal(button.textContent,'반영 완료');
 }
});

test('automatic proposal application checks the current draft and workspace after streaming',async()=>{
 for(const action of ['apply','generate'])for(const change of [null,{positive:'manual edit'},{session:'B'}]){
  const f=fixture(),context={session:'A',prompt_id:'p',expected_revision:1,positive:'original',negative:'keep',settings:{}};
  f.setContext(context);await f.chat.refresh();f.$('chat-action').value=action;
  const pending=deferred();let request,calls=[];
  f.transport(body=>{request=body;return pending.promise;});f.apply(async(turn,generate)=>calls.push({turn,generate}));
  f.$('chat-input').value='edit request';const sending=f.$('chat-form').onsubmit({preventDefault(){}});
  if(change){f.setContext({...context,...change});if(change.session){f.chat.switchWorkspace();await f.chat.refresh();}}
  const banner=f.$('llm-status').textContent;
  const turn={id:request.request_id,message:request.message,reply:'done',status:'SUCCEEDED',update:true,positive:'revised',negative:'keep',context:request};
  pending.resolve(turn);await sending;
  assert.equal(calls.length,change?0:1);
  if(!change)assert.equal(calls[0].generate,action==='generate');
  else if(change.session){assert.equal(f.$('llm-status').textContent,banner);assert.equal(f.$('chat-input').value,'');f.setContext(context);f.chat.switchWorkspace();assert.equal(f.$('chat-input').value,'edit request');}
  else {assert.match(f.$('llm-status').textContent,/변경/);assert.equal(f.$('chat-input').value,'edit request');}
 }
});

test('ready chat history cannot send a stale prompt while the workspace editor is loading',async()=>{
 const f=fixture();await f.chat.refresh();let sent=0;
 f.transport(async()=>{sent++;return {status:'SUCCEEDED'};});
 f.$('chat-input').value='keep this draft';f.setReady(false);f.$('chat-input').oninput();
 assert.equal(f.$('chat-send').disabled,true);
 await f.$('chat-form').onsubmit({preventDefault(){}});assert.equal(sent,0);assert.equal(f.$('chat-input').value,'keep this draft');
 f.setReady(true);f.$('chat-input').oninput();assert.equal(f.$('chat-send').disabled,false);
 await f.$('chat-form').onsubmit({preventDefault(){}});assert.equal(sent,1);
});

test('reasoning expansion survives changed pending content and final response',async()=>{
 for(const open of [true,false]){
  const f=fixture();await f.chat.refresh();const pending=deferred();let body,emit;
  f.transport((request,onEvent)=>{body=request;emit=onEvent;return pending.promise;});
  f.$('chat-input').value='question';const sending=f.$('chat-form').onsubmit({preventDefault(){}});
  emit({type:'reasoning',text:'first'});
  f.$('chat-history').querySelector('.chat-reasoning').open=open;
  emit({type:'reasoning',text:' second'});await f.chat.refresh();
  assert.equal(f.$('chat-history').querySelector('.chat-reasoning').open,open);
  const turn={id:body.request_id,message:'question',reply:'answer',reasoning:'first second',status:'SUCCEEDED',context:body};
  f.api(async()=>[turn]);pending.resolve(turn);await sending;
  const details=f.$('chat-history').querySelector('.chat-reasoning');
  assert.equal(details.open,open);assert.equal(details.children[1].textContent,'first second');
 }
});

test('late cancellation errors do not overwrite another workspace or a completed response',async()=>{
 for(const switchAway of [false,true]){
  const f=fixture();await f.chat.refresh();const reply=deferred(),cancel=deferred();
  f.transport(()=>reply.promise);f.api(path=>path.includes('/cancel?')?cancel.promise:Promise.resolve([]));
  f.$('chat-input').value='request A';
  const sending=f.$('chat-form').onsubmit({preventDefault(){}});
  const stopping=f.$('chat-stop').onclick();
  if(switchAway){f.setSession('B');f.chat.switchWorkspace();await f.chat.refresh();}
  else {reply.resolve({status:'SUCCEEDED'});await sending;}
  const status=f.$('llm-status').textContent;
  cancel.reject(Error('obsolete cancellation error'));await stopping;
  assert.equal(f.$('llm-status').textContent,status);
  if(switchAway){reply.resolve({status:'SUCCEEDED'});await sending;}
 }
});

test('current cancellation failure remains visible and permits retry',async()=>{
 const f=fixture();await f.chat.refresh();const reply=deferred();let calls=0;
 f.transport(()=>reply.promise);
 f.api(async path=>{if(!path.includes('/cancel?'))return [];calls++;if(calls===1)throw Error('retry cancellation');return {cancelled:true};});
 f.$('chat-input').value='request';const sending=f.$('chat-form').onsubmit({preventDefault(){}});
 await f.$('chat-stop').onclick();
 assert.equal(f.$('llm-status').textContent,'retry cancellation');assert.equal(f.$('chat-stop').disabled,false);
 await f.$('chat-stop').onclick();assert.equal(calls,2);assert.equal(f.$('chat-stop').disabled,true);
 reply.resolve({status:'CANCELLED'});await sending;
 assert.equal(f.$('chat-stop').hidden,true);
});

test('partial attachment failure preserves successes, allows retry and isolates background errors',async()=>{
 const original=globalThis.FileReader;
 globalThis.FileReader=class {readAsDataURL(){this.result='data:image/png;base64,AA==';this.onload();}};
 const settle=()=>new Promise(resolve=>setImmediate(resolve));
 const file=name=>({name,type:'image/png',size:1});
 try {
  const f=fixture();await f.chat.refresh();let calls=[];
  f.api(async(path,body)=>{calls.push(body.filename);if(body.filename==='bad.png')throw Error('offline');return {id:body.filename,filename:body.filename};});
  f.$('chat-upload').files=['ok.png','bad.png','later.png'].map(file);f.$('chat-upload').onchange();await settle();
  assert.deepEqual(calls,['ok.png','bad.png']);
  assert.equal(f.$('chat-attachments').children.length,1);
  assert.match(f.$('chat-upload-status').textContent,/bad.png.*offline.*1개 첨부됨, 2개 미첨부/);
  assert.equal(f.$('chat-upload').disabled,false);
  f.api(async(path,body)=>({id:body.filename,filename:body.filename}));
  f.$('chat-upload').files=['bad.png','later.png'].map(file);f.$('chat-upload').onchange();await settle();
  assert.equal(f.$('chat-attachments').children.length,3);
  const pending=deferred();f.api(()=>pending.promise);
  f.$('chat-upload').files=[file('background.png')];f.$('chat-upload').onchange();await settle();
  f.setSession('B');f.chat.switchWorkspace();pending.reject(Error('private A error'));await settle();
  assert.equal(f.$('chat-upload-status').textContent,'');assert.equal(f.$('chat-attachments').children.length,0);
  f.setSession('A');f.chat.switchWorkspace();assert.equal(f.$('chat-attachments').children.length,3);
 } finally {if(original===undefined)delete globalThis.FileReader;else globalThis.FileReader=original;}
});

test('corrupt saved options cannot lock chat and valid per-model settings survive switching',async()=>{
 const original=globalThis.localStorage,values=new Map();
 globalThis.localStorage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value)};
 try {
  const f=fixture();await f.chat.refresh();
  f.api(async()=>({capabilities:['vision','thinking'],thinking:{values:[false,true]}}));
  for(const bad of ['null','[]','42','"text"','{']) {
   values.set('image-maker.options.vision',bad);
   await f.$('llm-model').onchange();
   assert.equal(f.$('refresh-llm').disabled,false);
   assert.equal(f.$('llm-think').disabled,false);
   assert.equal(f.$('llm-temperature').value,'');
  }
  f.$('llm-temperature').value='0.7';f.$('llm-options').value='{"num_thread":8}';f.$('llm-temperature').onchange();
  f.$('llm-model').value='second';await f.$('llm-model').onchange();
  assert.equal(f.$('llm-temperature').value,'');
  f.$('llm-model').value='vision';await f.$('llm-model').onchange();
  assert.equal(f.$('llm-temperature').value,'0.7');assert.equal(f.$('llm-options').value,'{"num_thread":8}');
 } finally {if(original===undefined)delete globalThis.localStorage;else globalThis.localStorage=original;}
});

test('slow model list refresh cannot overwrite a newer selection or reasoning',async()=>{
 for(const fail of [false,true]) {
  const f=fixture(),pending=deferred();
  f.api(path=>path==='/api/llm/models'?pending.promise:Promise.resolve({capabilities:['vision'],thinking:{values:['high']}}));
  const refresh=f.$('refresh-llm').onclick();
  const refreshDisabled=f.$('refresh-llm').disabled;
  f.$('llm-model').value='new';await f.$('llm-model').onchange();
  if(fail)pending.reject(Error('obsolete list error'));
  else pending.resolve({models:[{name:'old'}]});
  await refresh;
  assert.equal(f.$('llm-model').value,'new');
  assert.equal(refreshDisabled,true);
  assert.equal(f.$('llm-think').options[1].value,'"high"');
  assert.notEqual(f.$('llm-status').textContent,'obsolete list error');
  assert.equal(f.$('refresh-llm').disabled,false);
 }
});

test('failed list request releases controls and a later refresh works',async()=>{
 const f=fixture();
 f.api(async()=>{throw Error('offline');});
 await f.$('refresh-llm').onclick();
 assert.equal(f.$('refresh-llm').disabled,false);
 f.api(async path=>path==='/api/llm/models'?{models:[{name:'vision'}]}:{capabilities:['vision'],thinking:{values:[true]}});
 await f.$('refresh-llm').onclick();
 assert.equal(f.$('llm-model').value,'vision');
 assert.equal(f.$('llm-think').options[1].value,'true');
 assert.equal(f.$('refresh-llm').disabled,false);
});

test('late failed model metadata cannot erase the newly selected reasoning options',async()=>{
 const f=fixture(),old=deferred();
 f.api(path=>path.endsWith('vision')?old.promise:Promise.resolve({capabilities:['vision'],thinking:{values:['medium']}}));
 const loading=f.$('llm-model').onchange();
 f.$('llm-model').value='new';await f.$('llm-model').onchange();
 old.reject(Error('old model unavailable'));await loading;
 assert.equal(f.$('llm-think').options.length,2);
 assert.equal(f.$('llm-think').options[1].value,'"medium"');
 assert.notEqual(f.$('llm-status').textContent,'old model unavailable');
 assert.equal(f.$('refresh-llm').disabled,false);
 const next=deferred();f.api(()=>next.promise);
 const pending=f.$('llm-model').onchange();f.$('llm-model').value='';await f.$('llm-model').onchange();
 assert.equal(f.$('refresh-llm').disabled,false);
 next.reject(Error('obsolete'));await pending;
 assert.equal(f.$('llm-think').options.length,0);
});

test('paged history keeps older turns on refresh and ignores responses after Workspace changes',async()=>{
  const f=fixture(),turn=id=>({id,message:id,reply:'reply',status:'SUCCEEDED'});
  const calls=[];
  f.api(async path=>{calls.push(path);return {items:[turn('new')],before:'new',has_more:true};});
  await f.chat.refresh();assert.equal(f.$('chat-older').hidden,false);
  Object.defineProperty(f.$('chat-history'),'scrollHeight',{get(){return this.children.length*100;}});
  f.$('chat-history').scrollTop=30;
  f.api(async path=>{calls.push(path);return {items:[turn('old')],before:null,has_more:false};});
  await f.$('chat-older').onclick();assert.equal(f.$('chat-history').children.length,2);
  assert.equal(f.$('chat-history').scrollTop,130);assert.equal(f.$('chat-older').hidden,true);
  assert.match(calls[1],/before=new/);
  f.api(async()=>({items:[turn('new'),turn('newest')],before:'new',has_more:true}));
  await f.chat.refresh();assert.equal(f.$('chat-history').children.length,3);
  assert.equal(f.$('chat-older').hidden,true);
  f.setSession('B');f.chat.switchWorkspace();
  f.api(async()=>({items:[turn('B')],before:'B',has_more:true}));await f.chat.refresh();
  const pending=deferred();f.api(()=>pending.promise);const loading=f.$('chat-older').onclick();
  f.setSession('C');f.chat.switchWorkspace();pending.resolve({items:[turn('private B')],before:null,has_more:false});await loading;
  assert.equal(f.$('chat-history').children.length,0);assert.equal(f.$('chat-older').hidden,true);
});

test('history refresh and older pages preserve unchanged expanded answers, but update changed turns',async()=>{
 const f=fixture(),turn={id:'one',message:'question',reply:'answer',reasoning:'thinking',status:'SUCCEEDED'};
 f.api(async()=>({items:[turn],before:'one',has_more:true}));await f.chat.refresh();
 const original=f.$('chat-history').children[0],details=original.children[1].children[1];details.open=true;
 f.api(async()=>({items:[{...turn}],before:'one',has_more:true}));await f.chat.refresh();
 assert.equal(f.$('chat-history').children[0],original);assert.equal(details.open,true);
 f.api(async()=>({items:[{...turn,id:'older'}],before:null,has_more:false}));await f.$('chat-older').onclick();
 assert.equal(f.$('chat-history').children[1],original);assert.equal(details.open,true);
 f.api(async()=>({items:[{...turn,reply:'updated answer'}],before:null,has_more:false}));await f.chat.refresh();
 assert.notEqual(f.$('chat-history').children[1],original);
 f.setSession('B');f.chat.switchWorkspace();f.api(async()=>({items:[turn],before:null,has_more:false}));await f.chat.refresh();
 assert.notEqual(f.$('chat-history').children[0],original);
});

test('older-page failure keeps cursor for retry and legacy fallback only handles missing endpoint',async()=>{
  const f=fixture();f.api(async()=>({items:[{id:'new',message:'one',status:'SUCCEEDED'}],before:'new',has_more:true}));
  await f.chat.refresh();f.api(async()=>{throw Error('read failed');});
  await f.$('chat-older').onclick();assert.equal(f.$('chat-older').disabled,false);assert.equal(f.$('chat-older').hidden,false);
  const calls=[];const api=async path=>{calls.push(path);if(path.includes('/page?'))throw Object.assign(Error('Not Found'),{status:404});return ['legacy'];};
  assert.deepEqual((await chatPage(api,'A')).items,['legacy']);assert.equal(calls.length,2);
  await assert.rejects(chatPage(async()=>{throw Object.assign(Error('broken'),{status:500});},'A'),/broken/);
  await assert.rejects(chatPage(api,'A','cursor'),/Not Found/);
});

test('workspace switch clears old messages immediately, keeps drafts and gates sends until history arrives',async()=>{
  const f=fixture();await f.chat.refresh();
  f.$('chat-input').value='draft A';f.$('chat-input').oninput();
  assert.equal(f.$('chat-send').disabled,false);
  f.setSession('B');f.chat.switchWorkspace();
  assert.equal(f.$('chat-history').children.length,0);assert.equal(f.$('chat-input').value,'');
  f.$('chat-input').value='draft B';f.$('chat-input').oninput();assert.equal(f.$('chat-send').disabled,true);
  const pending=deferred();f.api(()=>pending.promise);const refresh=f.chat.refresh();
  let sent=false;f.transport(async()=>{sent=true;});await f.$('chat-form').onsubmit({preventDefault(){}});assert.equal(sent,false);
  pending.resolve([]);await refresh;assert.equal(f.$('chat-send').disabled,false);
  f.setSession('A');f.chat.switchWorkspace();assert.equal(f.$('chat-input').value,'draft A');
  f.setSession('B');f.chat.switchWorkspace();assert.equal(f.$('chat-input').value,'draft B');
});

test('late history cannot render into the new workspace; failed history leaves sending disabled',async()=>{
  const f=fixture(),old=deferred();f.api(()=>old.promise);const a=f.chat.refresh();
  f.setSession('B');f.chat.switchWorkspace();f.api(async()=>{throw Error('history unreadable');});
  await assert.rejects(f.chat.refresh(),/unreadable/);
  old.resolve([{message:'private A history',status:'SUCCEEDED'}]);await a;
  assert.equal(f.$('chat-history').children.length,0);assert.equal(f.$('chat-send').disabled,true);
});

test('obsolete history failures cannot overwrite a newer refresh or workspace',async()=>{
 for(const switchSession of [false,true]){
  const f=fixture();await f.chat.refresh();const old=deferred();f.api(()=>old.promise);
  const pending=f.chat.refresh();
  if(switchSession){f.setSession('B');f.chat.switchWorkspace();}
  f.api(async()=>[]);await f.chat.refresh();
  f.$('chat-input').value='current draft';f.$('chat-input').oninput();
  const banner=f.$('llm-status').textContent;
  old.reject(Error('obsolete private history error'));
  await assert.doesNotReject(pending);
  assert.equal(f.$('llm-status').textContent,banner);
  assert.equal(f.$('chat-input').value,'current draft');
  assert.equal(f.$('chat-send').disabled,false);
 }
});

test('failed send restores its draft without overwriting text composed during the request',async()=>{
  for(const next of ['', 'new message']){
    const f=fixture();await f.chat.refresh();const pending=deferred();f.transport(()=>pending.promise);
    f.$('chat-input').value='original message';f.$('chat-input').oninput();
    const send=f.$('chat-form').onsubmit({preventDefault(){}});
    assert.equal(f.$('chat-input').value,'');assert.equal(f.$('refresh-llm').disabled,true);
    f.$('chat-input').value=next;f.$('chat-input').oninput();pending.reject(Error('connection lost'));await send;
    assert.equal(f.$('chat-input').value,next||'original message');assert.equal(f.$('refresh-llm').disabled,false);
  }
});

test('stored FAILED final turns restore the request for retry without replacing a newer draft',async()=>{
  for(const background of [false,true])for(const next of ['', 'new message']){
    const f=fixture();await f.chat.refresh();const pending=deferred();f.transport(()=>pending.promise);
    f.$('chat-input').value='retry this request';f.$('chat-input').oninput();
    const sending=f.$('chat-form').onsubmit({preventDefault(){}});
    f.$('chat-input').value=next;f.$('chat-input').oninput();
    if(background){f.setSession('B');f.chat.switchWorkspace();await f.chat.refresh();f.$('chat-input').value='draft B';f.$('chat-input').oninput();}
    pending.resolve({status:'FAILED',error:'output budget reached'});await sending;
    if(background){assert.equal(f.$('chat-input').value,'draft B');f.setSession('A');f.chat.switchWorkspace();}
    assert.equal(f.$('chat-input').value,next||'retry this request');
  }
});

test('a failed response in a background workspace restores only that workspace draft',async()=>{
  const f=fixture();await f.chat.refresh();const pending=deferred();f.transport(()=>pending.promise);
  f.$('chat-input').value='request A';f.$('chat-input').oninput();const send=f.$('chat-form').onsubmit({preventDefault(){}});
  f.setSession('B');f.chat.switchWorkspace();await f.chat.refresh();
  f.$('chat-input').value='draft B';f.$('chat-input').oninput();pending.reject(Error('offline'));await send;
  assert.equal(f.$('chat-input').value,'draft B');
  f.setSession('A');f.chat.switchWorkspace();assert.equal(f.$('chat-input').value,'request A');
});

test('reasoning streams separately and survives refreshing the active workspace',async()=>{
 const f=fixture();await f.chat.refresh();const pending=deferred();let emit;
 f.transport((body,onEvent)=>{emit=onEvent;return pending.promise;});
 f.$('chat-input').value='reason about this';
 const send=f.$('chat-form').onsubmit({preventDefault(){}});
 emit({type:'status',code:'cpu_fallback',text:'Ollama CPU 실행 중'});
 emit({type:'reasoning',text:'first '});emit({type:'reasoning',text:'second'});
 assert.equal(f.$('llm-status').textContent,'Ollama CPU 실행 중');
 emit({type:'reply',text:'answer'});
 assert.equal(f.$('llm-status').textContent,'Ollama CPU 실행 중');
 let panel=f.$('chat-history').children.at(-1).children[1].children[1];
 assert.equal(panel.children[1].textContent,'first second');
 await f.chat.refresh();
 panel=f.$('chat-history').children.at(-1).children[1].children[1];
 assert.equal(panel.children[1].textContent,'first second');
 f.setSession('B');f.chat.switchWorkspace();await f.chat.refresh();
 emit({type:'reasoning',text:'private A'});
 assert.equal(f.$('chat-history').children.length,1); // Only B's welcome, no A turn.
 pending.reject(Error('ended'));await send;
});

test('failed-message rewrite preserves a newer draft and ignores stale workspace buttons',async()=>{
 for(const state of ['FAILED','CANCELLED']){
  const f=fixture(),turn={id:'failed',message:'original request',reply:'I changed the prompt.',error:'invalid edit',status:state,context:{session:'A'}};
  f.api(async()=>[turn]);await f.chat.refresh();
  const assistant=f.$('chat-history').children[0].children[1],retry=assistant.children.at(-1);
  assert.match(assistant.children[0].textContent,state==='FAILED'?/응답 실패/:/응답 중단/);
  assert.equal(assistant.querySelector('.chat-proposal'),null);
  f.$('chat-input').value='new draft';f.$('chat-input').oninput();await retry.onclick();
  assert.equal(f.$('chat-input').value,'new draft');assert.match(f.$('llm-status').textContent,/작성 중/);
  f.$('chat-input').value='';f.$('chat-input').oninput();await retry.onclick();assert.equal(f.$('chat-input').value,turn.message);
  f.setSession('B');f.chat.switchWorkspace();f.$('chat-input').value='draft B';f.$('chat-input').oninput();await retry.onclick();
  assert.equal(f.$('chat-input').value,'draft B');
  f.setSession('A');f.chat.switchWorkspace();assert.equal(f.$('chat-input').value,turn.message);
 }
});

test('failed-message rewrite cannot fill the composer during a new response',async()=>{
 const f=fixture(),turn={id:'old',message:'old request',status:'FAILED',error:'failed',context:{session:'A'}};
 f.api(async()=>[turn]);await f.chat.refresh();const retry=f.$('chat-history').children[0].children[1].children.at(-1);
 const pending=deferred();f.transport(()=>pending.promise);f.$('chat-input').value='new request';f.$('chat-input').oninput();
 const sending=f.$('chat-form').onsubmit({preventDefault(){}});await retry.onclick();assert.equal(f.$('chat-input').value,'');
 pending.resolve({status:'SUCCEEDED',update:false});await sending;
});

test('automatic apply keeps a committed result when only its listing refresh warns',async()=>{
 const f=fixture(),context={session:'A',prompt_id:'p',expected_revision:1,positive:'original',negative:'keep',settings:{}};
 f.setContext(context);await f.chat.refresh();f.$('chat-action').value='generate';
 f.transport(async body=>({id:body.request_id,status:'SUCCEEDED',update:true,context,positive:'revised',negative:'keep'}));
 f.apply(async()=>{f.setContext({...context,expected_revision:2,positive:'revised'});return '저장 완료 · 목록 갱신 실패';});
 f.$('chat-input').value='edit request';await f.$('chat-form').onsubmit({preventDefault(){}});
 assert.equal(f.$('chat-input').value,'');assert.equal(f.$('llm-status').textContent,'저장 완료 · 목록 갱신 실패');
});

test('applied history and unreadable application receipts never offer duplicate application',async()=>{
 for(const application of [{application:{revision:2,job_id:null}},{application:{revision:2,job_id:'job'}},{application_error:'hash mismatch'}]){
  const f=fixture();f.api(async()=>[{id:'applied',message:'edit',reply:'edited',status:'SUCCEEDED',update:true,positive:'revised',negative:'',context:{session:'A'},...application}]);
  await f.chat.refresh();const card=f.$('chat-history').querySelector('.chat-proposal');
  assert.equal(card.children.length,4);assert.match(card.children[1].textContent,application.application?/반영 완료/:/확인하지 못/);
 }
});
