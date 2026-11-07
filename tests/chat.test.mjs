import test from 'node:test';
import assert from 'node:assert/strict';
import {sameDraft, streamChat, preferredModel, proposalSettings, validateAttachments, bindReferenceInput, tokenOptions,chatOptions} from '../backend/app/web/chat.mjs';

test('advanced options preserve blank defaults and reject duplicates or malformed JSON',()=>{
 assert.deepEqual(chatOptions({temperature:'0',seed:'42'},'{"num_thread":8,"stop":["END"]}'),{temperature:0,seed:42,num_thread:8,stop:['END']});
 assert.deepEqual(chatOptions({temperature:''},''),{});
 for(const [fields,extra] of [[{},'[]'],[{},'null'],[{},'{'],[{top_k:'1.5'},''],[{temperature:'Infinity'},''],[{temperature:'.5'},'{"temperature":1}']])assert.throws(()=>chatOptions(fields,extra));
});

test('user token budgets preserve defaults, unlimited and large custom values',()=>{
 assert.deepEqual(tokenOptions('',''),{num_predict:null,num_ctx:null});
 assert.deepEqual(tokenOptions('-1','65536'),{num_predict:-1,num_ctx:65536});
 assert.deepEqual(tokenOptions('131072','262144'),{num_predict:131072,num_ctx:262144});
 for(const [output,context] of [['0',''],['-2',''],['1.5',''],['','0'],['x','']])assert.throws(()=>tokenOptions(output,context));
});

test('attachment validation checks the entire batch before uploading',()=>{
 const png={type:'image/png',size:100};
 assert.doesNotThrow(()=>validateAttachments([png],3));
 assert.throws(()=>validateAttachments([png,png],3),/4개/);
 assert.throws(()=>validateAttachments([png,{type:'text/plain',size:10}],0),/PNG/);
 assert.throws(()=>validateAttachments([{...png,size:12*1024*1024+1}],0),/12MiB/);
});

test('picker, nested drag and paste use the same upload path; text paste is preserved',()=>{
 const classes=new Set(),events={},inputEvents={},uploads=[];
 const panel={classList:{add:n=>classes.add(n),remove:n=>classes.delete(n)},addEventListener:(n,f)=>events[n]=f};
 const input={addEventListener:(n,f)=>inputEvents[n]=f},button={};let clicks=0,prevented=0;
 const image={type:'image/png',size:20},picker={files:[image],value:'old',click:()=>clicks++};
 bindReferenceInput(panel,input,picker,button,files=>uploads.push(files));
 button.onclick();assert.equal(clicks,1);picker.onchange();assert.equal(picker.value,'');
 const drag={dataTransfer:{types:['Files'],files:[image]},preventDefault:()=>prevented++};
 events.dragenter(drag);events.dragenter(drag);events.dragleave(drag);assert.ok(classes.has('is-dragging'));
 events.dragover(drag);assert.equal(drag.dataTransfer.dropEffect,'copy');events.drop(drag);assert.equal(classes.size,0);
 inputEvents.paste({clipboardData:{files:[image]},preventDefault:()=>prevented++});
 inputEvents.paste({clipboardData:{files:[]},preventDefault:()=>assert.fail('text paste must work')});
 assert.deepEqual(uploads,[[image],[image],[image]]);assert.equal(prevented,5);
});

test('model selection preserves user preference and otherwise chooses a moderate local size', () => {
  const models=[{name:'large',size:18e9},{name:'tiny',size:3e9},{name:'medium',size:6e9}];
  assert.equal(preferredModel(models,'large'),'large');
  assert.equal(preferredModel(models,'missing'),'medium');
  assert.equal(preferredModel([],''),'');
});

test('proposal cannot overwrite a changed workspace, prompt, revision, text or settings', () => {
  const original={session:'workspace',prompt_id:'prompt',expected_revision:2,positive:'tea',negative:'blur',settings:{model:'model',seed:42}};
  assert.equal(sameDraft(original,{...original,settings:{seed:42,model:'model'}}),true);
  for (const patch of [{session:'other'}, {prompt_id:'other'}, {expected_revision:3}, {positive:'user edit'},
    {negative:'user edit'}, {settings:{model:'model',seed:43}}, {settings:{model:'model',seed:42,width:512}}]) {
    assert.equal(sameDraft(original,{...original,...patch}),false);
  }
});

test('LoRA values are compared across separate draft snapshots', () => {
  const a={session:'s',prompt_id:'p',expected_revision:1,positive:'',negative:'',settings:{loras:[{model:'l',weight:0.5}]}};
  const b=JSON.parse(JSON.stringify(a));assert.equal(sameDraft(a,b),true);
  b.settings.loras[0].weight=1;assert.equal(sameDraft(a,b),false);
});

test('stream transport decodes split Korean UTF-8 and preserves a final turn', async () => {
  const encoder=new TextEncoder();
  const bytes=encoder.encode(JSON.stringify({type:'reply',text:'안녕🙂'})+'\n'+JSON.stringify({type:'done',turn:{status:'SUCCEEDED'}})+'\n');
  const events=[];
  const fetcher=async()=>({ok:true,body:new ReadableStream({start(controller){for(let i=0;i<bytes.length;i+=2)controller.enqueue(bytes.slice(i,i+2));controller.close();}})});
  const turn=await streamChat({},event=>events.push(event),fetcher);
  assert.equal(events[0].text,'안녕🙂');assert.equal(turn.status,'SUCCEEDED');
});
test('truncated streams and backend errors never become successful proposals', async () => {
  for(const event of [{type:'reply',text:'unfinished'},{type:'error',text:'busy'}]) {
    const fetcher=async()=>({ok:true,body:new ReadableStream({start(c){c.enqueue(new TextEncoder().encode(JSON.stringify(event)+'\n'));c.close();}})});
    await assert.rejects(streamChat({},()=>{},fetcher),event.type==='error' ? /busy/ : /연결/);
  }
});


test('long proposals enable weighted encoding for SD only, validate its budget and preserve original context',async()=>{
 const turn={context:{settings:{model:'m',prompt_mode:'native'}},positive:'long scene',negative:''};
 const calls=[];const api=async(path,body)=>{calls.push(body);return {truncated:true};};
 assert.equal((await proposalSettings(turn,{family:'sdxl_candidate'},api)).prompt_mode,'weighted');
 assert.deepEqual(calls.map(c=>c.mode),['native','weighted']);
 assert.equal(turn.context.settings.prompt_mode,'native');
 calls.length=0;
 assert.equal((await proposalSettings(turn,{family:'anima'},api)).prompt_mode,'native');assert.equal(calls.length,0);
 assert.equal((await proposalSettings(turn,{family:'krea2'},api)).prompt_mode,'native');assert.equal(calls.length,0);
 assert.equal((await proposalSettings(turn,{family:'flux2'},api)).prompt_mode,'native');assert.equal(calls.length,0);
 assert.equal((await proposalSettings(turn,{family:'zimage'},api)).prompt_mode,'native');assert.equal(calls.length,0);
 await assert.rejects(proposalSettings(turn,{family:'sd1_candidate'},async(path,body)=>{if(body.mode==='weighted')throw Error('600 token limit');return {truncated:true};}),/600/);
 assert.equal((await proposalSettings(turn,{family:'sd2_candidate'},async()=>({truncated:false}))).prompt_mode,'native');
});
import {reasoningPanel} from '../backend/app/web/chat.mjs';
test('reasoning stays separate, preserves disclosure state and renders model text literally',()=>{
 const doc={createElement:()=>({children:[],append(...items){this.children.push(...items);}})};
 const panel=reasoningPanel(doc);
 assert.equal(panel.element.hidden,true);
 panel.update('<script>not markup</script>',true);
 assert.equal(panel.element.hidden,false);
 assert.equal(panel.element.children[1].textContent,'<script>not markup</script>');
 assert.match(panel.element.children[0].textContent,/생각하는 중/);
 panel.element.open=true;
 panel.update('completed',false);
 assert.equal(panel.element.open,true);
 assert.equal(panel.element.children[0].textContent,'생각 과정 보기');
});
