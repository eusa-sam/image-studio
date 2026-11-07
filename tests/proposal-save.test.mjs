import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {sameDraft} from '../backend/app/web/chat.mjs';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
function fixture(failure){
 const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{value:'',disabled:false});return nodes.get(id);};
 $('positive').value='original';$('negative').value='keep';
 let handler,posts=0,prompts=0,jobs=0,tab;const notices=[];
 const c=vm.createContext({$,workspaceReady:true,busy:false,revision:1,promptId:'p',modelInventory:[],sameDraft,
  generationReference:{pending:()=>false},document:{querySelectorAll:()=>[$('positive'),$('negative')]},
  setupChat:options=>{handler=options.apply;return {};},proposalSettings:async turn=>turn.context.settings,
  api:async()=>{posts++;if(failure==='save')throw Error('save failure');return {revision:2};},
  refreshPrompts:async()=>{prompts++;if(failure==='prompts')throw Error('prompt listing failed');},
  refreshJobs:async()=>{jobs++;if(failure==='jobs')throw Error('job listing failed');},
  showStudio:value=>tab=value,notice:text=>notices.push(text),updateFlow(){}});
 c.context=()=>({session:'A',prompt_id:'p',expected_revision:c.revision,positive:$('positive').value,negative:$('negative').value,settings:{model:'base',prompt_mode:'native'}});
 const turn={id:'request',context:c.context(),positive:'revised',negative:'keep'};
 vm.runInContext(source.slice(source.indexOf('const chat = setupChat('),source.indexOf('async function api(')),c);
 return {c,$,turn,notices,apply:generate=>handler(turn,generate),counts:()=>({posts,prompts,jobs,tab})};
}

test('saved chat proposal stays applied when either subsequent listing fails',async()=>{
 for(const failure of ['prompts','jobs'])for(const generate of [false,true]){
  const f=fixture(failure);const warning=await f.apply(generate);
  assert.match(warning,/저장/);assert.match(warning,/목록/);
  assert.equal(f.c.revision,2);assert.equal(f.$('positive').value,'revised');assert.equal(f.$('negative').value,'keep');
  assert.equal(f.c.busy,false);assert.equal(f.$('positive').disabled,false);
  assert.deepEqual(f.counts(),{posts:1,prompts:1,jobs:1,tab:generate?'results':'create'});
  await assert.rejects(f.apply(generate),/변경/);assert.equal(f.counts().posts,1);
 }
});

test('a rejected save preserves the original editor and does not refresh lists',async()=>{
 const f=fixture('save');await assert.rejects(f.apply(true),/save failure/);
 assert.equal(f.c.revision,1);assert.equal(f.$('positive').value,'original');assert.equal(f.c.busy,false);
 assert.deepEqual(f.counts(),{posts:1,prompts:0,jobs:0,tab:undefined});
});

test('direct editor reports a committed save separately from a listing failure',async()=>{
 for(const failure of ['prompts','jobs'])for(const generating of [false,true]){
  const f=fixture(failure);Object.assign(f.c,{session:'A',lastSubmission:null,crypto:{randomUUID:()=> 'id'},settings:()=>({model:'base'})});
  vm.runInContext(source.slice(source.indexOf("$('editor').onsubmit"),source.indexOf("$('sessions').onchange")),f.c);
  await f.$('editor').onsubmit({preventDefault(){},submitter:{id:generating?'generate':'save'}});
  assert.equal(f.c.revision,2);assert.equal(f.c.busy,false);assert.equal(f.$('positive').disabled,false);
  assert.equal(f.counts().posts,1);assert.equal(f.counts().prompts,1);assert.equal(f.counts().jobs,1);
  assert.match(f.notices.at(-1),/저장/);assert.match(f.notices.at(-1),/목록 갱신 실패/);
 }
});
