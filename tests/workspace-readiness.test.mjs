import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
test('direct editor submit cannot save the old prompt into a loading workspace',async()=>{
 const editor={};const c=vm.createContext({$:()=>editor,busy:false,workspaceReady:false});
 vm.runInContext(source.slice(source.indexOf("$('editor').onsubmit ="),source.indexOf("$('sessions').onchange =")),c);
 await editor.onsubmit({preventDefault(){}});assert.equal(c.busy,false);
});
test('workspace prompt loading blocks editing until success and retains the block after failure',async()=>{
 for(const fail of [false,true]){
  const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{value:id==='sessions'?'B':'p',selectedOptions:[],replaceChildren(){}});return nodes.get(id);};
  let resolve;const pending=new Promise(r=>resolve=r);
  const c=vm.createContext({$,busy:false,workspaceReady:true,workspaceEpoch:0,resultSelections:new Map([['previous','original.png']]),session:'A',displayedJobs:[],jobSignature:'',workspaceRecords:[],chat:{switchWorkspace(){},async refresh(){}},generationReference:{reset(){}},refreshPrompts:()=>pending,refreshJobs:async()=>{},loadPrompt:async()=>{if(fail)throw Error('load failed');},refreshStorageErrors:async()=>{}});
  vm.runInContext(source.slice(source.indexOf('async function changeSession()'),source.indexOf('async function scanModels()')),c);
  const work=vm.runInContext('changeSession()',c);
  assert.equal(c.workspaceReady,false);assert.equal($('editor').inert,true);assert.equal(c.resultSelections.size,0);
  assert.equal($('workspace-loading').hidden,false);assert.equal($('workspace-retry').disabled,true);
  assert.match($('workspace-loading-status').textContent,/불러오는 중/);
  resolve(true);
  if(fail){await assert.rejects(work,/load failed/);assert.equal(c.workspaceReady,false);assert.equal($('editor').inert,true);assert.match($('workspace-loading-status').textContent,/load failed/);assert.equal($('workspace-retry').disabled,false);c.loadPrompt=async()=>{};await vm.runInContext('changeSession()',c);assert.equal(c.workspaceReady,true);assert.equal($('editor').inert,false);}
  else {await work;assert.equal(c.workspaceReady,true);assert.equal($('editor').inert,false);}
  assert.equal($('workspace-loading').hidden,true);
 }
});

test('an old workspace load failure cannot replace the successful current workspace banner',async()=>{
 const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{value:'B',selectedOptions:[],replaceChildren(){}});return nodes.get(id);};
 let reject;const old=new Promise((_,no)=>reject=no);let calls=0;
 const c=vm.createContext({$,busy:false,workspaceReady:true,workspaceEpoch:0,resultSelections:new Map([['previous','original.png']]),session:'A',displayedJobs:[],jobSignature:'',workspaceRecords:[],chat:{switchWorkspace(){},async refresh(){}},generationReference:{reset(){}},refreshPrompts:()=>++calls===1?old:Promise.resolve(true),refreshJobs:async()=>{},loadPrompt:async()=>{},refreshStorageErrors:async()=>{}});
 vm.runInContext(source.slice(source.indexOf('async function changeSession()'),source.indexOf('async function scanModels()')),c);
 const previous=vm.runInContext('changeSession()',c);$('sessions').value='C';await vm.runInContext('changeSession()',c);
 reject(Error('old B failure'));await previous;
 assert.equal(c.session,'C');assert.equal(c.workspaceReady,true);assert.equal($('workspace-loading').hidden,true);
 assert.doesNotMatch($('workspace-loading-status').textContent,/old B failure/);
});
