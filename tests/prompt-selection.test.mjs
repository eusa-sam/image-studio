import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
const loader=source.slice(source.indexOf('async function loadPrompt('),source.indexOf('function applyPrompt('));
test('prompt selection loads one verified body, locks edits and discards a late Workspace response',async()=>{
 for(const outcome of ['success','failure','changed']){
  const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{disabled:false,value:'new'});return elements.get(id);};
  const editor={disabled:true};let resolve,reject,applied=null,url;
  const pending=new Promise((yes,no)=>{resolve=yes;reject=no;});
  const context=vm.createContext({$,busy:false,session:'A',workspaceEpoch:1,promptId:'old',savedPrompts:[],encodeURIComponent,
   document:{querySelectorAll:()=>[editor]},api:path=>{url=path;return pending;},applyPrompt:value=>{applied=value;},updateFlow(){},newPrompt(){}});
  vm.runInContext(loader,context);const work=vm.runInContext("loadPrompt('new')",context);
  assert.equal(context.busy,true);assert.equal($('sessions').disabled,true);assert.match(url,/\/api\/prompts\/new\?session=A/);
  if(outcome==='changed'){context.session='B';context.workspaceEpoch++;}
  if(outcome==='failure'){reject(Error('hash mismatch'));await assert.rejects(work,/hash mismatch/);assert.equal($('prompts').value,'old');}
  else {resolve({id:'new',positive:'verified'});await work;assert.equal(applied?.positive,outcome==='success'?'verified':undefined);}
  assert.equal(context.busy,false);assert.equal($('sessions').disabled,false);assert.equal(editor.disabled,true);
 }
});
