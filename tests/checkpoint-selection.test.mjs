import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
const scan=source.slice(source.indexOf('async function scanModels()'),source.indexOf('let modelScope'));

test('rescan retains a missing or rejected checkpoint instead of selecting the first available model',async()=>{
 for(const rejected of [false,true]) {
  const model={value:'old',children:[],replaceChildren(...children){this.children=children;this.value=children[0]?.value||'';},append(child){this.children.push(child);}};
  const type={value:'all',replaceChildren(){}};
  const models=[{id:'new',name:'new',available:true,path:'checkpoint/new'}];
  if(rejected)models.push({id:'old',name:'old',available:false,reason:'HASH_MISMATCH',path:'checkpoint/old'});
  const context=vm.createContext({$:id=>id==='model'?model:type,api:async()=>models,option:(value,text)=>({value,textContent:text}),modelScanVersion:0,modelInventory:[],modelTypes:[],refreshComponentChoices(){},renderModels(){},renderComposition(){}});
  vm.runInContext(scan,context);await vm.runInContext('scanModels()',context);
  assert.equal(model.value,'old');
  const retained=model.children.find(row=>row.value==='old');assert.equal(retained.disabled,true);
  assert.match(retained.textContent,rejected?/HASH_MISMATCH/:/누락/);
 }
});

test('late model scans retain current user selection and cannot replace a newer inventory or error',async()=>{
 for(const staleFailure of [false,true]) {
  const model={value:'old',children:[],replaceChildren(...children){this.children=children;this.value=children[0]?.value||'';},append(node){this.children.push(node);}};
  const type={value:'all',replaceChildren(){}};const pending=[];
  const context=vm.createContext({$:id=>id==='model'?model:type,api:()=>new Promise((resolve,reject)=>pending.push({resolve,reject})),option:(value,text)=>({value,textContent:text}),modelScanVersion:0,modelInventory:[],modelTypes:[],refreshComponentChoices(){},renderModels(){},renderComposition(){}});
  vm.runInContext(scan,context);const first=vm.runInContext('scanModels()',context);
  model.value='chosen';const second=vm.runInContext('scanModels()',context);model.value='latest';
  const current=[{id:'latest',name:'latest',available:true},{id:'chosen',name:'chosen',available:true}];
  pending[1].resolve(current);await second;assert.equal(model.value,'latest');
  if(staleFailure)pending[0].reject(Error('old failure'));else pending[0].resolve([{id:'old',available:true}]);
  await first;assert.equal(model.value,'latest');assert.equal(context.modelInventory,current);
  const failed=vm.runInContext('scanModels()',context);pending[2].reject(Error('current failure'));
  await assert.rejects(failed,/current failure/);assert.equal(context.modelInventory,current);assert.equal(model.value,'latest');
  const retry=vm.runInContext('scanModels()',context);pending[3].resolve(current);await retry;assert.equal(model.value,'latest');
 }
});

test('restoring a saved prompt keeps its missing checkpoint ID and text',()=>{
 const nodes=new Map();
 const $=id=>{if(!nodes.has(id))nodes.set(id,{value:'',options:[],append(node){this.options.push(node);}});return nodes.get(id);};
 const context=vm.createContext({$,promptId:null,revision:0,option:(value,text)=>({value,textContent:text}),generationReference:{restore(){}},embeddings:{restore(){}},selectLora(){},refreshComponentChoices(){},updateFlow(){}});
 vm.runInContext(source.slice(source.indexOf('function applyPrompt('),source.indexOf('function updateFlow(')),context);
 vm.runInContext("applyPrompt({id:'p',revision:2,positive:'original',negative:'blur',settings:{model:'missing'}})",context);
 assert.equal($('model').value,'missing');assert.equal($('model').options[0].value,'missing');
 assert.match($('model').options[0].textContent,/누락/);assert.equal($('positive').value,'original');
});
