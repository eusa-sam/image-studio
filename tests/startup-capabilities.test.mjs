import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
const start=source.slice(source.indexOf('async function start()'),source.indexOf('export const ready = start();'));
test('optional capability failure cannot block models, workspace and chat startup',async()=>{
 for(const mode of ['supported','legacy','failed','empty']) {
  const nodes=new Map(),calls=[];
  const $=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id);};
  const context=vm.createContext({$,originalInputSupported:true,api:async()=>{if(mode==='failed')throw Error('HTTP 503');return mode==='empty'?null:mode==='legacy'?{}:{components:{schemas:{Settings:{properties:{source_artifact:{}}}}}};},scanModels:async()=>calls.push('models'),refreshWorkspaceList:async()=>calls.push('workspaces'),changeSession:async()=>calls.push('workspace'),chat:{connect:async()=>calls.push('chat')},renderComposition(){},refreshImageRuntime:async()=>calls.push('runtime'),setInterval(){calls.push('polling');}});
  vm.runInContext(start,context);await vm.runInContext('start()',context);
  assert.deepEqual(calls,['models','workspaces','workspace','chat','runtime','polling']);
  assert.equal(context.originalInputSupported,mode==='supported');assert.equal($('source-version-status').hidden,mode==='supported');
  if(mode==='failed')assert.match($('source-version-status').textContent,/503/);
 }
});
test('essential model loading errors still reach startup error handling',async()=>{
 const context=vm.createContext({$:()=>({}),originalInputSupported:false,api:async()=>({}),scanModels:async()=>{throw Error('models unavailable');}});
 vm.runInContext(start,context);await assert.rejects(vm.runInContext('start()',context),/models unavailable/);
});
