import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
const start=source.indexOf('async function refreshJobs()');
const header=source.slice(start,source.indexOf('  const signature =',start))+'}';
test('older job refresh success and failure cannot overwrite current result or workspace',async()=>{
 for(const failure of [false,true]){
  const requests=[];const context=vm.createContext({session:'A',jobRefreshVersion:0,displayedJobs:[],refreshJobTimes(){},api:()=>new Promise((resolve,reject)=>requests.push({resolve,reject}))});
  vm.runInContext(header,context);const first=vm.runInContext('refreshJobs()',context),second=vm.runInContext('refreshJobs()',context);
  const latest=[{id:'job',status:'SUCCEEDED'}];requests[1].resolve(latest);await second;
  if(failure)requests[0].reject(Error('old failure'));else requests[0].resolve([{id:'job',status:'RUNNING'}]);
  await first;assert.equal(context.displayedJobs,latest);
  const other=vm.runInContext('refreshJobs()',context);context.session='B';requests[2].reject(Error('old workspace failure'));await other;assert.equal(context.displayedJobs,latest);
  const current=vm.runInContext('refreshJobs()',context);requests[3].reject(Error('current failure'));await assert.rejects(current,/current failure/);
 }
});
test('automatic refresh waits for every request even when one fails early',async()=>{
 const offset=source.indexOf('async function start()');
 const declaration=source.indexOf('  let polling=false;',offset);
 const begin=declaration>=0?declaration:source.indexOf('  setInterval(',offset);
 const block=source.slice(begin,source.indexOf('\n}',begin));
 let tick,finish,calls=0;const errors=[];
 const context=vm.createContext({setInterval(fn){tick=fn;},refreshJobs:async()=>{calls++;throw Error('jobs failed');},refreshStorageErrors(){},refreshImageRuntime:()=>new Promise(resolve=>finish=resolve),notice:message=>errors.push(message)});
 vm.runInContext(block,context);
 const first=tick();await new Promise(resolve=>setImmediate(resolve));const overlapping=tick();
 assert.equal(calls,1);finish();await first;await overlapping;assert.deepEqual(errors,['jobs failed']);
 const next=tick();assert.equal(calls,2);finish();await next;
});
