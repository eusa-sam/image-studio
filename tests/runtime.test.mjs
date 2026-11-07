import test from 'node:test';
import assert from 'node:assert/strict';
import {runtimeView,setupImageRuntime} from '../backend/app/web/runtime.mjs';

test('only an idle retained image model offers release',()=>{
  assert.equal(runtimeView('warm').release,true);
  assert.equal(runtimeView('cpu').release,true);
  assert.match(runtimeView('cpu').text,/RAM/);
  for(const state of ['running','released','unknown'])assert.equal(runtimeView(state).release,false);
});

test('release is disabled until the server confirms completion',async()=>{
  const status={},button={};let finish;
  const api=async(path)=>path.endsWith('/release')?await new Promise(resolve=>finish=resolve):{state:'warm'};
  const refresh=setupImageRuntime({status,button,api,onError:()=>assert.fail('unexpected error')});
  await refresh();assert.equal(button.hidden,false);
  const released=button.onclick();assert.equal(button.disabled,true);
  finish({state:'released'});await released;
  assert.equal(button.hidden,true);assert.equal(button.disabled,false);
  assert.match(status.textContent,/해제되어/);
});


test('late runtime replies cannot undo a completed release and release requests cannot duplicate',async()=>{
 for(const fail of [false,true]){
  const status={},button={},requests=[];
  const refresh=setupImageRuntime({status,button,api:(path)=>new Promise((resolve,reject)=>requests.push({path,resolve,reject})),onError:()=>assert.fail('unexpected error')});
  const old=refresh(),release=button.onclick();const duplicate=button.onclick(),during=refresh();
  assert.equal(requests.length,2);await duplicate;await during;assert.equal(button.disabled,true);
  requests[1].resolve({state:'released'});await release;
  if(fail)requests[0].reject(Error('old failure'));else requests[0].resolve({state:'warm'});
  await old;assert.match(status.textContent,/해제되어/);assert.equal(button.hidden,true);
 }
});
test('runtime refresh uses newest response and failed release can recover and retry',async()=>{
 const status={},button={},requests=[],errors=[];
 const refresh=setupImageRuntime({status,button,api:path=>new Promise((resolve,reject)=>requests.push({path,resolve,reject})),onError:error=>errors.push(error)});
 const old=refresh(),latest=refresh();requests[1].resolve({state:'cpu'});await latest;requests[0].resolve({state:'warm'});await old;assert.match(status.textContent,/RAM/);
 const release=button.onclick();requests[2].reject(Error('release failed'));await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(errors,['release failed']);assert.equal(button.disabled,false);
 requests[3].resolve({state:'warm'});await release;assert.equal(button.hidden,false);
 const retry=button.onclick();requests[4].resolve({state:'released'});await retry;assert.match(status.textContent,/해제되어/);
 const failure=refresh();requests[5].reject(Error('current failure'));await assert.rejects(failure,/current failure/);
});
