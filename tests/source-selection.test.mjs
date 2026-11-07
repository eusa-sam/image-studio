import test from 'node:test';
import assert from 'node:assert/strict';
import {sameDraft} from '../backend/app/web/chat.mjs';
test('default artifact fields remain compatible with old drafts but changed inputs conflict',()=>{
 const draft={session:'s',prompt_id:'p',expected_revision:1,positive:'x',negative:'',settings:{model:'m'}};
 const current={...draft,settings:{...draft.settings,source_artifact:'image.png',reference_artifact:'image.png'}};
 assert.equal(sameDraft(draft,current),true);assert.equal(sameDraft(current,draft),true);
 assert.equal(sameDraft(draft,{...current,settings:{...current.settings,source_artifact:'original.png'}}),false);
 assert.equal(sameDraft(draft,{...current,settings:{...current.settings,reference_artifact:'original.png'}}),false);
});


import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
test('original result action selects original pixels for SD or Klein and clears stale masks',()=>{
 const start=source.indexOf('    if(job.has_original && originalInputSupported) {'),end=source.indexOf("    if (job.status === 'SUCCEEDED')",start);
 for(const family of ['sdxl','flux2']) {
  const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{value:'old'});return nodes.get(id);};
  let button,reference;
  vm.runInNewContext(source.slice(start,end),{$,job:{id:'job',has_original:true},document:{createElement:()=>({})},article:{append(node){button=node;}},busy:false,workspaceReady:true,originalInputSupported:true,modelInventory:[],imageFamily:()=>family,generationReference:{restore(s){reference=s;}},updateFlow(){},focusSetting(){}});
  button.onclick();
  if(family==='sdxl'){assert.equal($('source_job').value,'job');assert.equal($('source_artifact').value,'original.png');assert.equal($('mask_id').value,'');}
  else {assert.equal(reference.reference_job,'job');assert.equal(reference.reference_artifact,'original.png');}
 }
});
