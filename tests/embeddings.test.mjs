import test from 'node:test';
import assert from 'node:assert/strict';
import {setupEmbeddings} from '../backend/app/web/embeddings.mjs';
import {sameDraft,proposalSettings} from '../backend/app/web/chat.mjs';
import {describePipeline,modelChoice,modelInventoryRows} from '../backend/app/web/pipeline.mjs';

function fixture() {
  const nodes=new Map();
  const doc={getElementById:id=>nodes.get(id),createElement:tag=>({tag,ownerDocument:doc,children:[],value:'',textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},setAttribute(name,value){this[name]=value;},get options(){return this.children.flatMap(child=>child.tag==='optgroup'?child.children:[child]);}})};
  for(const id of ['embedding','embedding-list','embedding-count','embedding-status'])nodes.set(id,doc.createElement('div'));
  const models=Array.from({length:5},(_,i)=>({id:`e${i}`,name:`style${i}`,path:`embeddings/style${i}.safetensors`,component:'embedding',profile:{family:'sdxl',tokens:[`<s${i}>`]}}));
  const state={base:{id:'base',available:true,family:'sdxl_candidate'},busy:false,changes:0,copied:[],failCopy:false};
  const ui=setupEmbeddings({doc,inventory:()=>models,checkpoint:()=>state.base,isBusy:()=>state.busy,changed:()=>state.changes++,copy:async text=>{if(state.failCopy)throw Error('denied');state.copied.push(text);}});
  return {ui,state,models,$:id=>nodes.get(id)};
}

test('embedding choices enforce limits and busy state; token copy never edits the draft',async()=>{
  const {ui,state,$}=fixture();
  for(let i=0;i<4;i++)assert.equal(ui.add(`e${i}`),true);
  assert.equal(ui.add('e4'),false);assert.equal(ui.add('e0'),false);
  assert.equal($('embedding').disabled,true);assert.equal(state.changes,4);
  const rows=$('embedding-list').children;
  await rows[0].children[1].onclick();assert.deepEqual(state.copied,['<s0>']);assert.equal(state.changes,4);
  state.failCopy=true;await rows[0].children[1].onclick();assert.match($('embedding-status').textContent,/복사하지 못/);
  state.busy=true;ui.refresh();rows[0].children[2].onclick();assert.equal(ui.get().length,4);assert.equal(ui.add('e4'),false);
  state.busy=false;$('embedding-list').children[0].children[2].onclick();assert.deepEqual(ui.get(),['e1','e2','e3']);assert.equal(state.changes,5);
  const copy=ui.get();copy.pop();assert.equal(ui.get().length,3);
});

test('restoration preserves missing and incompatible selections without silently deleting them',()=>{
  const {ui,state,$}=fixture();const saved=['e0','gone'];ui.restore(saved);saved.pop();
  state.base={id:'z',available:true,family:'zimage'};ui.refresh();
  assert.deepEqual(ui.get(),['e0','gone']);assert.equal(state.changes,0);
  assert.match($('embedding-list').children[0].children[0].children[2].textContent,/SD1\/SD2\/SDXL/);
  assert.match($('embedding-list').children[1].children[0].children[0].textContent,/누락/);
  assert.equal(ui.add('e1'),false);ui.restore();assert.deepEqual(ui.get(),[]);
});

test('embedding selection appears in composition and selected inventory, with incompatible families blocked',()=>{
  const {models,state}=fixture();const settings={model:'base',width:768,height:1024,steps:20,guidance:5,flow:'txt2img',embeddings:['e0']};
  const inventory=[{...state.base,path:'checkpoint/base.safetensors'},...models];
  assert.equal(modelChoice(models[0]).field,'embedding');
  assert.match(describePipeline(settings,inventory).components.find(c=>c.label==='임베딩').value,/style0/);
  assert.ok(describePipeline({...settings,embeddings:['gone']},inventory).errors.length);
  assert.ok(describePipeline({...settings,embeddings:['e0','e0']},inventory).errors.length);
  assert.deepEqual(modelInventoryRows(inventory,{settings,type:'embedding',scope:'selected'}).map(row=>row.model.id),['e0']);
});

test('chat compares embedding values, accepts legacy empty settings and forwards tokens without mutation',async()=>{
  const context={session:'s',positive:'<s0><s1>',negative:'blur',settings:{model:'base',prompt_mode:'native'}};
  assert.equal(sameDraft(context,{...context,settings:{...context.settings,embeddings:[]}}),true);
  const selected={...context,settings:{...context.settings,embeddings:['e0']}};
  assert.equal(sameDraft(selected,structuredClone(selected)),true);assert.equal(sameDraft(context,selected),false);
  const turn={context:selected,positive:context.positive,negative:context.negative},before=structuredClone(turn),calls=[];
  const settings=await proposalSettings(turn,{family:'sdxl_candidate'},async(path,body)=>{calls.push(body);return {truncated:true};});
  assert.equal(settings.prompt_mode,'weighted');assert.deepEqual(settings.embeddings,['e0']);
  assert.deepEqual(calls.map(c=>c.embeddings),[['e0'],['e0']]);assert.deepEqual(turn,before);
});
