import test from 'node:test';
import assert from 'node:assert/strict';
import {setupConversations} from '../backend/app/web/conversations.mjs';

function fixture(api) {
  const elements=new Map(),saved=new Map();let session='workspace-a',selected=0;
  const element=()=>({value:'',checked:false,open:true,children:[],setAttribute(){},addEventListener(){},replaceChildren(...children){this.children=children;}});
  const doc={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element,addEventListener(){}};
  const controller=setupConversations({api,doc,getSession:()=>session,onSelect:async()=>{selected++;},storage:{getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)}});
  return {controller,$:doc.getElementById.bind(doc),saved,setSession:value=>session=value,selections:()=>selected};
}
test('conversation controls retain edited titles and archive/restore the selected conversation',async()=>{
  let row={id:'one',title:'Original',revision:1,archived:false};const requests=[];
  const f=fixture(async(url,body)=>{if(!body)return [row];requests.push(body);row={...row,...body,revision:row.revision+1};return row;});
  await f.controller.load();f.$('conversation-title').value='Renamed';
  await f.$('conversation-rename').onclick();assert.equal(requests[0].title,'Renamed');
  await f.$('conversation-archive').onclick();assert.equal(f.controller.archived(),true);assert.equal(f.$('conversation-archived').checked,true);
  await f.$('conversation-archive').onclick();assert.equal(f.controller.archived(),false);
  assert.equal(f.selections(),3);assert.equal(f.saved.get('image-maker.conversation.workspace-a'),'one');
});
test('a late conversation list cannot replace the current workspace selection',async()=>{
  let resolveOld;const old=new Promise(resolve=>{resolveOld=resolve;});
  const f=fixture(async url=>url.includes('workspace-a')?old:[{id:'new',title:'New',revision:1}]);
  const loading=f.controller.load();assert.equal(f.controller.ready(),false);
  f.setSession('workspace-b');await f.controller.load();
  resolveOld([{id:'old',title:'Old',revision:1}]);assert.equal(await loading,false);
  assert.equal(f.controller.id(),'new');assert.equal(f.$('conversation-current').textContent,'New');
});
