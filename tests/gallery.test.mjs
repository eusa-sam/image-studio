import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
const block=source.slice(source.indexOf('let galleryPage='),source.indexOf('async function refreshWorkspaceList'));
function setup(){
 const elements=new Map(),requests=[];
 const $=id=>{if(!elements.has(id))elements.set(id,{value:'',textContent:'',disabled:false,replaceChildren(){},addEventListener(){}});return elements.get(id);};
 const context=vm.createContext({$,encodeURIComponent,api:url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject}))});
 vm.runInContext(block,context);
 const flush=()=>new Promise(resolve=>setImmediate(resolve));
 const resolve=index=>requests[index].resolve({items:[],errors:[],page:Number(new URL(requests[index].url,'http://local').searchParams.get('page')),size:24,total:100});
 return {$,requests,context,flush,resolve};
}
test('gallery pagination uses submitted search until a new search is submitted',async()=>{
 const h=setup();h.$('gallery-query').value='WAI';h.$('gallery-search').onsubmit({preventDefault(){}});
 h.resolve(0);await h.flush();
 h.$('gallery-query').value='unfinished ANIMA';h.$('gallery-next').onclick();
 assert.equal(h.requests[1].url,'/api/images?page=2&q=WAI');h.resolve(1);await h.flush();
 h.$('gallery-search').onsubmit({preventDefault(){}});
 assert.equal(h.requests[2].url,'/api/images?page=1&q=unfinished%20ANIMA');h.resolve(2);await h.flush();
});
test('superseded gallery rebuild does not issue another list scan or replace search results',async()=>{
 const h=setup();const pending=vm.runInContext('refreshGallery(true)',h.context);
 h.$('gallery-query').value='latest';h.$('gallery-search').onsubmit({preventDefault(){}});
 h.resolve(1);await h.flush();const status=h.$('gallery-status').textContent;
 h.requests[0].resolve({});await h.flush();
 assert.equal(h.requests.length,2);await pending;
 assert.equal(h.$('gallery-status').textContent,status);
});
