import test from 'node:test';
import assert from 'node:assert/strict';
import {rectanglePixels,setupMask} from '../backend/app/web/mask.mjs';
test('keyboard mask rectangles stay inside image coordinates',()=>{
  assert.deepEqual(rectanglePixels([25,25,50,50],768,1024),[192,256,384,512]);
  for(const values of [[90,0,20,20],[0,0,0,1],[-1,0,10,10],['bad',0,10,10]])assert.throws(()=>rectanglePixels(values,768,1024));
});


test('mask source version selects matching pixels and invalidates the previous mask',async()=>{
 const previousDocument=globalThis.document,previousImage=globalThis.Image;
 const nodes=new Map(),requests=[],loaded=[];let source='original.png';
 const context={clearRect(){},drawImage(){},fillRect(){}};
 const element=()=>({value:'',disabled:false,listeners:{},addEventListener(name,fn){this.listeners[name]=fn;},getContext(){return context;},toDataURL(){return 'data:image/png;base64,eA==';}});
 globalThis.document={getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);},createElement:element};
 globalThis.Image=class {naturalWidth=32;naturalHeight=48;async decode(){loaded.push(this.src);}};
 try {
  setupMask({getContext:()=>({session:'a/b',prompt_id:'p',settings:{source_job:'job',source_artifact:source}}),api:async(path,body)=>{requests.push(body);return {id:'saved'};},changed(){}});
  await nodes.get('mask-load').onclick();assert.match(loaded[0],/original.png/);
  await nodes.get('mask-save').onclick();assert.equal(requests[0].source_artifact,'original.png');assert.equal(nodes.get('mask_id').value,'saved');
  source='image.png';nodes.get('source_artifact').listeners.change();assert.equal(nodes.get('mask_id').value,'');
  await nodes.get('mask-save').onclick();assert.equal(requests.length,1);
  await nodes.get('mask-load').onclick();assert.match(loaded.at(-1),/image.png/);
  await nodes.get('mask-save').onclick();assert.equal(requests[1].source_artifact,'image.png');
 } finally {globalThis.document=previousDocument;globalThis.Image=previousImage;}
});
