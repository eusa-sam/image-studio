import test from 'node:test';
import assert from 'node:assert/strict';
import {referenceSettings,referenceUrl,setupGenerationReference} from '../backend/app/web/reference.mjs';
import {describePipeline,modelChoice,recordedPipeline} from '../backend/app/web/pipeline.mjs';

test('reference origin remains distinct from the img2img source and restores captured flow',()=>{
  assert.deepEqual(referenceSettings('upload:abc'),{reference_job:null,reference_attachment:'abc'});
  assert.deepEqual(referenceSettings('job:def'),{reference_job:'def',reference_attachment:null});
  assert.equal(referenceUrl('', 'workspace'),null);
  assert.equal(referenceUrl('upload:abc','a/b'),'/api/references/abc?session=a%2Fb');
  const components={ip_adapter:{path:'ip_adapter/style.safetensors',profile:{family:'sdxl'}},clip_vision:{path:'clip_vision/encoder.safetensors',profile:{family:'sdxl'}}};
  const settings={model:'wai',flow:'txt2img',prediction_type:'epsilon',width:512,height:512,ip_adapter:'adapter',clip_vision:'vision',ip_adapter_scale:0.7,reference_attachment:'abc'};
  const plan=recordedPipeline({settings,runner:'sdxl_txt2img',components});
  assert.deepEqual(plan.errors,[]);
  assert.match(plan.stages.find(s=>s.key==='reference').value,/사용자 참고/);
  assert.equal(settings.source_job,undefined);
  assert.equal(modelChoice({component:'clip_vision'}).field,'clip_vision');
  assert.ok(describePipeline({...settings,reference_job:'def'},[]).errors.some(e=>e.includes('한 장')));
});

test('reference selection survives job refresh and resets across workspaces',()=>{
  const nodes=new Map();
  const element=()=>({value:'',options:[],classList:{add(){},remove(){}},addEventListener(){},removeAttribute(k){delete this[k];},append(...items){this.options.push(...items);},replaceChildren(...items){this.options=items;this.value=items[0]?.value||'';}});
  const doc={createElement:element,getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);}};
  const ref=setupGenerationReference({doc,api:()=>assert.fail('no upload'),getSession:()=> 'workspace',isBusy:()=>false,changed(){}});
  ref.restore({reference_attachment:'abc'});
  ref.jobs([{id:'def',status:'SUCCEEDED',created_at:'2026-11-07'}]);
  assert.equal(nodes.get('generation-reference').value,'upload:abc');
  assert.match(nodes.get('generation-reference-preview').src,/references\/abc/);
  ref.restore({reference_job:'missing'});ref.jobs([]);
  assert.equal(nodes.get('generation-reference').value,'job:missing');
  ref.reset();
  assert.equal(nodes.get('generation-reference').value,'');
  assert.equal(nodes.get('generation-reference-preview').hidden,true);
});

test('late upload cannot replace Workspace, restored prompt, manual selection or removal',async()=>{
  const OriginalReader=globalThis.FileReader;
  globalThis.FileReader=class {readAsDataURL(){this.result='data:image/png;base64,eA==';this.onload();}};
  try {
    for(const transition of ['workspace','prompt','selection','clear']) {
      let complete,session='first',changes=0;const nodes=new Map();
      const element=()=>({value:'',options:[],classList:{add(){},remove(){}},addEventListener(){},removeAttribute(k){delete this[k];},append(...items){this.options.push(...items);},replaceChildren(...items){this.options=items;this.value=items[0]?.value||'';}});
      const doc={createElement:element,getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);}};
      const ref=setupGenerationReference({doc,getSession:()=>session,isBusy:()=>false,changed:()=>changes++,api:()=>new Promise(resolve=>complete=resolve)});
      nodes.get('generation-reference-file').files=[{name:'image.png',type:'image/png',size:1}];
      nodes.get('generation-reference-file').onchange();
      await new Promise(resolve=>setImmediate(resolve));
      assert.equal(ref.pending(),true);
      if(transition==='workspace'){session='second';ref.reset();}
      else if(transition==='prompt')ref.restore({reference_job:'chosen'});
      else {nodes.get('generation-reference').value=transition==='clear'?'':'job:chosen';nodes.get('generation-reference').onchange();}
      const expectedChanges=changes;
      complete({id:'old-upload',filename:'image.png'});
      await new Promise(resolve=>setImmediate(resolve));
      assert.equal(ref.pending(),false);
      assert.equal(nodes.get('generation-reference').value,['workspace','clear'].includes(transition)?'':'job:chosen');
      assert.equal(changes,expectedChanges);
      assert.equal(nodes.get('generation-reference').options.some(o=>o.value==='upload:old-upload'),false);
    }
  } finally {globalThis.FileReader=OriginalReader;}
});


test('preserved originals remain generated references through refresh and restore',()=>{
 let supported=true;
 assert.deepEqual(referenceSettings('job:def:original.png'),{reference_job:'def',reference_attachment:null,reference_artifact:'original.png'});
 assert.equal(referenceUrl('job:def:original.png','a/b'),'/api/jobs/def/original.png?session=a%2Fb');
 const nodes=new Map();
 const element=()=>({value:'',options:[],classList:{add(){},remove(){}},addEventListener(){},removeAttribute(k){delete this[k];},append(...items){this.options.push(...items);},replaceChildren(...items){this.options=items;this.value=items[0]?.value||'';}});
 const doc={createElement:element,getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);}};
 const ref=setupGenerationReference({doc,api:()=>assert.fail('no upload'),getSession:()=> 'workspace',isBusy:()=>false,supportsOriginal:()=>supported,changed(){}});
 ref.jobs([{id:'def',status:'PARTIAL_FAILED',has_original:true,created_at:'2026-11-07'}]);
 assert.ok(nodes.get('generation-reference').options.some(o=>o.value==='job:def:original.png'));
 assert.ok(!nodes.get('generation-reference').options.some(o=>o.value==='job:def'));
 ref.restore({reference_job:'def',reference_artifact:'original.png'});ref.jobs([]);
 assert.equal(nodes.get('generation-reference').value,'job:def:original.png');
 assert.match(nodes.get('generation-reference-preview').src,/original.png/);
 ref.reset();supported=false;ref.jobs([{id:'old',status:'SUCCEEDED',has_original:true,created_at:'2026-11-07'}]);
 assert.ok(nodes.get('generation-reference').options.some(o=>o.value==='job:old'));
 assert.ok(!nodes.get('generation-reference').options.some(o=>o.value.endsWith(':original.png')));
});
