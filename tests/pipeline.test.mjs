import test from 'node:test';
import assert from 'node:assert/strict';
import {describePipeline, modelChoice, recordedPipeline, modelInventoryRows, modelSelectionSettings} from '../backend/app/web/pipeline.mjs';

test('model selection preserves user steps, seed, dimensions and supported flows',()=>{
 const current={steps:24,width:768,height:1024,seed:123,random_seed:true,flow:'reference_edit',reference_attachment:'reference',source_job:'old',mask_id:'mask',guidance:5,prediction_type:'epsilon',prompt_mode:'weighted',loras:[{model:'style',weight:0.8}]};
 const snapshot=structuredClone(current);
 for(const family of ['flux2','zimage','krea2','sdxl']) {
  const selected=modelSelectionSettings(family,current);
  for(const key of ['steps','width','height','seed','random_seed','loras','reference_attachment'])assert.deepEqual(selected[key],current[key]);
  assert.equal(selected.flow,family==='flux2'||family==='sdxl'?'reference_edit':'txt2img');
  if(family==='flux2'||family==='zimage'){assert.equal(selected.source_job,'');assert.equal(selected.mask_id,'');}
 }
 for(const flow of ['txt2img','img2img','inpaint'])assert.equal(modelSelectionSettings('krea2',{...current,flow}).flow,flow);
 assert.deepEqual(current,snapshot);
});

test('Z-Image choices explain validated preparation variants without guessing from names',()=>{
 const model={id:'z',name:'renamed',family:'zimage',available:true};
 assert.match(modelChoice({...model,variant:'original'}).detail,/원본 번들.*변환 필요/);
 assert.match(modelChoice({...model,variant:'bf16-local-v1'}).detail,/BF16 준비본.*변환 생략/);
 assert.doesNotMatch(modelChoice({...model,name:'bf16'}).detail,/준비본|변환 생략/);
 assert.equal(modelChoice({...model,variant:'bf16-local-v1'}).field,'model');
});

const inventory=[
  {id:'wai',name:'wai.safetensors',path:'checkpoint/wai.safetensors',available:true},
  {id:'krea',name:'krea.safetensors',path:'checkpoint/krea.safetensors',available:false,reason:'unsupported quantization'},
  {id:'lora',name:'style.safetensors',path:'lora/style.safetensors',available:false},
  {id:'vae',name:'sdxl_vae.safetensors',path:'vae/sdxl_vae.safetensors',component:'vae',profile:{scale:1}},
  {id:'up',name:'2x.pth',path:'upscale_models/2x.pth',component:'upscaler',profile:{scale:2}},
];
const base={model:'wai',width:512,height:512,steps:20,guidance:5,seed:42,flow:'txt2img',prediction_type:'epsilon',strength:0.5};

test('Z-Image flow uses native Qwen, no negative and explicit unsupported combinations',()=>{
 const model={id:'z',name:'Z-Image-Turbo',family:'zimage',available:true,path:'checkpoint/z'};
 const settings={...base,model:'z',guidance:0,steps:9,prompt_mode:'native',prediction_type:'flow'};
 const models=[model,...inventory];const plan=describePipeline(settings,models);
 assert.deepEqual(plan.errors,[]);assert.match(plan.stages[0].value,/Negative 미사용/);
 assert.match(plan.stages[1].value,/Qwen3 4B/);assert.match(plan.stages[1].detail,/9 steps/);
 assert.match(modelChoice(model).detail,/Z-Image-Turbo/);
 for(const changed of [{guidance:1},{flow:'reference_edit'},{vae:'vae'},{loras:[{model:'lora',weight:1}]}])assert.ok(describePipeline({...settings,...changed},models).errors.length);
 assert.match(recordedPipeline({runner:'zimage_txt2img',model_path:'checkpoint/z',settings}).stages[1].detail,/Z-Image-Turbo/);
 assert.match(describePipeline({...settings,upscaler:'up'},models).stages.find(s=>s.key==='upscale').detail,/RAM 여유 시 모델 보관/);
});

test('default flow shows built-in CLIP/VAE, skipped upscale, and no applied LoRA', () => {
  const plan=describePipeline(base,inventory);
  assert.deepEqual(plan.errors,[]);
  assert.equal(plan.output,'512 × 512');
  assert.deepEqual(plan.stages.map(s => s.key),['input','clip','sampling','vae','detail','upscale','output']);
  assert.equal(plan.stages[5].skipped,true);
  assert.equal(plan.stages[3].value,'Checkpoint 내장 VAE');
  assert.equal(plan.components.find(c => c.label === 'LoRA').state,'미적용');
});

test('img2img, external VAE and native upscale are reflected together', () => {
  const plan=describePipeline({...base,flow:'img2img',source_job:'source-uuid',vae:'vae',upscaler:'up'},inventory);
  assert.deepEqual(plan.errors,[]);
  assert.match(plan.stages[0].title,/원본/);
  assert.match(plan.stages[0].detail,/VAE 인코딩/);
  assert.equal(plan.stages[3].value,'sdxl_vae.safetensors');
  assert.equal(plan.output,'1024 × 1024');
  assert.equal(plan.stages[5].skipped,false);
});

test('missing/incompatible selections show blockers and 1024 upscale is allowed', () => {
  assert.equal(describePipeline({...base,model:'krea'},inventory).errors.length,1);
  assert.equal(describePipeline({...base,vae:'lora'},inventory).errors.length,1);
  assert.equal(describePipeline({...base,upscaler:'vae'},inventory).errors.length,1);
  assert.deepEqual(describePipeline({...base,upscaler:'up',width:1024},inventory).errors,[]);
  const edit=describePipeline({...base,flow:'img2img',steps:1,strength:0.1},inventory);
  assert.equal(edit.errors.length,2);
});

test('inventory distinguishes selectable components from visible unsupported files', () => {
  assert.equal(modelChoice(inventory[0]).field,'model');
  assert.equal(modelChoice(inventory[1]).field,null);
  assert.equal(modelChoice(inventory[2]).field,null);
  assert.match(modelChoice(inventory[2]).detail,/적용되지/);
  assert.equal(modelChoice(inventory[3]).field,'vae');
  assert.equal(modelChoice(inventory[4]).field,'upscaler');
});

test('inventory preserves scanner failures instead of reporting a missing profile',()=>{
  for(const path of ['lora/broken.safetensors','vae/broken.safetensors','clip/broken.safetensors','detector/broken.onnx']) {
    const model={id:'broken',name:'broken',path,available:false,reason:'구성요소 프로필과 폴더/형식이 맞지 않습니다.'};
    assert.equal(modelChoice(model).field,null);
    assert.equal(modelChoice(model).detail,model.reason);
    assert.equal(modelInventoryRows([model],{settings:base,query:'폴더/형식'}).length,1);
    assert.equal(modelInventoryRows([model],{settings:base,query:'프로필 없음'}).length,0);
  }
});

test('past job uses captured component paths and scale, not current UI selections', () => {
  const plan={settings:{...base,vae:'old-vae',upscaler:'old-up'},model_path:'checkpoint/old.safetensors',components:{
    vae:{path:'vae/old.safetensors',profile:{scale:1}},
    upscaler:{path:'upscale_models/old.pth',profile:{scale:4}},
  }};
  const before=JSON.stringify(plan);
  const display=recordedPipeline(plan);
  assert.equal(display.stages[2].value,'old.safetensors');
  assert.equal(display.stages[3].value,'old.safetensors');
  assert.equal(display.output,'2048 × 2048');
  assert.equal(JSON.stringify(plan),before);
});

test('LoRA selection appears in sampling and captured history', () => {
  const m=[...inventory,{id:'xl',name:'pixel.safetensors',component:'lora'}];
  const s={...base,loras:[{model:'xl',weight:0.7}]};
  assert.equal(modelChoice(m.at(-1)).field,'lora0');
  assert.match(describePipeline(s,m).stages[2].detail,/pixel.safetensors \(0.7\)/);
  assert.equal(describePipeline({...base,loras:[{model:'lora',weight:1}]},m).errors.length,1);
  assert.match(recordedPipeline({settings:s,loras:[{id:'xl',path:'lora/pixel.safetensors'}]}).stages[2].detail,/pixel/);
});

test('face and hand detector roles and recorded detail flow stay explicit',()=>{
 const detectors=[...inventory,{id:'face',name:'face.onnx',component:'detector',profile:{target:'face'}}];
 const settings={...base,face_detector:'face',detail_resolution:1024,detail_strength:.3,detail_max_regions:4};
 const flow=describePipeline(settings,detectors);assert.equal(flow.stages[4].skipped,false);assert.match(flow.stages[4].detail,/1024px/);assert.match(flow.stages[4].detail,/영역마다 6단계/);
 assert.equal(modelChoice(detectors.at(-1)).field,'face_detector');
 assert.equal(describePipeline({...base,hand_detector:'face'},detectors).errors.length,1);
 const history=recordedPipeline({settings,components:{face_detector:{path:'detectors/face.onnx',profile:{target:'face'}}}});
 assert.equal(history.stages[4].value,'face.onnx');
});

 test('SD1 shows one encoder and blocks SDXL components, including recorded history',()=>{
 const rows=[{...inventory[0],family:'sd1_candidate'},...inventory.slice(1)];
 assert.equal(describePipeline(base,rows).stages[1].value,'내장 CLIP 1개');
 assert.ok(describePipeline({...base,vae:'vae'},rows).errors.some(e=>e.includes('계열')));
 assert.equal(recordedPipeline({runner:'sd1_txt2img',settings:base,model_path:'sd1.safetensors'}).stages[1].value,'내장 CLIP 1개');
 });

test('ANIMA flow shows Qwen/T5 and rejects SD prediction',()=>{
 const rows=[{...inventory[0],family:'anima'}];
 const plan=describePipeline({...base,prediction_type:'flow'},rows);
 assert.deepEqual(plan.errors,[]);assert.equal(plan.stages[1].value,'Qwen3 + T5 tokenizer');
 assert.ok(describePipeline(base,rows).errors.some(e=>e.includes('flow')));
});

test('Turbo profile survives history and validates guidance',()=>{
 const plan={runner:'anima_txt2img',model_path:'anima-turbo.safetensors',settings:{...base,prediction_type:'flow'},anima:{variant:'turbo'}};
 assert.ok(recordedPipeline(plan).errors.some(e=>e.includes('Turbo')));
 plan.settings.guidance=1;assert.deepEqual(recordedPipeline(plan).errors,[]);
 assert.ok(recordedPipeline(plan).stages[1].detail.includes('turbo'));
});

test('SD2 uses OpenCLIP-H and preserves recorded family',()=>{
 const rows=[{...inventory[0],family:'sd2_candidate'},...inventory.slice(1)];
 assert.ok(describePipeline(base,rows).stages[1].detail.includes('OpenCLIP-H'));
 assert.ok(describePipeline({...base,vae:'vae'},rows).errors.some(e=>e.includes('계열')));
 assert.ok(recordedPipeline({runner:'sd2_txt2img',settings:base,model_path:'sd2.safetensors'}).stages[1].detail.includes('OpenCLIP-H'));
});

test('external CLIP slots show selected file and reject wrong family',()=>{
 const clip={id:'clip1',name:'encoder.safetensors',component:'text_encoder',profile:{family:'sdxl',slot:1}};
 const settings={...base,clip:'clip1'};
 assert.equal(describePipeline(settings,[...inventory,clip]).stages[1].value,'encoder.safetensors');
 assert.ok(describePipeline({...base,clip2:'clip1'},[...inventory,clip]).errors.some(e=>e.includes('CLIP')));
 assert.equal(recordedPipeline({runner:'sdxl_txt2img',settings,components:{clip:{path:'text_encoder/encoder.safetensors',profile:clip.profile}}}).stages[1].value,'encoder.safetensors');
});

test('inpaint requires a source and saved mask and rejects ANIMA',()=>{
 const settings={...base,flow:'inpaint',source_job:'source',mask_id:'mask',strength:.5};
 assert.deepEqual(describePipeline(settings,inventory).errors,[]);
 assert.ok(describePipeline({...settings,mask_id:null},inventory).errors.some(e=>e.includes('마스크')));
 assert.ok(describePipeline({...settings,prediction_type:'flow'},[{...inventory[0],family:'anima'}]).errors.some(e=>e.includes('마스크')));
});


test('dedicated nine channel model guards flow and survives recorded history',()=>{
 const rows=[{...inventory[0],family:'sd1_inpaint_candidate'}];
 assert.ok(describePipeline(base,rows).errors.some(e=>e.includes('9채널')));
 const settings={...base,flow:'inpaint',source_job:'source',mask_id:'mask',strength:.7};
 assert.deepEqual(describePipeline(settings,rows).errors,[]);
 const history=recordedPipeline({runner:'sd1_inpaint',inpaint_channels:9,settings,model_path:'inpaint.safetensors'});
 assert.ok(history.stages[2].detail.includes('9채널'));
 assert.equal(history.stages[1].value,'내장 CLIP 1개');
});


test('Krea2 uses profile components and exposes Turbo limits',()=>{
 const model={id:'krea',name:'Krea Turbo',path:'checkpoint/krea',family:'krea2',available:true,krea2:{text_encoder:'text_encoders/qwen.safetensors',vae:'vae/qwen.safetensors'}};
 const settings={...base,model:'krea',guidance:0,prediction_type:'flow',prompt_mode:'native'};
 const plan=describePipeline(settings,[model]);
 assert.deepEqual(plan.errors,[]);
 assert.match(modelChoice(model).detail,/Krea2/);
 const lora={id:'style',name:'Krea style',component:'lora',profile:{family:'krea2'}};
 const applied=describePipeline({...settings,loras:[{model:'style',weight:.6}]},[model,lora]);
 assert.deepEqual(applied.errors,[]);assert.match(applied.stages[2].detail,/Krea style/);
 const face={id:'face',name:'Face',component:'detector',profile:{target:'face'}};
 const detailed=describePipeline({...settings,face_detector:'face'},[model,face]);
 assert.deepEqual(detailed.errors,[]);assert.equal(detailed.stages[4].skipped,false);
 assert.match(plan.stages[0].value,/Negative 미사용/);
 assert.equal(plan.stages[1].value,'qwen.safetensors');
 assert.equal(plan.stages[3].value,'qwen.safetensors');
 assert.ok(describePipeline({...settings,guidance:5},[model]).errors.length);
 assert.ok(describePipeline({...settings,flow:'img2img'},[model]).errors.length);
 assert.deepEqual(describePipeline({...settings,flow:'img2img',source_job:'source'},[model]).errors,[]);
 const masked=describePipeline({...settings,flow:'inpaint',source_job:'source',mask_id:'mask'},[model]);
 assert.deepEqual(masked.errors,[]);assert.match(masked.stages[0].detail,/단계별 마스크 보존/);
 assert.ok(describePipeline({...settings,flow:'inpaint',source_job:'source'},[model]).errors.length);
 assert.match(recordedPipeline({runner:'krea2_txt2img',model_path:'checkpoint/krea',krea2:model.krea2,settings}).stages[1].detail,/Qwen3-VL/);
});


test('Klein flow distinguishes native reference editing and blocks incompatible stages',()=>{
 const models=[...inventory,{id:'klein',name:'Klein 4B',family:'flux2',available:true}];
 const settings={...base,model:'klein',prediction_type:'flow',guidance:1,steps:4,prompt_mode:'native'};
 const text=describePipeline(settings,models);assert.deepEqual(text.errors,[]);
 assert.match(text.stages.find(s=>s.key==='clip').value,/Qwen3 4B/);
 assert.match(text.stages[0].value,/Negative 미사용/);
 assert.ok(describePipeline({...settings,flow:'reference_edit'},models).errors.length);
 const ref={...settings,flow:'reference_edit',reference_attachment:'abcd1234'};
 const edit=describePipeline(ref,models);assert.deepEqual(edit.errors,[]);
 assert.match(edit.stages.find(s=>s.key==='reference').value,/사용자 참고/);
 assert.match(edit.stages.find(s=>s.key==='reference').detail,/강도 기반 img2img와 다름/);
 assert.match(recordedPipeline({runner:'flux2_reference_edit',model_path:'checkpoint/klein',settings:ref}).title,/참고 이미지/);
 assert.ok(describePipeline({...settings,flow:'img2img',source_job:'old'},models).errors.length);
 assert.ok(describePipeline({...settings,face_detector:'face'},models).errors.length);
 assert.match(modelChoice(models.at(-1)).detail,/Klein/);
 assert.match(describePipeline({...settings,upscaler:'up'},models).stages.find(s=>s.key==='upscale').detail,/RAM 여유 시 모델 보관/);
});


test('inventory filters normalize words and retain incompatible selections',()=>{
 const z={id:'z',name:'꽃밭 Z Turbo',path:'checkpoint/z-bf16',family:'zimage',available:true};
 const sd={id:'sd',name:'WAI',path:'checkpoint/wai',family:'sdxl_candidate',available:true};
 const lora={id:'l',name:'Dress',path:'lora/dress',component:'lora',profile:{family:'sdxl'}};
 const broken={id:'broken',name:'Lost profile',path:'lora/broken',available:false};
 const up={id:'up',name:'BSRGAN',path:'upscale_models/up',component:'upscaler',profile:{scale:2}};
 const models=[z,sd,lora,broken,up],snapshot=JSON.stringify(models);
 const settings={model:'z',loras:[{model:'l'},{model:'broken'}],upscaler:'up'};
 const ids=options=>modelInventoryRows(models,{settings,...options}).map(row=>row.model.id);
 assert.deepEqual(ids({query:'꽃밭'.normalize('NFD')+' BF16'}),['z']);
 assert.deepEqual(ids({query:'ｗａｉ'}),['sd']);
 assert.deepEqual(ids({type:'lora',scope:'compatible'}),[]);
 assert.deepEqual(ids({scope:'compatible'}),['z','sd','up']);
 assert.deepEqual(ids({scope:'selected'}),['z','l','broken','up']);
 assert.deepEqual(ids({scope:'selected',query:'dress'}),['l']);
 assert.deepEqual(ids({query:'nothing'}),[]);
 assert.deepEqual(ids({scope:'compatible',type:'lora',settings:{model:'sd'}}),['l']);
 assert.equal(JSON.stringify(models),snapshot);
 const incompatible=modelInventoryRows(models,{settings}).find(row=>row.model.id==='l');
 assert.equal(incompatible.selected,true);assert.match(incompatible.incompatibility,/SDXL 전용/);
 assert.match(modelChoice(z).detail,/Z-Image LoRA · 디테일러 미지원/);
});


test('Klein accepts only matching profiled LoRA and shows it in the flow',()=>{
 const model={id:'klein',name:'Klein',family:'flux2',available:true,path:'checkpoint/klein'};
 const lora={id:'style',name:'Style',component:'lora',profile:{family:'flux2'},path:'lora/style.safetensors'};
 const settings={...base,model:'klein',guidance:1,prediction_type:'flow',prompt_mode:'native',loras:[{model:'style',weight:.7}]};
 assert.deepEqual(describePipeline(settings,[model,lora]).errors,[]);
 assert.ok(describePipeline(settings,[model,{...lora,profile:{family:'sdxl'}}]).errors.length);
 assert.match(JSON.stringify(describePipeline(settings,[model,lora]).stages),/Style/);
});

test('Z-Image accepts only matching profiled LoRA and shows it in the flow',()=>{
 const model={id:'klein',name:'Klein',family:'zimage',available:true,path:'checkpoint/klein'};
 const lora={id:'style',name:'Style',component:'lora',profile:{family:'zimage'},path:'lora/style.safetensors'};
 const settings={...base,model:'klein',guidance:0,prediction_type:'flow',prompt_mode:'native',loras:[{model:'style',weight:.7}]};
 assert.deepEqual(describePipeline(settings,[model,lora]).errors,[]);
 assert.ok(describePipeline(settings,[model,{...lora,profile:{family:'sdxl'}}]).errors.length);
 assert.match(JSON.stringify(describePipeline(settings,[model,lora]).stages),/Style/);
});
