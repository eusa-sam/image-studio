import test from 'node:test';
import assert from 'node:assert/strict';
import {jobProgress,jobStatusLabel,renderJobProgress,jobTiming,resultPreview} from '../backend/app/web/pipeline.mjs';

test('first generation callback wait does not invent a step count',()=>{
  const first=jobProgress({status:'RUNNING',progress:{stage:'generate'}});
  assert.match(first.label,/첫 단계 완료 대기/);
  assert.equal(first.count,null);
  const running=jobProgress({status:'RUNNING',progress:{stage:'generate',step:1,total:4}});
  assert.doesNotMatch(running.label,/완료 대기/);
  assert.deepEqual(running.count,{value:1,max:4});
  assert.doesNotMatch(jobProgress({status:'QUEUED',progress:{stage:'generate'}}).label,/첫 단계/);
});

test('progress counts describe only the current phase and never fake loading percentages',()=>{
  const state=p=>jobProgress({status:'RUNNING',progress:p});
  assert.equal(state({stage:'load'}).count,null);
  assert.equal(state({stage:'decode',step:8,total:8}).count,null);
  assert.equal(state({stage:'generate',step:9,total:8}).count,null);
  assert.deepEqual(state({stage:'generate',step:3,total:8}).count,{value:3,max:8});
  assert.match(state({stage:'detail',region:2,regions:4,target:'hand',step:3,total:6}).label,/손.*2\/4.*3\/6/);
  assert.equal(state({stage:'detail',step:6,total:6,decoding:true}).count,null);
  assert.match(state({stage:'upscale',step:2,total:4}).label,/2\/4 타일/);
  assert.equal(jobProgress({status:'QUEUED',progress:{stage:'generate',step:8,total:8}}).count,null);
  for(const status of ['SUCCEEDED','FAILED','CANCELLED','INTERRUPTED','PARTIAL_FAILED'])assert.equal(jobProgress({status,progress:{stage:'generate'}}),null);
  assert.equal(jobStatusLabel('SUCCEEDED'),'완료');
});

test('memory labels distinguish the initial measurement and budgets from live usage',()=>{
  const state=jobProgress({status:'RUNNING',progress:{stage:'generate',memory_policy:{mode:'partial',free_vram_bytes:12*1024**3,resident_budget_bytes:8*1024**3,cpu_cache_budget_bytes:16*1024**3}}});
  assert.match(state.memory,/시작 시 가용 VRAM 12.0 GiB/);
  assert.match(state.memory,/가중치 VRAM 예산 8.0 GiB/);
  assert.match(state.memory,/RAM 캐시 상한 16.0 GiB/);
  const blocks=jobProgress({status:'RUNNING',progress:{stage:'generate',memory_policy:{mode:'block_offload'}}});
  assert.equal(blocks.memory,'일부 블록 GPU 상주');
});

test('native progress is indeterminate for loading and removed after completion',()=>{
  const doc={createElement(tag){return {tag,setAttribute(key,value){this[key]=value;}};}};
  const node={ownerDocument:doc,dataset:{},replaceChildren(...children){this.children=children;}};
  renderJobProgress(node,{status:'RUNNING',progress:{stage:'generate',step:2,total:8}});
  assert.equal(node.children[1].value,2);assert.equal(node.children[1].max,8);
  assert.ok(node.children[1]['aria-label']);
  const bar=node.children[1];
  renderJobProgress(node,{status:'RUNNING',progress:{stage:'generate',step:2,total:8}});
  assert.equal(node.children[1],bar);
  renderJobProgress(node,{status:'RUNNING',progress:{stage:'load'}});
  assert.equal('value' in node.children[1],false);
  renderJobProgress(node,{status:'SUCCEEDED'});
  assert.equal(node.hidden,true);assert.deepEqual(node.children,[]);
});

test('output copy time is included while legacy completion timestamps remain supported',()=>{
 const job={status:'SUCCEEDED',created_at:'2026-11-06T00:00:00Z',started_at:'2026-11-06T00:00:05Z',finished_at:'2026-11-06T00:00:20Z'};
 assert.equal(jobTiming({...job,output_status:'PENDING'},Date.parse('2026-11-06T00:00:25Z')),'출력 폴더 저장 중 · 총 25초');
 assert.equal(jobTiming({...job,output_status:'SUCCEEDED',output_finished_at:'2026-11-06T00:00:30Z'}),'실행 25초 · 대기 5초 · 총 30초');
 assert.equal(jobTiming(job),'실행 15초 · 대기 5초 · 총 20초');
});

test('elapsed labels separate queue and execution, freeze terminal time, and label legacy totals',()=>{
 const job={created_at:'2026-11-06T00:00:00Z',updated_at:'2026-11-06T00:00:10Z'};
 const now=Date.parse('2026-11-06T00:01:15Z');
 assert.equal(jobTiming({...job,status:'QUEUED'},now),'대기 중 · 1분 15초');
 assert.equal(jobTiming({...job,status:'RUNNING'},now),'실행 중 · 1분 5초 · 대기 10초');
 const done={...job,status:'SUCCEEDED',started_at:job.updated_at,finished_at:'2026-11-06T00:01:00Z',updated_at:'2026-11-06T01:00:00Z'};
 assert.equal(jobTiming(done,now),'실행 50초 · 대기 10초 · 총 1분 0초');
 assert.equal(jobTiming(done,now+999999),jobTiming(done,now));
 assert.equal(jobTiming({...job,status:'FAILED'},now),'총 10초 · 대기 포함');
 assert.equal(jobTiming({status:'SUCCEEDED'}),'소요 시간 기록 없음');
 assert.equal(jobTiming({...job,status:'QUEUED'},0),'대기 중 · 0초');
});
test('export retry excludes the days before a manual retry from generation time',()=>{
 const job={status:'SUCCEEDED',created_at:'2026-11-01T00:00:00Z',finished_at:'2026-11-01T00:00:20Z',output_retry_started_at:'2026-11-07T00:00:00Z',output_finished_at:'2026-11-07T00:00:03Z'};
 assert.equal(jobTiming({...job,output_status:'PENDING'},Date.parse('2026-11-07T00:00:02Z')),'출력 다시 저장 중 · 2초');
 assert.equal(jobTiming({...job,output_status:'SUCCEEDED'}),'생성 총 20초 · 출력 재저장 3초');
 assert.equal(jobTiming({...job,output_status:'FAILED'}),'생성 총 20초 · 출력 재저장 미완료 3초');
 assert.match(jobTiming({...job,output_status:'INTERRUPTED',output_finished_at:'2026-11-01T00:00:22Z'}),/시간 기록 없음/);
});
test('result image opens a new tab and original toggle changes preview and full-size link together',()=>{
 const doc={createElement(tag){return {tag,children:[],append(...nodes){this.children.push(...nodes);},setAttribute(k,v){this[k]=v;}};}};
 const preview=resultPreview(doc,{id:'job',status:'SUCCEEDED',has_original:true},'2026-11/workspace');
 const [controls,link]=preview.children;
 assert.equal(link.target,'_blank');assert.equal(link.rel,'noopener noreferrer');
 assert.match(link.href,/image.png\?session=2026-11%2Fworkspace/);
 controls.children[0].onclick();assert.match(link.href,/original.png/);
 assert.equal(link.children[0].src,link.href);assert.equal(controls.children[0]['aria-pressed'],'true');
 controls.children[1].onclick();assert.match(link.href,/image.png/);assert.equal(controls.children[0]['aria-pressed'],'false');
 assert.equal(resultPreview(doc,{id:'job',has_original:false},'workspace').children.length,1);
});

test('failed or interrupted postprocessing opens the preserved original without offering a missing final image',()=>{
 const doc={createElement(tag){return {tag,children:[],append(...nodes){this.children.push(...nodes);},setAttribute(k,v){this[k]=v;}};}};
 for(const status of ['PARTIAL_FAILED','CANCELLED','INTERRUPTED']) {
  const preview=resultPreview(doc,{id:'job',status,has_original:true},'2026-11/workspace');
  assert.equal(preview.children.length,1);
  const link=preview.children[0];assert.equal(link.tag,'a');assert.equal(link.target,'_blank');
  assert.match(link.href,/original.png\?session=2026-11%2Fworkspace/);
  assert.equal(link.children[0].src,link.href);assert.match(link.children[0].alt,/보존된 생성 원본/);
 }
});


test('result preview restores the selected original after refresh and reports only user changes',()=>{
 const doc={createElement(tag){return {tag,children:[],append(...nodes){this.children.push(...nodes);},setAttribute(k,v){this[k]=v;}};}};
 const changes=[];
 const preview=resultPreview(doc,{id:'job',status:'SUCCEEDED',has_original:true},'workspace',{selected:'original.png',onSelect:name=>changes.push(name)});
 const [controls,link]=preview.children;
 assert.match(link.href,/original.png/);assert.equal(controls.children[0]['aria-pressed'],'true');assert.deepEqual(changes,[]);
 controls.children[1].onclick();assert.deepEqual(changes,['image.png']);assert.match(link.href,/image.png/);
 const missing=resultPreview(doc,{id:'job',status:'SUCCEEDED',has_original:false},'workspace',{selected:'original.png'});
 assert.match(missing.children[0].href,/image.png/);
});
