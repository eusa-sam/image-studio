import {setupAppSettings} from './settings.mjs?v=20261107-ui32';
import {setupEmbeddings} from './embeddings.mjs?v=20261107-ui32';
import {setupGenerationReference,referenceSettings} from './reference.mjs?v=20261107-ui32';
import { setupImageRuntime } from './runtime.mjs?v=20261107-ui32';
import {setupMask} from './mask.mjs?v=20261107-ui32';
import { modelTypes, modelType, setupNavigation, setupStudioTabs, renderWorkspaces } from './navigation.mjs?v=20261107-ui32';
import { setupChat, sameDraft, proposalSettings } from './chat.mjs?v=20261107-ui32';
import { modelInventoryRows, imageFamily, modelSelectionSettings, describePipeline, renderPipeline, recordedPipeline, modelChoice, componentCompatibility, renderComponentSelect, jobStatusLabel, renderJobProgress, jobTiming, resultPreview } from './pipeline.mjs?v=20261107-ui32';
const $ = id => document.getElementById(id);
const refreshImageRuntime=setupImageRuntime({status:$('image-runtime-status'),button:$('release-image-model'),api,onError:message=>notice(message)});
let modelInventory = [];
let modelScanVersion = 0;
let componentModel=null;
let displayedJobs=[];
function refreshJobTimes(){for(const job of displayedJobs){const node=$('job-time-'+job.id);if(node)node.textContent=jobTiming(job);}}
setInterval(refreshJobTimes,1000);
let workspaceRecords = [];
for (let i=0;i<4;i++) {
  const label=document.createElement('label');label.textContent=`LoRA ${i+1}`;
  const select=document.createElement('select');select.id=`lora${i}`;select.append(option('','사용 안 함'));
  const weight=document.createElement('input');weight.id=`lora-weight${i}`;weight.type='number';weight.min='-2';weight.max='2';weight.step='any';weight.value='1';weight.setAttribute('aria-label',`LoRA ${i+1} 강도`);
  label.append(select,weight);$('lora-slots').append(label);
}

let preferences;
try { preferences = window.localStorage; } catch { /* Navigation works without storage. */ }
setupNavigation(document, preferences);
const showStudio=setupStudioTabs(document);
let session, promptId, revision = 0, savedPrompts = [], busy = false, jobSignature = '';
const resultSelections=new Map();
let jobRefreshVersion=0;
let lastSubmission = null;
let workspaceEpoch = 0;
let workspaceReady = false;
let originalInputSupported=false;
const generationReference=setupGenerationReference({doc:document,api,getSession:()=>session,isBusy:()=>busy,supportsOriginal:()=>originalInputSupported,changed:renderComposition});
const embeddings=setupEmbeddings({doc:document,inventory:()=>modelInventory,checkpoint:()=>modelInventory.find(m=>m.id===$('model').value),isBusy:()=>busy,changed:()=>{renderComposition();renderModels();checkTokens();}});
function context() {
  return {session, prompt_id:promptId, expected_revision:revision, positive:$('positive').value,
    negative:$('negative').value, settings:settings()};
}
const chat = setupChat({api, getContext:context, isWorkspaceReady:()=>workspaceReady, apply:async (turn,generate) => {
  if (!workspaceReady || busy || generationReference.pending() || !sameDraft(turn.context,context())) throw new Error('작업실이 변경되었거나 저장 중입니다. 제안을 다시 확인하세요.');
  busy=true;
  const controls=[...document.querySelectorAll('#editor input, #editor textarea, #editor select, #editor button'), $('sessions'), $('workspace-picker'), $('new-session'), $('rescan')];
  controls.forEach(c => c.disabled=true);
  try {
    const effective=await proposalSettings(turn,modelInventory.find(m=>m.id===turn.context.settings.model),api);
    const receipt=await api('/api/prompts',{...turn.context,settings:effective, positive:turn.positive, negative:turn.negative, request_id:turn.id, generate});
    revision=receipt.revision; $('positive').value=turn.positive; $('negative').value=turn.negative;
    $('prompt_mode').value=effective.prompt_mode;
    if(effective.prompt_mode!==turn.context.settings.prompt_mode)notice('긴 프롬프트를 빠뜨리지 않도록 가중치 인코딩으로 전환했습니다.');
    $('revision').textContent=`revision ${revision}`;
    const warning=await refreshAfterSave(generate);
    showStudio(generate?'results':'create');
    return warning;
  } finally { busy=false; controls.forEach(c => c.disabled=false); updateFlow(); }
}});
async function refreshAfterSave(generate) {
  const results=await Promise.allSettled([refreshPrompts(promptId),refreshJobs()]);
  const errors=results.filter(result=>result.status==='rejected').map(result=>result.reason.message);
  return errors.length ? (generate?'프롬프트 저장과 생성 요청을 접수했습니다.':'프롬프트를 저장했습니다.')+' 목록 갱신 실패: '+errors.join(' · ')+' · 새로고침해 저장된 내용을 확인하세요.' : '';
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) {const error=new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));error.status=response.status;throw error;}
  return data;
}
function notice(message) { $('notice').textContent = message; }
async function refreshStorageErrors() {
  const errors=await api('/api/storage-errors');
  $('storage-warning').hidden=!errors.length;
  $('storage-warning-title').textContent=`읽지 못한 저장 기록 ${errors.length}개 · 정상 기록은 계속 사용할 수 있습니다`;
  $('storage-errors').replaceChildren(...errors.map(error=>{const row=document.createElement('li');row.textContent=error.path+': '+error.error;return row;}));
}
function option(value, text) { const item = document.createElement('option'); item.value = value; item.textContent = text; return item; }
function settings() {
  const result = {...referenceSettings($('generation-reference').value),ip_adapter:$('ip_adapter').value||null,clip_vision:$('clip_vision').value||null,ip_adapter_scale:Number($('ip_adapter_scale').value),random_seed:$('random_seed').checked,prompt_mode:$('prompt_mode').value,model:$('model').value, prediction_type:$('prediction_type').value,
    clip:$('clip').value || null,clip2:$('clip2').value || null,vae:$('vae').value || null, upscaler:$('upscaler').value || null,
    face_detector:$('face_detector').value || null, hand_detector:$('hand_detector').value || null, flow:$('flow').value,
    mask_id:$('flow').value==='inpaint'?($('mask_id').value||null):null,
    source_job:['img2img','inpaint'].includes($('flow').value) ? ($('source_job').value || null) : null, strength:Number($('strength').value),
    loras:[0,1,2,3].filter(i => $(`lora${i}`).value).map(i => ({model:$(`lora${i}`).value,weight:Number($(`lora-weight${i}`).value)}))};
  for (const key of ['width','height','steps','guidance','seed','detail_strength','detail_steps','detail_confidence','detail_max_regions','detail_resolution']) result[key] = Number($(key).value);
  if(embeddings.get().length)result.embeddings=embeddings.get();
  if(result.source_job&&$('source_artifact').value==='original.png')result.source_artifact='original.png';
  return result;
}
let tokenTimer, tokenVersion=0;
function checkTokens() {
  clearTimeout(tokenTimer);
  const version=++tokenVersion;
  const body={mode:$('prompt_mode').value,positive:$('positive').value, negative:$('negative').value, model:$('model').value, family:imageFamily(modelInventory.find(m=>m.id===$('model').value))};
  if(embeddings.get().length)body.embeddings=embeddings.get();
  $('token-count').textContent='토큰 수 확인 중…';
  tokenTimer=setTimeout(async () => {
    try {
      const result=await api('/api/prompt-tokens',body);
      if (version !== tokenVersion) return;
      $('token-count').textContent=result.encoders.map(r => `${r.encoder}: Positive ${r.positive}/${r.limit} · Negative ${r.negative}/${r.limit}`).join(' | ') + (result.truncated ? ' — 초과한 뒷부분은 생성 시 잘립니다. 프롬프트를 줄이세요.' : body.mode==='weighted' ? ' · 최대 '+Math.max(...result.encoders.map(r=>r.chunks))+'개 CLIP 블록 · 잘림 없이 가중치 인코딩' : ['anima','flux2','zimage'].includes(body.family) ? ' · 인코더 템플릿·특수 토큰 포함' : ' · 시작/종료 토큰 제외') + (result.negative_used===false ? ' · Turbo: Negative 미사용' : '');
      $('token-count').className=result.truncated ? 'hint token-warning' : 'hint';
    } catch(e) { if (version === tokenVersion) $('token-count').textContent='토큰 수 확인 불가: '+e.message; }
  },400);
}
function selectLora(select,id) {
  if(id && ![...select.options].some(o => o.value===id)) select.append(option(id,'누락 또는 미지원 LoRA · '+id));
  select.value=id || '';
}
function newPrompt() { generationReference.restore({}); promptId = crypto.randomUUID(); revision = 0; $('positive').value = ''; $('negative').value = ''; $('random_seed').checked=true; $('seed').disabled=true; $('revision').textContent = '미저장'; $('prompts').value = ''; renderComposition(); checkTokens(); }
async function loadPrompt(id) {
  if(busy)return;
  if(!id)return newPrompt();
  const reference=session,epoch=workspaceEpoch,previous=promptId;
  busy=true;
  const controls=[...document.querySelectorAll('#editor input, #editor textarea, #editor select, #editor button'),$('sessions'),$('workspace-picker'),$('new-session')];
  const disabled=controls.map(control=>control.disabled);controls.forEach(control=>control.disabled=true);
  try {
    const cached=savedPrompts.find(p=>p.id===id);
    const prompt=cached?.settings?cached:await api('/api/prompts/'+encodeURIComponent(id)+'?session='+encodeURIComponent(reference));
    if(reference!==session || epoch!==workspaceEpoch)return;
    applyPrompt(prompt);
  } catch(error) {
    if(reference===session && epoch===workspaceEpoch)$('prompts').value=previous || '';
    throw error;
  } finally {busy=false;controls.forEach((control,index)=>control.disabled=disabled[index]);updateFlow();}
}
function applyPrompt(prompt) {
  $('mask_id').value='';$('prompt_mode').value='native';
  promptId = prompt.id; revision = prompt.revision;
  $('positive').value = prompt.positive; $('negative').value = prompt.negative;
  $('ip_adapter').value='';$('clip_vision').value='';$('ip_adapter_scale').value='0.7';
  generationReference.restore(prompt.settings);
  $('clip').value='';$('clip2').value='';$('vae').value=''; $('upscaler').value=''; $('flow').value='txt2img'; $('source_job').value=''; $('source_artifact').value='image.png'; $('strength').value='0.5';
  for(const [key,value] of Object.entries({face_detector:'',hand_detector:'',detail_strength:0.3,detail_steps:20,detail_confidence:0.35,detail_max_regions:4,detail_resolution:1024}))$(key).value=value;
  for (const [key, value] of Object.entries(prompt.settings)) if ($(key)) {
    if(['model','clip','clip2','ip_adapter','clip_vision','source_job'].includes(key) && value && ![...$(key).options].some(item=>item.value===value))$(key).append(option(value,'누락 또는 미지원 · '+value));
    $(key).value = value;
  }
  $('random_seed').checked=prompt.settings.random_seed ?? true;
  embeddings.restore(prompt.settings.embeddings);
  for(let i=0;i<4;i++) {const lora=prompt.settings.loras?.[i];selectLora($(`lora${i}`),lora?.model);$(`lora-weight${i}`).value=lora?.weight ?? 1;}
  $('revision').textContent = `revision ${revision}`;
  refreshComponentChoices();
  updateFlow();
}
function updateFlow() {
  embeddings.refresh();
  const family=imageFamily(modelInventory.find(m=>m.id===$('model').value)),klein=family==='flux2';
  for(const option of $('flow').options)option.disabled=family==='zimage'?option.value!=='txt2img':klein? !['txt2img','reference_edit'].includes(option.value):option.value==='reference_edit';
  if($('flow').value==='reference_edit')$('generation-reference-panel').open=true;
  $('seed').disabled=busy || $('random_seed').checked;
  checkTokens();
  const active=['img2img','inpaint'].includes($('flow').value);
  $('mask-editor').hidden=$('flow').value!=='inpaint';
  $('source_job').disabled=busy || !active; $('source_job').required=active; $('source_artifact').disabled=busy || !active || !originalInputSupported;
  $('strength').disabled=busy || !active; $('source-hint').hidden=!active;
  renderComposition(); renderModels();
}
function focusSetting(id) {
  showStudio('create');
  const target=$(id);
  for(let parent=target.parentElement;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;
  target.scrollIntoView({behavior:'smooth',block:'center'});
  if (!target.disabled) target.focus({preventScroll:true});
}
function refreshComponentChoices() {
  const checkpoint=modelInventory.find(m=>m.id===$('model').value);
  for(const [field,label] of [['ip_adapter','사용 안 함'],['clip_vision','인코더 선택'],['clip','기본 인코더 (내장/프로필)'],['clip2','기본 인코더'],['vae','기본 VAE (내장/프로필)'],['upscaler','사용 안 함'],['face_detector','사용 안 함'],['hand_detector','사용 안 함'],...[0,1,2,3].map(i=>['lora'+i,'사용 안 함'])]) {
    renderComponentSelect($(field),field,checkpoint,modelInventory,label);
  }
  componentModel=$('model').value;
  embeddings.refresh();
}
function renderComposition() {
  if(componentModel!==$('model').value)refreshComponentChoices();
  const selected=settings(), plan=describePipeline(selected,modelInventory);
  const sourceUrl=selected.flow === 'img2img' && selected.source_job && session ?
    `/api/jobs/${encodeURIComponent(selected.source_job)}/${selected.source_artifact||'image.png'}?session=${encodeURIComponent(session)}` : null;
  renderPipeline($('pipeline-map'),plan,{onSelect:focusSetting,sourceUrl});
  $('pipeline-summary').textContent=`${plan.title} · 예상 출력 ${plan.output} · Seed ${selected.random_seed?'매번 랜덤':selected.seed}`;
  $('pipeline-state').textContent=plan.errors.length ? '구성 확인 필요' : '구성 선택됨';
  $('pipeline-state').className='status-chip ' + (plan.errors.length ? 'status-warning' : 'status-selected');
  $('pipeline-errors').hidden=!plan.errors.length;
  $('pipeline-errors').replaceChildren(...plan.errors.map(error => {const li=document.createElement('li');li.textContent=error;return li;}));
  $('component-summary').replaceChildren(...plan.components.map(component => {
    const card=document.createElement('button');card.type='button';card.className='component-card' + (component.unsupported ? ' component-unavailable' : '');
    const label=document.createElement('span');label.className='component-label';label.textContent=component.label;
    const value=document.createElement('strong');value.textContent=component.value;
    const badge=document.createElement('span');badge.className='status-chip';badge.textContent=component.state;
    card.append(label,value,badge);
    card.onclick=() => {
      if (component.target) focusSetting(component.target);
      else { $('nav-models').click(); $('model-type').value=component.category; renderModels(); $('models-panel').scrollIntoView({behavior:'smooth',block:'nearest'}); }
    };
    return card;
  }));
  const llm=$('llm-model').value;
  const action=$('chat-action').selectedOptions[0]?.textContent || '제안만 보기';
  $('llm-flow-hint').textContent=llm ? `대화 도우미: ${llm} → ${action}. 위 흐름은 이미지 생성 버튼 기준입니다.` : 'LLM은 선택 사항입니다. 직접 프롬프트를 작성해도 위 흐름으로 생성됩니다.';
}
async function refreshPrompts(selectId) {
  const reference=session, epoch=workspaceEpoch;
  let prompts;
  try {prompts=await api('/api/prompts/summaries?session='+encodeURIComponent(reference));}
  catch(error){if(error.status!==404)throw error;prompts=await api('/api/prompts?session='+encodeURIComponent(reference));}
  if (reference !== session || epoch !== workspaceEpoch) return false;
  savedPrompts=prompts;
  $('prompts').replaceChildren(option('', '새 프롬프트'), ...savedPrompts.map(p => option(p.id, (p.title ?? p.positive?.slice(0,60)) || '(빈 프롬프트)')));
  $('prompts').value = selectId || savedPrompts.at(-1)?.id || '';
  return true;
}
async function changeSession() {
  if(busy){$('sessions').value=session;return;}
  workspaceReady=false;$('editor').inert=true;
  const epoch=++workspaceEpoch;
  $('workspace-loading').hidden=false;$('workspace-retry').disabled=true;
  $('workspace-loading-status').textContent='Workspace를 불러오는 중… 완료되면 작업실과 대화를 사용할 수 있어요.';
  try {
    displayedJobs=[];
    session = $('sessions').value; jobSignature = ''; resultSelections.clear(); $('jobs').replaceChildren();
    chat.switchWorkspace();
    generationReference.reset();
    $('chat-image').value='latest';
    $('workspace-name').textContent = $('sessions').selectedOptions[0]?.textContent || 'Workspace';
    const record=workspaceRecords.find(row=>row.ref===session);
    $('workspace-title').value=record?.title || '';
    $('archive-workspace').textContent=record?.archived?'보관 해제':'보관';
    if (!await refreshPrompts()) return;
    await refreshJobs();
    if (epoch !== workspaceEpoch) return;
    await loadPrompt($('prompts').value);
    if(epoch!==workspaceEpoch)return;
    workspaceReady=true;$('editor').inert=false;
    await chat.refresh();
    await refreshStorageErrors();
    if(epoch===workspaceEpoch)$('workspace-loading').hidden=true;
  } catch(error) {
    if(epoch!==workspaceEpoch)return;
    $('workspace-loading-status').textContent='Workspace를 불러오지 못했습니다: '+error.message;
    $('workspace-retry').disabled=false;
    throw error;
  }
}
async function scanModels() {
  const version=++modelScanVersion;
  let models;
  try {models=await api('/api/models');}
  catch(error){if(version!==modelScanVersion)return;throw error;}
  if(version!==modelScanVersion)return;
  const selected = $('model').value;
  $('model').replaceChildren(...models.filter(m => m.available).map(m => option(m.id,m.name)));
  if(selected) {
    if(!models.some(m=>m.available && m.id===selected)) {
      const previous=models.find(m=>m.id===selected),missing=option(selected,(previous?.name||selected)+' · '+(previous?.reason||'누락 또는 미지원 Checkpoint'));
      missing.disabled=true;$('model').append(missing);
    }
    $('model').value=selected;
  }
  modelInventory = models;
  refreshComponentChoices();
  const type = $('model-type').value;
  $('model-type').replaceChildren(option('all', `전체 (${models.length})`), ...modelTypes.map(([id, label]) =>
    option(id, `${label} (${models.filter(m => modelType(m.path) === id).length})`)));
  $('model-type').value = type;
  renderModels();
  renderComposition();
}
let modelScope = 'all';
function renderModels() {
  const type = $('model-type').value;
  const models = modelInventoryRows(modelInventory,{type,query:$('model-search').value,scope:modelScope,settings:settings()});
  for(const scope of ['all','compatible','selected'])$('model-scope-'+scope).setAttribute('aria-pressed',String(scope===modelScope));
  $('model-count').textContent = `${models.length}개 표시 / 전체 ${modelInventory.length}개 · 현재 Checkpoint 기준`;
  $('inventory').replaceChildren();
  if (!models.length) { const empty = document.createElement('p'); empty.textContent = '조건에 맞는 모델이 없습니다. 검색어를 바꾸거나 필터를 초기화하세요.'; $('inventory').append(empty); return; }
  for (const [id, label] of modelTypes) {
    const group = models.filter(row => modelType(row.model.path) === id);
    if (!group.length) continue;
    const heading = document.createElement('h3'); heading.textContent = `${label} · ${group.length}`; $('inventory').append(heading);
    $('inventory').append(...group.map(({model:m,choice,selected,incompatibility}) => {
    const div = document.createElement('div'); div.className = 'model' + (selected ? ' model-selected' : '');
    const badge=document.createElement('span');badge.className='status-chip ' + (incompatibility ? 'status-warning' : selected ? 'status-selected' : choice.field ? 'status-available' : 'status-muted');
    badge.textContent=incompatibility ? (selected?'현재 선택 · 호환 안 됨':'현재 Checkpoint와 호환 안 됨') : selected ? '현재 선택됨' : choice.label;
    const title = document.createElement('strong'); title.textContent = m.name;
    const info = document.createElement('p'); info.textContent = incompatibility ? incompatibility+' · '+choice.detail : choice.detail;
    const path = document.createElement('p'); path.className = 'model-path'; path.textContent = m.path;
    div.append(badge, title, path, info);
    if (choice.field) {
      const select=document.createElement('button');select.type='button';select.textContent=selected ? '선택됨' : '이 모델 선택';select.disabled=selected || busy || !!incompatibility;
      select.onclick=() => {if (busy || incompatibility) return; if(m.component==='embedding'){embeddings.add(m.id);focusSetting('embedding');return;} if (m.component==='lora') {const slot=[0,1,2,3].find(i => !$(`lora${i}`).value);if(slot===undefined){notice('LoRA는 최대 4개입니다. 기존 선택을 해제하세요.');return;}$(`lora${slot}`).value=m.id;} else {$(choice.field).value=m.id;$(choice.field).dispatchEvent(new Event('change',{bubbles:true}));}updateFlow();};div.append(select);
    } else {
      const reason=document.createElement('details');const summary=document.createElement('summary');summary.textContent='검사 정보';
      const text=document.createElement('p');text.textContent=`${m.family} · ${m.reason}`;reason.append(summary,text);div.append(reason);
    }
    return div;
    }));
  }
}
async function refreshJobs() {
  if (!session) return;
  const selectedSession = session,version=++jobRefreshVersion;
  let jobs;
  try {jobs=await api('/api/jobs?session=' + encodeURIComponent(selectedSession));}
  catch(error){if(session!==selectedSession||version!==jobRefreshVersion)return;throw error;}
  if (session !== selectedSession || version!==jobRefreshVersion) return;
  displayedJobs=jobs;refreshJobTimes();
  const signature = JSON.stringify(jobs.map(({progress,...job})=>job));
  if (signature === jobSignature) {
    for(const job of jobs){const node=$('job-progress-'+job.id);if(node)renderJobProgress(node,job);}
    return;
  }
  jobSignature = signature;
  generationReference.jobs(jobs);
  $('result-count').textContent=String(jobs.length);
  const source=$('source_job').value;
  const chatImage=$('chat-image').value;
  $('chat-image').replaceChildren(option('latest','이 Workspace의 최근 생성 이미지'),option('none','이미지 없이 대화'),...jobs.filter(j=>j.status==='SUCCEEDED').map(j=>option(j.id,j.id.slice(0,8)+' · '+new Date(j.created_at).toLocaleString())));
  $('chat-image').value=['latest','none'].includes(chatImage)||jobs.some(j=>j.id===chatImage&&j.status==='SUCCEEDED')?chatImage:'latest';
  $('source_job').replaceChildren(option('','완료된 생성 작업 선택'), ...jobs.filter(j => j.status === 'SUCCEEDED' || j.has_original).map(j =>
    option(j.id,`${j.id.slice(0,8)}${j.status!=='SUCCEEDED'?' · 원본만 보존':''} · ${new Date(j.created_at).toLocaleString()}`)));
  if (jobs.some(j => j.id === source && (j.status === 'SUCCEEDED' || j.has_original))) $('source_job').value=source;
  $('jobs').replaceChildren(...jobs.sort((a,b) => b.created_at.localeCompare(a.created_at)).map(job => {
    const article = document.createElement('article');
    const title = document.createElement('strong'); title.textContent = `${job.output_status==='PENDING'?'이미지 완료 · 출력 저장 중':jobStatusLabel(job.status)} · ${job.id.slice(0,8)}`; article.append(title);
    const info = document.createElement('p'); info.textContent = new Date(job.created_at).toLocaleString()+(job.plan?.settings ? ' · Seed '+job.plan.settings.seed : ''); article.append(info);
    const progress=document.createElement('div');progress.id='job-progress-'+job.id;progress.className='job-progress';progress.setAttribute('aria-live','polite');renderJobProgress(progress,job);article.append(progress);
    const timing=document.createElement('p');timing.id='job-time-'+job.id;timing.className='job-time';timing.textContent=jobTiming(job);article.append(timing);
    if (job.plan?.settings) {
      const details=document.createElement('details');details.className='job-pipeline';
      const summary=document.createElement('summary');summary.textContent='요청 당시 구성 / 흐름 보기';
      const flow=document.createElement('div');renderPipeline(flow,recordedPipeline(job.plan));
      details.append(summary,flow);article.append(details);
    }
    if (['QUEUED','RUNNING','AWAITING_VALIDATION'].includes(job.status)) {
      const button = document.createElement('button'); button.textContent = '작업 취소';
      button.onclick = () => api(`/api/jobs/${job.id}/cancel?session=${encodeURIComponent(selectedSession)}`,{}).then(refreshJobs).catch(e => notice(e.message)); article.append(button);
    }
    if (job.error) { const error = document.createElement('pre'); error.textContent = job.error; article.append(error); }
    if(job.output_path || job.output_error){const output=document.createElement('p');output.className='hint';output.textContent=job.output_error?'출력 폴더 저장 실패 · Workspace 원본은 보존됨: '+job.output_error:'출력 위치: '+job.output_path;article.append(output);}
    if(job.status==='SUCCEEDED' && ['FAILED','INTERRUPTED'].includes(job.output_status)) {
      const retry=document.createElement('button');retry.textContent='출력 폴더에 다시 저장';
      const session=selectedSession;
      retry.onclick=async()=>{retry.disabled=true;retry.textContent='다시 저장 중…';try{await api(`/api/jobs/${job.id}/export?session=${encodeURIComponent(session)}`,{});await refreshJobs();}catch(error){notice(error.message);}finally{retry.disabled=false;retry.textContent='출력 폴더에 다시 저장';}};
      article.append(retry);
    }
    if (job.has_original) {
      const link=document.createElement('a'); link.className='download'; link.textContent='생성 원본 저장';
      link.href=`/api/jobs/${job.id}/original.png?session=${encodeURIComponent(selectedSession)}`;
      link.download=`${job.id}-original.png`; article.append(link);
      if (job.status !== 'SUCCEEDED') {
        article.append(resultPreview(document,job,selectedSession,{selected:resultSelections.get(job.id),onSelect:name=>resultSelections.set(job.id,name)}));
      }
    }
    const detailFiles=(job.artifacts || []).filter(name=>name.startsWith('detail'));
    if(detailFiles.length) {
      const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='영역 보정 결과 / 마스크 / 전후 비교';details.append(summary);
      for(const name of detailFiles) {const link=document.createElement('a');link.className='download';link.href=`/api/jobs/${job.id}/${name}?session=${encodeURIComponent(selectedSession)}`;link.download=name;link.textContent=name==='detailed.png' ? '보정본' : name==='detail-mask.png' ? '보정 마스크' : name;details.append(link);}
      for(let i=0;i<8;i++) {
        if(!detailFiles.includes(`detail-${i}-before.png`)&&!detailFiles.includes(`detail-${i}-after.png`))continue;
        const pair=document.createElement('div');pair.className='detail-comparison';
        for(const side of ['before','after']) {
          const name=`detail-${i}-${side}.png`;if(!detailFiles.includes(name))continue;
          const figure=document.createElement('figure'),link=document.createElement('a'),image=document.createElement('img'),caption=document.createElement('figcaption');
          link.href=`/api/jobs/${job.id}/${name}?session=${encodeURIComponent(selectedSession)}`;link.target='_blank';link.rel='noopener noreferrer';link.title='새 탭에서 보정 영역 크게 보기';
          image.src=link.href;image.loading='lazy';image.alt=`영역 ${i+1} ${side==='before' ? '보정 전' : '보정 후'}`;caption.textContent=image.alt;
          link.append(image);figure.append(link,caption);pair.append(figure);
        }
        details.append(pair);
      }
      article.append(details);
    }
    if(job.has_original && originalInputSupported) {
      const originalEdit=document.createElement('button');originalEdit.type='button';originalEdit.textContent='원본으로 변형';
      originalEdit.onclick=()=>{if(busy||!workspaceReady)return;const family=imageFamily(modelInventory.find(m=>m.id===$('model').value));if(family==='zimage'){notice('Z-Image-Turbo는 현재 이미지 변형을 지원하지 않습니다.');return;}if(family==='flux2'){$('flow').value='reference_edit';generationReference.restore({reference_job:job.id,reference_artifact:'original.png'});}else{$('flow').value='img2img';$('source_job').value=job.id;$('source_artifact').value='original.png';$('mask_id').value='';}updateFlow();focusSetting(family==='flux2'?'generation-reference':'source_job');};
      article.append(originalEdit);
    }
    if (job.status === 'SUCCEEDED') {
      const edit= document.createElement('button'); edit.textContent='최종 결과로 변형';
      edit.onclick=() => { if (busy) return; if(imageFamily(modelInventory.find(m=>m.id===$('model').value))==='zimage'){notice('Z-Image-Turbo는 현재 텍스트 생성만 지원합니다. 이미지 변형을 지원하는 모델을 선택하세요.');return;} if(imageFamily(modelInventory.find(m=>m.id===$('model').value))==='flux2'){$('flow').value='reference_edit';generationReference.restore({reference_job:job.id});updateFlow();focusSetting('generation-reference');}else{$('flow').value='img2img'; $('source_job').value=job.id; $('source_artifact').value='image.png';$('mask_id').value=''; updateFlow(); focusSetting('source_job');} };
      article.append(edit);
      const reuseSeed=document.createElement('button');reuseSeed.type='button';reuseSeed.textContent='이 Seed 고정';
      reuseSeed.onclick=()=>{if(busy)return;$('random_seed').checked=false;$('seed').value=job.plan.settings.seed;updateFlow();focusSetting('seed');};
      if(job.plan?.settings)article.append(reuseSeed);
      const base = `/api/jobs/${job.id}/`; const query = '?session=' + encodeURIComponent(selectedSession);
      article.append(resultPreview(document,job,selectedSession,{selected:resultSelections.get(job.id),onSelect:name=>resultSelections.set(job.id,name)}));
      for (const [name,label] of [['image.png','이미지 저장'],['report.json','실행 설정 / 보고서']]) {
        const link = document.createElement('a'); link.href = base + name + query; link.download = `${job.id}-${name}`; link.textContent = label; link.className = 'download'; article.append(link);
      }
    }
    return article;
  }));
  renderComposition();
}
$('editor').onsubmit = async event => {
  event.preventDefault(); if (busy || !workspaceReady) return;
  if(generationReference.pending()){notice('참고 이미지 첨부가 끝난 뒤 생성하세요.');return;}
  busy = true;
  const generating = event.submitter?.id === 'generate';
  const body = {session, prompt_id:promptId, expected_revision:revision, positive:$('positive').value,
    negative:$('negative').value, settings:settings(), generate:generating};
  const signature = JSON.stringify(body);
  if (!lastSubmission || lastSubmission.signature !== signature) lastSubmission = {signature, id:crypto.randomUUID()};
  body.request_id = lastSubmission.id;
  const controls = [...document.querySelectorAll('#editor input, #editor textarea, #editor select, #editor button'), $('sessions'), $('workspace-picker'), $('new-session')];
  controls.forEach(control => control.disabled = true);
  try {
    const receipt = await api('/api/prompts', body);
    revision = receipt.revision; $('revision').textContent = `revision ${revision}`;
    const warning=await refreshAfterSave(generating);
    if(generating)showStudio('results');
    notice(warning || (generating ? '저장했습니다. 작업은 차례로 실행됩니다.' : '프롬프트를 저장했습니다.'));
  } catch (error) { notice(error.message); }
  finally { busy = false; controls.forEach(control => control.disabled = false); updateFlow(); }
};
$('sessions').onchange = () => changeSession().catch(e => notice(e.message));
$('workspace-retry').onclick = () => changeSession().catch(e => notice(e.message));
$('prompts').onchange = () => loadPrompt($('prompts').value).catch(e=>notice(e.message));
$('editor').addEventListener('invalid',event=>focusSetting(event.target.id),true);
$('new-prompt').onclick = newPrompt;
$('rescan').onclick = () => scanModels().catch(e => notice(e.message));
$('model-type').onchange = renderModels;
$('model-search').oninput=renderModels;
for(const scope of ['all','compatible','selected'])$('model-scope-'+scope).onclick=()=>{modelScope=scope;renderModels();};
$('model-filter-reset').onclick=()=>{$('model-search').value='';$('model-type').value='all';modelScope='all';renderModels();$('model-search').focus();};
$('flow').onchange = updateFlow;
$('model').addEventListener('change',()=>{
  const family=imageFamily(modelInventory.find(m=>m.id===$('model').value));
  if(['flux2','zimage','krea2'].includes(family)) {
    const selected=modelSelectionSettings(family,settings());
    for(const key of ['prediction_type','guidance','prompt_mode','flow','source_job','mask_id'])$(key).value=selected[key] ?? '';
    updateFlow();
  }
});
$('random_seed').onchange=updateFlow;
$('editor').addEventListener('input',event => {if (event.target.tagName !== 'TEXTAREA') renderComposition(); else checkTokens();});
$('editor').addEventListener('change',event => {renderComposition();renderModels();if(['model','prompt_mode'].includes(event.target.id))checkTokens();});
$('llm-model').addEventListener('change',renderComposition);
$('chat-action').addEventListener('change',renderComposition);
$('new-session').onclick = async () => { try { const title = 'Workspace ' + new Date().toLocaleString(); const result = await api('/api/sessions', {title}); await refreshWorkspaceList(result.ref); await changeSession(); } catch(e) {notice(e.message);} };
async function start() {
  originalInputSupported=false;
  try {
    const schema=await api('/openapi.json');
    originalInputSupported=!!schema?.components?.schemas?.Settings?.properties?.source_artifact;
  } catch(error) {
    $('source-version-status').textContent='원본 입력 지원 여부를 확인하지 못했습니다. 새로고침해 다시 확인하세요: '+error.message;
  }
  $('source-version-status').hidden=originalInputSupported;
  $('startup-status').textContent='모델 폴더를 확인하는 중…';
  await scanModels();
  $('startup-status').textContent='Workspace 목록을 불러오는 중…';
  await refreshWorkspaceList();
  $('startup-status').textContent='프롬프트·이미지·대화를 불러오는 중…';
  await changeSession();
  $('startup-status').textContent='Ollama 모델을 확인하는 중…';
  await chat.connect();
  renderComposition();
  await refreshImageRuntime();
  let polling=false;
  setInterval(async()=>{
    if(polling)return;
    polling=true;
    try {
      const results=await Promise.allSettled([refreshJobs().then(refreshStorageErrors),refreshImageRuntime()]);
      for(const result of results)if(result.status==='rejected')notice(result.reason.message);
    } finally {polling=false;}
  },2000);
}
export const ready = start();

$('backup-data').onclick = async () => {
  const button=$('backup-data');button.disabled=true;
  $('backup-status').textContent='백업 중… 잠시 저장 요청이 대기할 수 있습니다.';
  try {const result=await api('/api/backups',{});$('backup-status').textContent=`백업 완료: data/backups/${result.id} · ${result.files}개 파일 · 모델 가중치 제외`;}
  catch(error){$('backup-status').textContent=`백업 실패: ${error.message}`;}
  finally {button.disabled=false;}
};

let galleryPage=1,galleryEpoch=0,galleryQuery='';
async function refreshGallery(rebuild=false) {
  const epoch=++galleryEpoch;
  $('gallery-status').textContent='이미지 목록을 불러오는 중…';
  try {
    if(rebuild) await api('/api/images/rebuild',{});
    if(epoch!==galleryEpoch)return;
    const result=await api(`/api/images?page=${galleryPage}&q=${encodeURIComponent(galleryQuery)}`);
    if(epoch!==galleryEpoch)return;
    $('gallery-items').replaceChildren();
    $('gallery-status').textContent=`${result.total}개 · ${result.page}페이지`+(result.errors.length?` · 읽지 못한 기록 ${result.errors.length}개`: '');
    $('gallery-prev').disabled=galleryPage<=1;$('gallery-next').disabled=galleryPage*result.size>=result.total;
    for(const row of result.items) {
      const card=document.createElement('article'),link=document.createElement('a'),image=document.createElement('img'),label=document.createElement('p'),button=document.createElement('button');
      link.href=`/api/jobs/${row.job_id}/${row.name}?session=${encodeURIComponent(row.session)}`;link.target='_blank';link.rel='noopener';
      image.src=`/api/images/${row.job_id}/thumbnail?session=${encodeURIComponent(row.session)}&name=${encodeURIComponent(row.name)}`;image.loading='lazy';image.decoding='async';image.alt=`${row.title} · ${row.name}`;link.append(image);
      label.textContent=`${row.title} · ${row.name==='original.png'?'원본':row.name==='detailed.png'?'보정본':'최종 결과'}`;
      button.type='button';button.textContent='Workspace 열기';button.onclick=async()=>{if(busy)return;try {$('include-archived').checked=true;await refreshWorkspaceList(row.session);await changeSession();}catch(error){notice(error.message);}};
      card.append(link,label,button);$('gallery-items').append(card);
    }
  } catch(error){if(epoch===galleryEpoch)$('gallery-status').textContent=error.message;}
}
$('nav-gallery').addEventListener('click',()=>refreshGallery());
$('gallery-search').onsubmit=event=>{event.preventDefault();galleryQuery=$('gallery-query').value;galleryPage=1;refreshGallery();};
$('gallery-rebuild').onclick=()=>{galleryPage=1;refreshGallery(true);};
$('gallery-prev').onclick=()=>{galleryPage=Math.max(1,galleryPage-1);refreshGallery();};
$('gallery-next').onclick=()=>{galleryPage++;refreshGallery();};

async function refreshWorkspaceList(selected=session) {
  workspaceRecords=await api('/api/sessions?include_archived='+$('include-archived').checked);
  if(!workspaceRecords.length){await api('/api/sessions',{title:'새 Workspace'});workspaceRecords=await api('/api/sessions');}
  renderWorkspaces($('sessions'),$('workspace-search-status'),workspaceRecords,selected,$('workspace-search').value,$('workspace-list'));
}
async function updateWorkspace(archive) {
  if(busy)return;
  const record=workspaceRecords.find(row=>row.ref===session);if(!record)return;
  busy=true;
  const controls=['rename-workspace','archive-workspace','sessions','workspace-picker','new-session','include-archived'].map($);
  controls.forEach(control=>control.disabled=true);
  try {
    const result=await api('/api/sessions/update',{session:record.ref,expected_revision:record.revision,title:$('workspace-title').value,archived:archive?!record.archived:!!record.archived});
    if(result.archived)$('include-archived').checked=true;
    await refreshWorkspaceList(record.ref);
    $('workspace-name').textContent=$('sessions').selectedOptions[0]?.textContent || result.title;
    $('archive-workspace').textContent=result.archived?'보관 해제':'보관';
    $('workspace-status').textContent=result.archived?'보관했습니다. 대화와 이미지는 그대로 유지됩니다.':'저장했습니다.';
  } catch(error){$('workspace-status').textContent=error.message;await refreshWorkspaceList(session);}
  finally {busy=false;controls.forEach(control=>control.disabled=false);}
}
$('workspace-search').oninput=()=>renderWorkspaces($('sessions'),$('workspace-search-status'),workspaceRecords,$('sessions').value,$('workspace-search').value,$('workspace-list'));
$('rename-workspace').onclick=()=>updateWorkspace(false);
$('archive-workspace').onclick=()=>updateWorkspace(true);
$('include-archived').onchange=async()=>{if(busy)return;try {await refreshWorkspaceList();await changeSession();}catch(error){notice(error.message);}};

setupMask({api,getContext:context,changed:renderComposition});

setupAppSettings({api,onSaved:async()=>{await scanModels();await chat.connect();await refreshImageRuntime();}});
