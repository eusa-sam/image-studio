import {modelType} from './navigation.mjs?v=20261107-ui32';

export function jobStatusLabel(status) {
  return ({AWAITING_VALIDATION:'검사 대기',QUEUED:'순서 대기',RUNNING:'작업 중',SUCCEEDED:'완료',FAILED:'실패',PARTIAL_FAILED:'후처리 실패 · 원본 보존',CANCELLED:'취소됨',INTERRUPTED:'중단됨'})[status] || status;
}

export function jobProgress(job) {
  if(!['AWAITING_VALIDATION','QUEUED','RUNNING'].includes(job.status))return null;
  const p=job.progress || {};
  const labels={verify:'모델 파일 확인 중',load:'모델을 준비하는 중',condition:'프롬프트와 입력 이미지 준비 중',generate:'이미지 생성 중',decode:'이미지 복원 중',detect:'얼굴·손 영역 찾는 중',detail:'영역 보정 중',upscale:'이미지 확대 중',save:'결과 저장 중'};
  let label=job.status==='RUNNING'?(labels[p.stage] || '작업 시작 중'):'GPU 작업 순서를 기다리는 중';
  let count=null;
  if(job.status==='RUNNING') {
    if(p.stage==='generate'&&!Number.isInteger(p.step))label+=' · 첫 단계 완료 대기';
    if(p.stage==='detail') {
      label=(p.target==='face'?'얼굴':p.target==='hand'?'손':'영역')+' 보정 중';
      if(Number.isInteger(p.region)&&Number.isInteger(p.regions)&&p.region>0&&p.region<=p.regions)label+=` · 영역 ${p.region}/${p.regions}`;
      if(p.decoding)label+=' · 이미지 복원 중';
    }
    if(['generate','detail','upscale'].includes(p.stage)&&!p.decoding&&Number.isInteger(p.step)&&Number.isInteger(p.total)&&p.total>0&&p.step>=0&&p.step<=p.total) {
      count={value:p.step,max:p.total};label+=` · ${p.step}/${p.total} ${p.stage==='upscale'?'타일':'단계'}`;
    }
  }
  const modes={full:'GPU 전체 적재',partial:'GPU·RAM 부분 적재',model_offload:'모델 단위 오프로딩',sequential_offload:'레이어 단위 오프로딩',block_offload:'일부 블록 GPU 상주',auto_model_offload:'가용 VRAM에 따라 자동 전환'};
  const policy=job.status==='RUNNING'?p.memory_policy:null;
  let memory=policy?modes[policy.mode] || '': '';
  if(memory)for(const [key,title] of [['free_vram_bytes','시작 시 가용 VRAM'],['resident_budget_bytes','가중치 VRAM 예산'],['cpu_cache_budget_bytes','RAM 캐시 상한']]) {
    if(key==='cpu_cache_budget_bytes'&&policy.mode!=='partial')continue;
    if(Number.isFinite(policy[key])&&policy[key]>=0)memory+=` · ${title} ${(policy[key]/1024**3).toFixed(1)} GiB`;
  }
  return {label,count,memory};
}

export function renderJobProgress(container,job) {
  const state=jobProgress(job);
  const signature=JSON.stringify(state);
  if(container.dataset.progress===signature)return;
  container.dataset.progress=signature;
  container.hidden=!state;
  if(!state){container.replaceChildren();return;}
  const doc=container.ownerDocument,label=doc.createElement('p'),bar=doc.createElement('progress');
  label.textContent=state.label;bar.setAttribute('aria-label',state.label);
  if(state.count){bar.max=state.count.max;bar.value=state.count.value;}
  const nodes=[label,bar];
  if(state.memory){const memory=doc.createElement('small');memory.textContent=state.memory;nodes.push(memory);}
  container.replaceChildren(...nodes);
}

export function imageFamily(checkpoint) {
  if(['flux2','zimage'].includes(checkpoint?.family))return checkpoint.family;
  return checkpoint?.family==='krea2' ? 'krea2' : checkpoint?.family==='anima' ? 'anima' : checkpoint?.family==='sd2_candidate' ? 'sd2' : ['sd1_candidate','sd1_inpaint_candidate'].includes(checkpoint?.family) ? 'sd1' : 'sdxl';
}

export function componentCompatibility(model, checkpoint, field) {
  if(field==='model')return model?.available ? null : model?.reason || '실행기 준비가 필요합니다.';
  if(!model?.component)return model?.reason || '모델이 없거나 호환 프로필이 없습니다.';
  if(!checkpoint?.available)return 'Checkpoint를 먼저 선택하세요.';
  const family=imageFamily(checkpoint), profile=model.profile || {};
  const expected=field.startsWith('lora')?'lora':['clip','clip2'].includes(field)?'text_encoder':field.endsWith('_detector')?'detector':field;
  if(model.component!==expected)return '이 슬롯의 구성요소가 아닙니다.';
  if(expected==='upscaler')return null;
  if(expected==='embedding'&&!['sd1','sd2','sdxl'].includes(family))return '임베딩은 SD1/SD2/SDXL만 지원합니다.';
  if(['zimage','flux2'].includes(family)&&expected!=='lora')return `${family==='flux2'?'Klein':'Z-Image-Turbo'}는 공식 번들 구성과 업스케일러를 사용합니다. 이 추가 구성은 아직 미지원입니다.`;
  if(expected==='detector')return profile.target===field.split('_')[0] ? null : '검출 대상이 다릅니다.';
  if(expected==='text_encoder' && (family==='anima'||(field==='clip2'&&family!=='sdxl')))return `${family.toUpperCase()}에는 이 인코더 슬롯이 없습니다.`;
  if((profile.family || 'sdxl')!==family)return `${(profile.family || 'sdxl').toUpperCase()} 전용 · 현재 ${family.toUpperCase()}`;
  if(expected==='text_encoder'&&profile.slot!==(field==='clip2'?2:1))return '인코더 슬롯이 다릅니다.';
  return null;
}

export function componentChoices(field, checkpoint, models, selected='') {
  const expected=field.startsWith('lora')?'lora':['clip','clip2'].includes(field)?'text_encoder':field.endsWith('_detector')?'detector':field;
  const rows=models.filter(m=>m.component===expected && (expected!=='text_encoder'||m.profile.slot===(field==='clip2'?2:1)) && (expected!=='detector'||m.profile.target===field.split('_')[0])).map(model=>({id:model.id,name:model.name,reason:componentCompatibility(model,checkpoint,field)}));
  if(selected && !rows.some(row=>row.id===selected)) {
    const model=models.find(m=>m.id===selected);
    rows.push({id:selected,name:model?.name || selected,reason:model ? componentCompatibility(model,checkpoint,field) : '누락 또는 이 슬롯에서 미지원'});
  }
  return rows;
}

export function renderComponentSelect(select, field, checkpoint, models, label) {
  const selected=select.value,doc=select.ownerDocument;
  const option=(value,text)=>{const node=doc.createElement('option');node.value=value;node.textContent=text;return node;};
  const groups=[doc.createElement('optgroup'),doc.createElement('optgroup')];
  groups[0].label='현재 모델과 호환';groups[1].label='선택 불가 · 기존 선택은 유지';
  for(const row of componentChoices(field,checkpoint,models,selected)) {
    const node=option(row.id,row.name+(row.reason?' · '+row.reason:''));node.disabled=!!row.reason;node.title=row.reason || '실행 시 가중치/shape 최종 검사';
    groups[row.reason?1:0].append(node);
  }
  select.replaceChildren(option('',label),...groups.filter(group=>group.children.length));
  select.value=selected;
}

export function modelChoice(model) {
  if(model.available&&model.family==='zimage') {
    const variant={original:'원본 번들 · 로딩 시 BF16 변환 필요', 'bf16-local-v1':'BF16 준비본 · 로딩 시 변환 생략'}[model.variant];
    return {field:'model',label:'Checkpoint 선택 가능',detail:`Z-Image-Turbo${variant?' · '+variant:''} · 텍스트 생성 · 9 steps (8회 추론) 권장 · Negative 미사용 · Z-Image LoRA · 디테일러 미지원`};
  }
  if(model.available&&model.family==='flux2')return {field:'model',label:'Checkpoint 선택 가능',detail:'FLUX.2 Klein 4B · 텍스트 생성 / 참고 이미지 한 장 편집 · 4 steps 권장 · FLUX.2 LoRA · 디테일러 미지원'};
  if (model.available) return {field:'model', label:'Checkpoint 선택 가능', detail:`${model.family==='krea2' ? 'Krea2 Turbo' : model.family==='anima' ? 'ANIMA' : model.family==='sd2_candidate' ? 'SD2' : ['sd1_candidate','sd1_inpaint_candidate'].includes(model.family) ? 'SD1' : 'SDXL'} ${model.family==='sd1_inpaint_candidate'?'9채널 · 부분 수정 전용':'구조 후보'} · 로딩 시 최종 검사`};
  if(model.component==='embedding')return {field:'embedding',label:'임베딩 선택 가능',detail:`${model.profile.family} · 필요 토큰 ${(model.profile.tokens||[]).join(' ')}`};
  if(model.component==='text_encoder')return {field:model.profile.slot===2?'clip2':'clip',label:'CLIP 선택 가능',detail:`${model.profile.family} 슬롯 ${model.profile.slot} · 전체 가중치 검사`};
  if(model.component==='detector') return {field:model.profile.target+'_detector',label:'자동 보정 검출기 선택 가능',detail:model.profile.target==='face' ? '얼굴 전용 · 로컬 CPU 검출' : '손 전용 · 로컬 CPU 검출'};
  if (model.component === 'lora') return {field:'lora0',label:`${model.profile?.family || 'sdxl'} LoRA 선택 가능`,detail:'프로필 후보 · 실행 시 해시/대상 레이어/shape 검사'};
  if (model.component === 'vae') return {field:'vae', label:'VAE 선택 가능', detail:`${model.profile?.family || 'sdxl'} 프로필 · 실행 시 SHA256 검사`};
  if (model.component === 'upscaler') return {field:'upscaler', label:`${model.profile.scale}× 확대 선택 가능`, detail:'프로필 배율 · 실행 시 로더 검사'};
  if(['ip_adapter','clip_vision'].includes(model.component))return {field:model.component,label:'SDXL 이미지 참고 선택 가능',detail:'IP-Adapter + CLIP Vision + 참고 이미지 한 장 필요'};
  const kind=modelType(model.path);
  const detail={lora:'계열별 호환 프로필 없음 · 현재 생성에 적용되지 않습니다.',
    text_encoder:'표준 CLIP 파일의 계열·슬롯·해시 프로필이 필요합니다.',
    clip_vision:'이미지 인코더 입력 흐름 미구현', detector:'호환되는 얼굴/손 ONNX 프로필이 필요합니다.',
    vae:'계열별 호환 프로필이 필요합니다.', upscaler:'호환 프로필이 필요합니다.'}[kind];
  return {field:null, label:'선택 불가', detail:model.reason || detail || '현재 실행기 미지원'};
}

export function modelSelectionSettings(family, settings) {
  if(!['flux2','zimage','krea2'].includes(family))return {...settings};
  const result={...settings,prediction_type:'flow',guidance:family==='flux2'?1:0,prompt_mode:'native'};
  const flows=family==='flux2'?['txt2img','reference_edit']:family==='krea2'?['txt2img','img2img','inpaint']:['txt2img'];
  if(!flows.includes(result.flow))result.flow='txt2img';
  if(family!=='krea2'){result.source_job='';result.mask_id='';}
  return result;
}

export function describePipeline(settings, inventory) {
  const find=id => inventory.find(m => m.id === id);
  const checkpoint=find(settings.model), vae=find(settings.vae), upscaler=find(settings.upscaler);
  const family=imageFamily(checkpoint);
  const referenceEdit=family==='flux2'&&settings.flow==='reference_edit';
  const errors=[];
  if(settings.flow==='reference_edit'&&family!=='flux2')errors.push('참고 이미지 편집은 Klein을 선택하세요.');
  if(family==='flux2') {
    if(!['txt2img','reference_edit'].includes(settings.flow)||settings.guidance!==1||settings.prompt_mode!=='native'||settings.prediction_type!=='flow')errors.push('Klein: 텍스트 생성 / 참고 이미지 편집 · flow · Guidance 1 · 기본 인코딩 · 4 steps 권장');
    if(['clip','clip2','vae','ip_adapter','clip_vision','face_detector','hand_detector','source_job','mask_id'].some(key=>settings[key]))errors.push('Klein은 공식 번들 인코더/VAE를 사용합니다. 추가 디테일러·마스크 구성은 아직 미지원입니다.');
    if(referenceEdit&&Number(!!settings.reference_job)+Number(!!settings.reference_attachment)!==1)errors.push('생성 결과 또는 사용자 참고 이미지 한 장을 선택하세요.');
  }
  if(family==='zimage') {
    if(settings.flow!=='txt2img'||settings.guidance!==0||settings.prompt_mode!=='native'||settings.prediction_type!=='flow')errors.push('Z-Image-Turbo: 텍스트 생성 · flow · Guidance 0 · 기본 인코딩 · 9 steps 권장');
    if(['clip','clip2','vae','ip_adapter','clip_vision','face_detector','hand_detector','source_job','mask_id','reference_job','reference_attachment'].some(key=>settings[key]))errors.push('Z-Image-Turbo의 추가 참고 이미지·디테일러·마스크 구성은 아직 미지원입니다.');
  }
  const dedicated=checkpoint?.family==='sd1_inpaint_candidate';
  if(dedicated && settings.flow!=='inpaint')errors.push('9채널 전용 모델은 마스크가 있는 부분 수정에서만 사용할 수 있습니다.');
  if(family==='krea2' && (!['txt2img','img2img','inpaint'].includes(settings.flow)||settings.guidance!==0||settings.prompt_mode!=='native'||settings.clip2))errors.push('Krea2 Turbo: 텍스트 생성/이미지 변형/마스크 수정 · Guidance 0 · 기본 인코딩.');
  const reference=settings.reference_attachment?'사용자 참고 · '+settings.reference_attachment.slice(0,8):settings.reference_job?(settings.reference_artifact==='original.png'?'생성 원본 · ':'생성 결과 · ')+settings.reference_job.slice(0,8):'참고 이미지 미선택';
  if(settings.ip_adapter) {
    for(const field of ['ip_adapter','clip_vision']) {const reason=componentCompatibility(find(settings[field]),checkpoint,field);if(reason)errors.push(field+': '+reason);}
    if(family!=='sdxl')errors.push('IP-Adapter는 SDXL에서만 사용할 수 있습니다.');
    if(Number(!!settings.reference_job)+Number(!!settings.reference_attachment)!==1)errors.push('생성 결과 또는 사용자 참고 이미지 한 장을 선택하세요.');
    if(!Number.isFinite(settings.ip_adapter_scale)||settings.ip_adapter_scale<0||settings.ip_adapter_scale>2)errors.push('참고 강도는 0~2입니다.');
  } else if(!referenceEdit&&(settings.clip_vision||settings.reference_job||settings.reference_attachment))errors.push(family==='flux2'?'참고 이미지 편집을 선택하세요.':'이미지를 참고하려면 IP-Adapter를 선택하세요.');
  const clips=['clip','clip2'].filter(key=>settings[key]);
  const embeddings=settings.embeddings||[];
  if(embeddings.length>4||new Set(embeddings).size!==embeddings.length)errors.push('임베딩은 중복 없이 최대 4개입니다.');
  for(const id of embeddings){const reason=componentCompatibility(find(id),checkpoint,'embedding');if(reason)errors.push('임베딩: '+reason);}
  for(const key of clips){const item=find(settings[key]);if(family==='anima'||(key==='clip2'&&family!=='sdxl')||item?.component!=='text_encoder'||item.profile.family!==family||item.profile.slot!==(key==='clip'?1:2))errors.push('CLIP 계열/슬롯/프로필을 확인하세요: '+key);}
  const clipNames=clips.map(key=>find(settings[key])?.name || '누락 CLIP').join(' + ');

  if(family==='anima' && settings.prompt_mode==='weighted')errors.push('ANIMA는 기본 프롬프트 모드를 선택하세요.');
  const variant=checkpoint?.anima?.variant;
  if(variant==='turbo' && settings.guidance!==1)errors.push('ANIMA Turbo: Guidance 1 · 권장 Steps 8~12');
  if((['anima','krea2','flux2','zimage'].includes(family)) !== (settings.prediction_type==='flow'))errors.push('ANIMA·Krea2·Klein·Z-Image는 flow, SD 계열은 epsilon/v_prediction을 선택하세요.');
  for(const m of [vae,...(settings.loras || []).map(l=>find(l.model))].filter(Boolean)) {
    if(m.profile && (m.profile.family || 'sdxl')!==family) errors.push(`${m.name}: ${family} Checkpoint와 계열이 다릅니다.`);
  }
  const loras=settings.loras || [];
  if(new Set(loras.map(l => l.model)).size !== loras.length) errors.push('같은 LoRA를 중복 선택할 수 없습니다.');
  if(loras.some(l => find(l.model)?.component !== 'lora')) errors.push('선택한 LoRA의 프로필을 확인하세요.');
  const loraNames=loras.map(l => `${find(l.model)?.name || '누락 LoRA'} (${l.weight})`).join(', ');
  if (!checkpoint?.available) errors.push('실행 가능한 Checkpoint를 선택하세요.');
  if (settings.vae && vae?.component !== 'vae') errors.push('선택한 VAE를 찾을 수 없거나 프로필이 없습니다.');
  if (settings.upscaler && upscaler?.component !== 'upscaler') errors.push('선택한 업스케일러를 찾을 수 없거나 프로필이 없습니다.');
  const detectors=['face','hand'].filter(target=>settings[target+'_detector']);
  for(const target of detectors) if(find(settings[target+'_detector'])?.profile?.target!==target)errors.push('검출기 대상/프로필을 확인하세요: '+target);
  const detailNames=detectors.map(target=>find(settings[target+'_detector'])?.name || '누락 검출기').join(' + ');
  const editing=['img2img','inpaint'].includes(settings.flow);
  if(settings.flow==='inpaint' && (!settings.mask_id || family==='anima'))errors.push('마스크를 저장하고 SD 또는 Krea2 모델을 선택하세요.');
  if (editing && !settings.source_job) errors.push('변형할 입력 이미지를 선택하세요.');
  if (editing && Math.floor(settings.steps * settings.strength) < 1) errors.push('Steps × 변형 강도가 1 이상이어야 합니다.');
  const scale=settings.upscaler && upscaler?.component === 'upscaler' ? upscaler.profile.scale : 1;
  const size=`${settings.width} × ${settings.height}`;
  const vaeName=settings.vae ? vae?.name || 'VAE 선택 확인 필요' : family==='krea2' ? checkpoint.krea2?.vae?.split('/').at(-1) || 'Qwen Image VAE' : 'Checkpoint 내장 VAE';
  return {errors, title:referenceEdit?'참고 이미지로 편집하기':settings.flow==='inpaint'?'칠한 부분 수정':editing ? '이미지에서 변형하기' : '텍스트로 새 이미지 만들기',
    output:`${settings.width * scale} × ${settings.height * scale}`, stages:[
      {key:'input', title:editing ? '프롬프트 + 원본 이미지' : '프롬프트', value:editing ? (settings.source_job?(settings.source_artifact==='original.png'?'원본 · ':'최종 · ')+settings.source_job.slice(0,8):'입력 이미지 미선택') : ['krea2','flux2','zimage'].includes(family) ? 'Positive · Negative 미사용' : 'Positive / Negative',
        detail:editing ? `${settings.flow==='inpaint'?'마스크 '+(settings.mask_id?.slice(0,8)||'미선택')+' · ':''}${family==='krea2' && settings.flow==='inpaint'?'단계별 마스크 보존 · ':''}${size}로 맞춤 · VAE 인코딩 · 강도 ${settings.strength}` : '직접 작성 또는 LLM 제안 반영', target:editing ? 'source_job' : 'positive', tone:'violet'},
      {key:'clip', title:settings.prompt_mode==='weighted' ? '텍스트 인코딩 · 긴 프롬프트/가중치' : '텍스트 인코딩', value:clipNames || (['flux2','zimage'].includes(family) ? 'Qwen3 4B · 공식 번들' : family==='krea2' ? checkpoint.krea2?.text_encoder?.split('/').at(-1) || 'Qwen3-VL 4B' : family==='anima' ? 'Qwen3 + T5 tokenizer' : ['sd1','sd2'].includes(family) ? '내장 CLIP 1개' : '내장 CLIP 2개'), detail:family==='zimage' ? 'Z-Image-Turbo · 9 steps / 8회 추론 권장 · 상황에 맞춰 GPU 적재/오프로딩' : family==='flux2' ? 'FLUX.2 distilled · 4 steps 권장 · 상황에 맞춰 GPU 적재/오프로딩' : family==='krea2' ? 'Qwen3-VL → 28층 Transformer · 순차 CPU 오프로딩 · Turbo 8 steps 권장' : family==='anima' ? `전용 conditioner → Cosmos transformer · ${variant || 'base bundle'}` : family==='sd2' ? 'SD2의 OpenCLIP-H · 표준 CLIP 교체 가능' : family==='sd1' ? 'SD1의 CLIP-L · 표준 CLIP 교체 가능' : 'SDXL의 CLIP-L + OpenCLIP bigG · 표준 CLIP 교체 가능',target:'model',tone:'violet'},
      ...(settings.ip_adapter||referenceEdit?[{key:'reference',title:'이미지 참고',value:reference,detail:referenceEdit?'Klein 원본 참고 인코딩 · 강도 기반 img2img와 다름':`${find(settings.ip_adapter)?.name||'어댑터 미선택'} · ${find(settings.clip_vision)?.name||'인코더 미선택'} · 강도 ${settings.ip_adapter_scale}`,target:'generation-reference',tone:'violet'}]:[]),
      {key:'sampling', title:'이미지 생성', value:checkpoint?.name || 'Checkpoint 미선택',
        detail:`${dedicated?'9채널 인페인팅 · ':''}${loraNames ? 'LoRA: '+loraNames+' · ' : ''}${settings.steps} steps · CFG ${settings.guidance} · ${settings.prediction_type || 'epsilon'}`,target:'model',tone:'blue'},
      {key:'vae', title:'VAE 디코딩', value:vaeName,detail:`${size} 생성 원본 별도 저장`,target:'vae',tone:'teal'},
      {key:'detail',title:'자동 영역 보정',value:detailNames || '사용 안 함',detail:detectors.length ? `추가 생성 · ${settings.detail_resolution || 512}px · 영역마다 ${Math.floor((settings.detail_steps || 20)*(settings.detail_strength ?? 0.3))}단계 · 강도 ${settings.detail_strength} · 최대 ${settings.detail_max_regions}개 · 마스크 합성` : '보정 없이 생성 원본 유지',target:'face_detector',tone:'violet',skipped:!detectors.length},
      {key:'upscale', title:'업스케일', value:settings.upscaler ? upscaler?.name || '업스케일러 확인 필요' : '사용 안 함',
        detail:settings.upscaler ? `${scale}× 타일 확대 · ${['krea2','flux2','zimage'].includes(family)?'GPU 공간 확보 · RAM 여유 시 모델 보관':'생성 모델 해제 후 실행'}` : '확대 없이 원본 해상도 유지',target:'upscaler',tone:'amber',skipped:!settings.upscaler},
      {key:'output', title:'결과 저장', value:`${settings.width * scale} × ${settings.height * scale}`,detail:'PNG + 원본 + 실행 보고서',target:'jobs',tone:'teal'},
    ], components:[
      {label:'Checkpoint',value:checkpoint?.name || '선택 필요',state:checkpoint?.available ? '선택됨' : '미선택',target:'model'},
      {label:'VAE',value:vaeName,state:settings.vae ? '외부 VAE' : family==='krea2' ? '프로필 기본값' : '내장 사용',target:'vae'},
      {label:'LoRA',value:loraNames || '사용 안 함',state:loras.length ? `${loras.length}개 선택` : '미적용',target:'lora0'},
      ...(embeddings.length?[{label:'임베딩',value:embeddings.map(id=>find(id)?.name||'누락 임베딩').join(', '),state:`${embeddings.length}개 선택`,target:'embedding'}]:[]),
      {label:'CLIP',value:clipNames || (['flux2','zimage'].includes(family) ? 'Qwen3 4B · 공식 번들' : family==='krea2' ? 'Qwen3-VL 4B · 프로필 기본값' : 'Checkpoint 내장 인코더'),state:clips.length?'외부 선택':'내장 사용',target:'clip'},
      {label:'이미지 참고',value:settings.ip_adapter||referenceEdit?reference:'사용 안 함',state:referenceEdit?'Klein 참고 편집':settings.ip_adapter?'강도 '+settings.ip_adapter_scale:'건너뜀',target:referenceEdit?'generation-reference':'ip_adapter'},
      {label:'업스케일',value:upscaler?.name || '사용 안 함',state:settings.upscaler ? `${scale}× 선택` : '건너뜀',target:'upscaler'},
      {label:'디테일러',value:detailNames || '사용 안 함',state:detectors.length ? '영역 보정' : '건너뜀',target:'face_detector'},
    ]};
}

export function renderPipeline(container, plan, {onSelect, sourceUrl}={}) {
  const doc=container.ownerDocument;
  const list=doc.createElement('ol'); list.className='pipeline-stages'; list.setAttribute('aria-label','이미지 생성 순서');
  for (const [index,stage] of plan.stages.entries()) {
    const item=doc.createElement('li'); item.className=`pipeline-stage tone-${stage.tone}${stage.skipped ? ' stage-skipped' : ''}`;
    const card=doc.createElement(onSelect ? 'button' : 'div'); card.className='stage-card';
    if (onSelect) {card.type='button';card.onclick=() => onSelect(stage.target);card.setAttribute('aria-label',`${stage.title}: ${stage.value}. 설정으로 이동`);}
    const heading=doc.createElement('span');heading.className='stage-heading';heading.textContent=`0${index+1}  ${stage.title}`;
    const value=doc.createElement('strong');value.textContent=stage.value;
    const detail=doc.createElement('span');detail.className='stage-detail';detail.textContent=stage.detail;
    card.append(heading,value,detail);
    if (stage.key === 'input' && sourceUrl) {const image=doc.createElement('img');image.src=sourceUrl;image.alt='선택된 입력 이미지';image.className='source-thumbnail';card.append(image);}
    item.append(card);list.append(item);
  }
  container.replaceChildren(list);
}

export function recordedPipeline(plan) {
  const s=plan.settings;
  const inventory=[{id:s.model,name:plan.model_path?.split(/[\\/]/).at(-1) || s.model,available:true,anima:plan.anima,krea2:plan.krea2,family:plan.runner?.startsWith('zimage_') ? 'zimage' : plan.runner?.startsWith('flux2_') ? 'flux2' : plan.runner?.startsWith('krea2_') ? 'krea2' : plan.inpaint_channels===9 ? 'sd1_inpaint_candidate' : plan.runner?.startsWith('anima_') ? 'anima' : plan.runner?.startsWith('sd2_') ? 'sd2_candidate' : plan.runner?.startsWith('sd1_') ? 'sd1_candidate' : 'sdxl_candidate'}];
  for (const [kind,component] of Object.entries(plan.components || {})) {
    if(kind.startsWith('embedding')){inventory.push({id:s.embeddings?.[Number(kind.slice(9))],name:component.path.split(/[\\/]/).at(-1),component:'embedding',profile:component.profile});continue;}
    inventory.push({id:s[kind],name:component.path.split(/[\\/]/).at(-1),component:['clip','clip2'].includes(kind)?'text_encoder':kind.endsWith('_detector') ? 'detector' : kind,profile:component.profile});
  }
  for(const lora of plan.loras || []) inventory.push({id:lora.id,name:lora.path.split(/[\\/]/).at(-1),component:'lora',profile:lora.profile});
  return describePipeline(s,inventory);
}

export function jobTiming(job, now=Date.now()) {
  const time=value=>typeof value==='string'?Date.parse(value):NaN;
  const duration=ms=>{const seconds=Math.max(0,Math.floor(ms/1000));return seconds>=3600?`${Math.floor(seconds/3600)}시간 ${Math.floor(seconds%3600/60)}분 ${seconds%60}초`:seconds>=60?`${Math.floor(seconds/60)}분 ${seconds%60}초`:`${seconds}초`;};
  const created=time(job.created_at),started=time(job.started_at),finished=time(job.output_finished_at || job.finished_at || job.updated_at);
  const retry=time(job.output_retry_started_at);
  if(Number.isFinite(retry)) {
    if(job.output_status==='PENDING')return '출력 다시 저장 중 · '+duration(now-retry);
    const generated=time(job.finished_at),end=time(job.output_finished_at);
    return (Number.isFinite(created)&&Number.isFinite(generated)?'생성 총 '+duration(generated-created)+' · ':'')+
      (job.output_status==='SUCCEEDED'?'출력 재저장 ':'출력 재저장 미완료 ')+(Number.isFinite(end)&&end>=retry?duration(end-retry):'시간 기록 없음');
  }
  if(job.output_status==='PENDING')return Number.isFinite(created)?'출력 폴더 저장 중 · 총 '+duration(now-created):'출력 폴더 저장 중';
  if(['AWAITING_VALIDATION','QUEUED'].includes(job.status))return Number.isFinite(created)?'대기 중 · '+duration(now-created):'대기 시간 확인 불가';
  if(job.status==='RUNNING') {
    const start=Number.isFinite(started)?started:time(job.updated_at);
    return Number.isFinite(start)?'실행 중 · '+duration(now-start)+(Number.isFinite(created)?' · 대기 '+duration(start-created):''):'실행 시간 확인 불가';
  }
  if(!Number.isFinite(created)||!Number.isFinite(finished))return '소요 시간 기록 없음';
  return Number.isFinite(started)?'실행 '+duration(finished-started)+' · 대기 '+duration(started-created)+' · 총 '+duration(finished-created):'총 '+duration(finished-created)+' · 대기 포함';
}

export function resultPreview(doc, job, session, {selected, onSelect}={}) {
  const container=doc.createElement('div'),link=doc.createElement('a'),image=doc.createElement('img');
  link.target='_blank';link.rel='noopener noreferrer';link.title='새 탭에서 이미지 크게 보기';
  image.loading='lazy';link.append(image);
  const base=`/api/jobs/${encodeURIComponent(job.id)}/`,query='?session='+encodeURIComponent(session);
  const show=(name,label)=>{link.href=base+name+query;image.src=link.href;image.alt=label;};
  const originalOnly=job.has_original && job.status!=='SUCCEEDED';
  const originalSelected=job.has_original && (originalOnly || selected==='original.png');
  show(originalSelected?'original.png':'image.png',originalSelected?'보존된 생성 원본 · 클릭하면 새 탭에서 열립니다':'생성 결과 · 클릭하면 새 탭에서 열립니다');
  if(job.has_original && !originalOnly) {
    const controls=doc.createElement('div');controls.className='actions';
    const buttons=[['original.png','보정 전 원본'],['image.png','최종 결과']].map(([name,label])=>{
      const button=doc.createElement('button');button.type='button';button.textContent=label;
      button.setAttribute('aria-pressed',String(name===(originalSelected?'original.png':'image.png')));
      button.onclick=()=>{show(name,label+' · 새 탭에서 열기');onSelect?.(name);buttons.forEach(item=>item.setAttribute('aria-pressed',String(item===button)));};
      controls.append(button);return button;
    });
    container.append(controls);
  }
  container.append(link);return container;
}


export function modelInventoryRows(models, {type='all',query='',scope='all',settings={}}={}) {
  const normalize=value=>String(value ?? '').normalize('NFKC').toLocaleLowerCase();
  const terms=normalize(query).trim().split(/\s+/).filter(Boolean);
  const checkpoint=models.find(model=>model.id===settings.model);
  const assigned=new Set(['model','clip','clip2','vae','upscaler','ip_adapter','clip_vision','face_detector','hand_detector'].map(field=>settings[field]).concat((settings.loras||[]).map(row=>row.model),settings.embeddings||[]).filter(Boolean));
  return models.filter(model=>type==='all'||modelType(model.path)===type).map(model=>{
    const choice=modelChoice(model);
    const selected=assigned.has(model.id);
    const incompatibility=choice.field ? componentCompatibility(model,checkpoint,choice.field) : null;
    return {model,choice,selected,incompatibility};
  }).filter(row=>{
    const text=normalize([row.model.name,row.model.path,row.model.family,row.choice.detail].join(' '));
    return terms.every(term=>text.includes(term)) && (scope==='selected'?row.selected:scope==='compatible'?row.choice.field&&!row.incompatibility:true);
  });
}
