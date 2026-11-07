import {bindReferenceInput,validateAttachments} from './chat.mjs?v=20261107-ui32';

export function referenceSettings(value) {
  const [kind,id,artifact]=value.split(':');
  return {reference_job:kind==='job'?id||null:null,reference_attachment:kind==='upload'?id||null:null,...(kind==='job'&&artifact==='original.png'?{reference_artifact:artifact}:{})};
}

export function referenceUrl(value,session) {
  const s=referenceSettings(value),query='?session='+encodeURIComponent(session);
  return s.reference_job?'/api/jobs/'+encodeURIComponent(s.reference_job)+'/'+(s.reference_artifact||'image.png')+query:
    s.reference_attachment?'/api/references/'+encodeURIComponent(s.reference_attachment)+query:null;
}

export function setupGenerationReference({doc,api,getSession,isBusy,changed,supportsOriginal=()=>true}) {
  const $=id=>doc.getElementById(id),select=$('generation-reference'),preview=$('generation-reference-preview'),status=$('generation-reference-status');
  let uploading=false,epoch=0;
  const option=(value,label)=>{const node=doc.createElement('option');node.value=value;node.textContent=label;return node;};
  function render() {
    if(!uploading)status.textContent='';
    const url=referenceUrl(select.value,getSession());
    preview.hidden=!url;
    if(url){preview.src=url;preview.alt=select.value.startsWith('job:')?(select.value.endsWith(':original.png')?'보존된 생성 원본 참고':'생성 최종 결과 참고'):'사용자가 첨부한 참고 이미지';}
    else preview.removeAttribute('src');
  }
  function restore(settings, invalidate=true) {
    if(invalidate)epoch++;
    const value=settings.reference_attachment?'upload:'+settings.reference_attachment:settings.reference_job?'job:'+settings.reference_job+(settings.reference_artifact==='original.png'?':original.png':''):'';
    if(value&&![...select.options].some(o=>o.value===value))select.append(option(value,(value.startsWith('job:')?(settings.reference_artifact==='original.png'?'저장된 생성 원본':'저장된 최종 결과'):'저장된 사용자 참고')+' · '+value.split(':')[1].slice(0,8)));
    select.value=value;render();
  }
  function jobs(rows) {
    const value=select.value;
    const uploads=[...select.options].filter(o=>o.value.startsWith('upload:'));
    select.replaceChildren(option('','참고 이미지 선택'),...uploads,...rows.flatMap(j=>[...(j.status==='SUCCEEDED'?[option('job:'+j.id,'최종 결과 · '+j.id.slice(0,8)+' · '+new Date(j.created_at).toLocaleString())]:[]),...(j.has_original&&supportsOriginal()?[option('job:'+j.id+':original.png','생성 원본 · '+j.id.slice(0,8)+' · '+new Date(j.created_at).toLocaleString())]:[])]));
    restore(referenceSettings(value),false);
  }
  async function upload(files) {
    if(uploading||isBusy()||!files.length)return;
    const session=getSession(),current=epoch;
    try {
      if(files.length!==1)throw Error('참고 이미지 한 장을 선택하세요.');
      validateAttachments(files,0);uploading=true;$('generation-reference-upload').disabled=true;
      status.textContent='참고 이미지 저장 중…';
      const file=files[0];
      const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(Error('파일을 읽을 수 없습니다.'));reader.readAsDataURL(file);});
      const meta=await api('/api/references',{session,filename:file.name,data});
      if(session!==getSession()||current!==epoch)return;
      select.append(option('upload:'+meta.id,'사용자 참고 · '+meta.filename));
      select.value='upload:'+meta.id;render();changed();status.textContent='첨부 완료 · SDXL IP-Adapter 또는 Klein 참고 이미지 편집에 사용할 수 있습니다.';
    } catch(error){if(current===epoch)status.textContent=error.message;}
    finally{uploading=false;$('generation-reference-upload').disabled=isBusy();}
  }
  select.onchange=()=>{epoch++;status.textContent='';render();changed();};
  preview.onerror=()=>{preview.hidden=true;status.textContent='참고 이미지를 불러오지 못했습니다. 다시 선택하세요.';};
  bindReferenceInput($('generation-reference-panel'),select,$('generation-reference-file'),$('generation-reference-upload'),upload);
  return {jobs,restore,reset(){epoch++;select.replaceChildren(option('','참고 이미지 선택'));status.textContent='';render();},pending:()=>uploading};
}
