export function setupAppSettings({doc=document,api,onSaved}) {
  const $=id=>doc.getElementById(id),dialog=$('application-settings');
  const fields=['model_path','output_path','ollama_url','ollama_endpoint','llm_provider','openai_base_url','openai_response_format','openai_token_field'];
  let current=null,busy=false;
  const fill=row=>{current=row;for(const key of fields)$('setting-'+key).value=row[key]??'';$('setting-openai_api_key').value='';$('setting-openai_key_clear').checked=false;$('setting-key-status').textContent=row.openai_key_registered?'API 키 등록됨':'API 키 없음';provider();};
  const values=()=>{const row={...current,...Object.fromEntries(fields.filter(key=>Object.hasOwn(current,key)).map(key=>[key,$('setting-'+key).value.trim()]))};const key=$('setting-openai_api_key').value;if(key||$('setting-openai_key_clear').checked)row.openai_api_key=$('setting-openai_key_clear').checked?'':key;return row;};
  function provider(){const compatible=$('setting-llm_provider').value==='openai_compatible';$('settings-ollama').hidden=compatible;$('settings-compatible').hidden=!compatible;}
  $('setting-llm_provider').onchange=provider;
  const status=text=>{$('application-settings-status').textContent=text;};
  function lock(value){busy=value;$('application-settings-fields').disabled=value||!current;$('settings-close').disabled=value;}
  $('nav-settings').onclick=async()=>{
    current=null;for(const key of fields)$('setting-'+key).value='';
    dialog.showModal();lock(true);status('설정을 불러오는 중…');
    try{fill(await api('/api/settings'));status('변경 내용은 저장한 뒤 새 작업부터 적용됩니다.');}
    catch(error){status('설정을 불러오지 못했습니다: '+error.message+' · 닫고 다시 열어 재시도하세요.');}
    finally{lock(false);}
  };
  $('settings-close').onclick=()=>{$('setting-openai_api_key').value='';dialog.close();};
  dialog.addEventListener('cancel',event=>{if(busy)event.preventDefault();else $('setting-openai_api_key').value='';});
  $('application-settings-form').onsubmit=async event=>{
    event.preventDefault();if(busy||!current)return;lock(true);status('설정을 저장하고 있어요…');
    try{
      fill(await api('/api/settings',values()));status('저장 완료 · 모델 목록과 LLM 연결을 갱신합니다.');
      try{await onSaved();status('저장 완료 · 새 작업부터 적용됩니다.');}
      catch(error){status('설정은 저장됐지만 목록을 갱신하지 못했습니다: '+error.message);}
    }catch(error){status(error.message);}
    finally{lock(false);}
  };
  $('settings-test').onclick=async()=>{
    if(busy||!current)return;lock(true);status('LLM 서버에 연결하는 중…');
    try{const result=await api('/api/settings/connection-test',values());status(`연결 성공 · ${result.provider==='openai_compatible'?'OpenAI 호환':'Ollama '+result.version} · 대화 모델 ${result.models.length}개`);}
    catch(error){status(error.message);}
    finally{lock(false);}
  };
}
