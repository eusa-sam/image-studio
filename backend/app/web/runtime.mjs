export function runtimeView(state) {
  if(state==='warm')return {text:'모델 유지 중 · 같은 구성으로 더 빠르게 다시 생성할 수 있어요.',release:true};
  if(state==='cpu')return {text:'GPU는 비우고 모델을 RAM에 보관 중이에요. 대화 후 더 빠르게 다시 생성할 수 있어요.',release:true};
  if(state==='running')return {text:'이미지 작업 진행 중 · 중단은 작업 카드에서 할 수 있어요.',release:false};
  if(state==='released')return {text:'이미지 모델 메모리가 해제되어 있어요.',release:false};
  return {text:'이미지 모델 상태를 확인하고 있어요.',release:false};
}

export function setupImageRuntime({status,button,api,onError}) {
  let pending=false,version=0;
  function render(state) {
    const view=runtimeView(state);
    status.textContent=pending?'이미지 모델 메모리를 해제하는 중…':view.text;
    button.hidden=!view.release&&!pending;
    button.disabled=pending;
  }
  async function refresh(){
    if(pending)return;
    const current=++version;
    try {const state=await api('/api/image-runtime');if(current===version)render(state.state);}
    catch(error){if(current===version)throw error;}
  }
  button.onclick=async()=>{
    if(pending)return;
    version++;pending=true;render('warm');
    try{const state=await api('/api/image-runtime/release',{});pending=false;render(state.state);}
    catch(error){pending=false;render('unknown');onError(error.message);await refresh().catch(()=>{});}
  };
  return refresh;
}
