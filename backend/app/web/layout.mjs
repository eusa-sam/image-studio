export function fitPanels(width,left,right,collapsed=false,changed=null) {
  const available=Math.max(0,width-360),minimumLeft=collapsed?64:220;
  left=collapsed?64:Math.max(220,Number.isFinite(left)?left:250);
  right=Math.max(320,Number.isFinite(right)?right:390);
  if(changed==='right')right=Math.min(right,Math.max(320,available-left));
  else left=Math.min(left,Math.max(minimumLeft,available-right));
  right=Math.min(right,Math.max(320,available-left));
  return {left,right};
}

export function setupPanelResize({doc=document,win=window,storage=globalThis.localStorage}={}) {
  const layout=doc.getElementById('layout'),leftPanel=doc.querySelector('.sidebar'),rightPanel=doc.querySelector('.llm-panel');
  const key='image-maker.panel-widths';let preferred=null,drag=null;
  try{const saved=JSON.parse(storage?.getItem(key)||'null');if(saved&&Number.isFinite(saved.left)&&Number.isFinite(saved.right))preferred=saved;}catch{}
  const collapsed=()=>layout.classList.contains('sidebar-collapsed');
  const defaults=()=>({left:win.innerWidth>=1600?270:250,right:win.innerWidth>=1600?430:390});
  const handles={left:doc.getElementById('resize-left'),right:doc.getElementById('resize-right')};
  const inputs={left:doc.getElementById('panel-width-left'),right:doc.getElementById('panel-width-right')};
  function render(changed=null) {
    const desktop=win.innerWidth>1050;
    const size=fitPanels(layout.clientWidth,(preferred||defaults()).left,(preferred||defaults()).right,collapsed(),changed);
    for(const side of ['left','right']) {
      layout.style.setProperty('--'+side+'-width',size[side]+'px');
      handles[side].setAttribute('aria-valuenow',String(Math.round(size[side])));
      handles[side].setAttribute('aria-valuemin',String(side==='left'?220:320));
      handles[side].setAttribute('aria-valuemax',String(Math.floor(layout.clientWidth-360-size[side==='left'?'right':'left'])));
      handles[side].hidden=!desktop||(side==='left'&&collapsed());
      inputs[side].disabled=!desktop||(side==='left'&&collapsed());
      inputs[side].value=Math.round(size[side]);
    }
    return size;
  }
  function save(){try{storage?.setItem(key,JSON.stringify(preferred));}catch{}}
  function set(side,value){
    if(!Number.isFinite(value))return;
    const rememberedLeft=(preferred||defaults()).left;
    const actual=render();actual[side]=value;
    preferred=fitPanels(layout.clientWidth,actual.left,actual.right,collapsed(),side);
    if(collapsed())preferred.left=rememberedLeft;
    render(side);save();
  }
  for(const side of ['left','right']) {
    const handle=handles[side];
    handle.addEventListener('pointerdown',event=>{
      if(event.button!==0||win.innerWidth<=1050)return;
      event.preventDefault();drag={side,x:event.clientX,width:side==='left'?leftPanel.getBoundingClientRect().width:rightPanel.getBoundingClientRect().width};
      handle.setPointerCapture(event.pointerId);layout.classList.add('resizing');
    });
    handle.addEventListener('pointermove',event=>{if(drag?.side===side)set(side,drag.width+(event.clientX-drag.x)*(side==='left'?1:-1));});
    const end=()=>{drag=null;layout.classList.remove('resizing');};
    for(const type of ['pointerup','pointercancel','lostpointercapture'])handle.addEventListener(type,end);
    handle.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();set(side,render()[side]+(event.key==='ArrowRight'?1:-1)*(side==='left'?1:-1)*(event.shiftKey?40:10));});
    inputs[side].onchange=()=>set(side,Number(inputs[side].value));
  }
  doc.getElementById('panel-width-reset').onclick=()=>{preferred=null;try{storage?.removeItem(key);}catch{}render();};
  new win.MutationObserver(()=>render()).observe(layout,{attributes:true,attributeFilter:['class']});
  win.addEventListener('resize',()=>render());render();
}
