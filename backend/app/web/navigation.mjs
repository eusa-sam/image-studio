export const modelTypes = [
  ['checkpoint', 'Checkpoint'], ['lora', 'LoRA'], ['embedding','임베딩'], ['vae', 'VAE'],
  ['text_encoder', 'CLIP / 텍스트 인코더'], ['clip_vision', 'CLIP Vision'], ['ip_adapter', 'IP-Adapter'],
  ['upscaler', '업스케일러'], ['detector', '디테일러 / 검출기'], ['other', '기타'],
];

// Folder categories describe storage roles, not verified model compatibility.
export function modelType(path) {
  const folder = path.replaceAll('\\', '/').split('/')[0].toLowerCase();
  const aliases = {
    checkpoint:'checkpoint', checkpoints:'checkpoint', diffusion_models:'checkpoint', unet:'checkpoint',
    lora:'lora', loras:'lora', embeddings:'embedding', vae:'vae', vaes:'vae', clip:'text_encoder',
    text_encoder:'text_encoder', text_encoders:'text_encoder', clip_vision:'clip_vision', ip_adapter:'ip_adapter',
    upscale_models:'upscaler', upscalers:'upscaler', detector:'detector', detectors:'detector',
    ultralytics:'detector', detection_models:'detector',
  };
  return Object.hasOwn(aliases, folder) ? aliases[folder] : 'other';
}

export function setupNavigation(doc, storage) {
  const layout = doc.getElementById('layout');
  const toggle = doc.getElementById('toggle-sidebar');
  function collapse(value) {
    layout.classList.toggle('sidebar-collapsed', value);
    doc.getElementById('sidebar-content').hidden = value;
    toggle.setAttribute('aria-expanded', String(!value));
    toggle.setAttribute('aria-label', value ? '메뉴 펼치기' : '메뉴 접기');
    toggle.title = value ? '메뉴 펼치기' : '메뉴 접기';
    try { storage?.setItem('image-maker.sidebar-collapsed', String(value)); } catch { /* Storage may be disabled. */ }
  }
  let initial = false;
  try { initial = storage?.getItem('image-maker.sidebar-collapsed') === 'true'; } catch { /* Use expanded menu. */ }
  collapse(initial);
  toggle.onclick = () => collapse(toggle.getAttribute('aria-expanded') === 'true');
  for (const [name, panel] of [['workspaces','workspace-panel'], ['models','models-panel'], ['gallery','gallery-panel']]) {
    doc.getElementById('nav-' + name).onclick = () => {
      collapse(false);
      for (const [other, target] of [['workspaces','workspace-panel'], ['models','models-panel'], ['gallery','gallery-panel']]) {
        doc.getElementById('nav-' + other).setAttribute('aria-pressed', String(other === name));
        doc.getElementById(target).hidden = target !== panel;
      }
    };
  }
}
export function setupStudioTabs(doc) {
  const names=['create','results'];
  function select(name, focus=false) {
    for(const item of names) {
      const tab=doc.getElementById('tab-'+item), selected=item===name;
      tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;
      doc.getElementById('studio-'+item).hidden=!selected;
      if(selected&&focus)tab.focus();
    }
  }
  for(const name of names) {
    const tab=doc.getElementById('tab-'+name);
    tab.onclick=()=>select(name);
    tab.onkeydown=event=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
      event.preventDefault();select(event.key==='Home'?'create':event.key==='End'?'results':names.find(n=>n!==name),true);
    };
  }
  select('create');return select;
}


export function renderWorkspaces(select, status, records, selected, query='', list=null) {
  const term=query.trim().normalize('NFC').toLocaleLowerCase();
  const matches=records.filter(row=>(row.title+' '+row.ref).normalize('NFC').toLocaleLowerCase().includes(term));
  const current=records.find(row=>row.ref===selected) || records.at(-1);
  const shown=[...matches];
  if(current&&!shown.some(row=>row.ref===current.ref))shown.unshift(current);
  const nodes=shown.map(row=>{
    const node=select.ownerDocument.createElement('option');
    node.value=row.ref;node.textContent=(row.archived?'[숨김] ':'')+row.title;
    return node;
  });
  select.replaceChildren(...nodes);
  select.value=current?.ref || '';
  status.textContent=term ? `${matches.length}개 검색됨`+(current&&!matches.includes(current)?' · 현재 Workspace는 목록에 유지합니다.':'') : `${records.length}개 Workspace`;
  if(list) {
    const ordered=[...shown].sort((a,b)=>(b.updated_at||b.created_at||'').localeCompare(a.updated_at||a.created_at||''));
    const buttons=ordered.map(row=>{
      const button=list.ownerDocument.createElement('button');
      button.type='button';button.className='workspace-card';button.value=row.ref;
      button.setAttribute('aria-current',String(row.ref===select.value));button.tabIndex=row.ref===select.value?0:-1;
      const title=list.ownerDocument.createElement('strong');title.textContent=row.title;
      const detail=list.ownerDocument.createElement('span');
      const date=new Date(row.updated_at||row.created_at);
      detail.textContent=(row.archived?'숨김 · ':'')+(Number.isNaN(date.getTime())?row.ref.split('/')[0]:date.toLocaleDateString('ko-KR',{year:'numeric',month:'short',day:'numeric'}));
      button.append(title,detail);button.title=row.title+' · '+row.ref;
      button.onclick=()=>{
        if(select.disabled||select.value===row.ref)return;
        select.value=row.ref;
        for(const item of buttons){item.setAttribute('aria-current',String(item===button));item.tabIndex=item===button?0:-1;}
        if(term&&current&&!matches.includes(current)) {
          renderWorkspaces(select,status,records,row.ref,query,list);
          Array.from(list.children).find(item=>item.value===row.ref)?.focus();
        }
        select.onchange?.();
      };
      button.onkeydown=event=>{
        if(select.disabled||!['ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
        event.preventDefault();
        const index=buttons.indexOf(button);
        const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length;
        for(const item of buttons)item.tabIndex=-1;
        buttons[next].tabIndex=0;buttons[next].focus();
      };
      return button;
    });
    list.replaceChildren(...buttons);
  }
}
