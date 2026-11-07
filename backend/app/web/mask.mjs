export function rectanglePixels(values,width,height) {
  const [x,y,w,h]=values.map(Number);
  if(![x,y,w,h].every(Number.isFinite)||x<0||y<0||w<=0||h<=0||x+w>100||y+h>100)throw Error('사각형은 이미지 안의 0~100% 범위로 입력하세요.');
  return [Math.floor(x*width/100),Math.floor(y*height/100),Math.max(1,Math.round(w*width/100)),Math.max(1,Math.round(h*height/100))];
}

export function setupMask({api,getContext,changed}) {
  const $=id=>document.getElementById(id),canvas=$('mask-canvas'),ctx=canvas.getContext('2d');
  const selected=document.createElement('canvas'),paint=selected.getContext('2d');
  let base=null,key='',version=0,active=null,last=null;
  const currentKey=()=>{const c=getContext();return c.session+'/'+c.settings.source_job+'/'+(c.settings.source_artifact||'image.png');};
  function valid(){if($('mask-save').disabled)throw Error('저장 작업이 끝난 뒤 수정하세요.');if(!base||key!==currentKey())throw Error('현재 입력 이미지를 먼저 여세요.');}
  function redraw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(base)ctx.drawImage(base,0,0);ctx.globalAlpha=.55;ctx.drawImage(selected,0,0);ctx.globalAlpha=1;}
  function dirty(){version++;$('mask_id').value='';$('mask-status').textContent='수정 영역을 변경했습니다. 마스크를 저장하세요.';changed();redraw();}
  async function image(url){const result=new Image();result.src=url;await result.decode();return result;}
  $('mask-load').onclick=async()=>{
    const c=getContext(),requested=currentKey(),loadVersion=++version;
    try {
      if(!c.settings.source_job)throw Error('입력 이미지를 선택하세요.');
      const source=await image(`/api/jobs/${c.settings.source_job}/${c.settings.source_artifact||'image.png'}?session=${encodeURIComponent(c.session)}`);
      const saved=c.settings.mask_id?await image(`/api/masks/${c.settings.mask_id}?session=${encodeURIComponent(c.session)}`):null;
      if(requested!==currentKey()||loadVersion!==version||c.prompt_id!==getContext().prompt_id||c.settings.mask_id!==getContext().settings.mask_id)return;
      base=source;key=requested;canvas.width=selected.width=source.naturalWidth;canvas.height=selected.height=source.naturalHeight;
      if(saved){paint.drawImage(saved,0,0);const pixels=paint.getImageData(0,0,selected.width,selected.height);for(let i=0;i<pixels.data.length;i+=4){const on=pixels.data[i]>=128;pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=255;pixels.data[i+3]=on?255:0;}paint.putImageData(pixels,0,0);}
      redraw();$('mask-status').textContent=saved?'저장된 마스크를 열었습니다.':'수정할 영역을 칠하세요.';
    }catch(error){$('mask-status').textContent=error.message;}
  };
  const position=event=>{const bounds=canvas.getBoundingClientRect();return [(event.clientX-bounds.left)*canvas.width/bounds.width,(event.clientY-bounds.top)*canvas.height/bounds.height];};
  function stroke(event){const point=position(event);paint.globalCompositeOperation=$('mask-erase').checked?'destination-out':'source-over';paint.strokeStyle='white';paint.lineWidth=Number($('mask-brush').value);paint.lineCap='round';paint.beginPath();paint.moveTo(...(last||point));paint.lineTo(point[0]+.01,point[1]+.01);paint.stroke();paint.globalCompositeOperation='source-over';last=point;dirty();}
  canvas.onpointerdown=event=>{try{valid();active=event.pointerId;last=null;canvas.setPointerCapture(active);stroke(event);}catch(error){$('mask-status').textContent=error.message;}};
  canvas.onpointermove=event=>{if(active===event.pointerId&&key===currentKey())stroke(event);};
  canvas.onpointerup=canvas.onpointercancel=()=>{active=null;last=null;};
  function edit(action){try{valid();action();dirty();}catch(error){$('mask-status').textContent=error.message;}}
  $('mask-clear').onclick=()=>edit(()=>paint.clearRect(0,0,selected.width,selected.height));
  $('mask-all').onclick=()=>edit(()=>{paint.fillStyle='white';paint.fillRect(0,0,selected.width,selected.height);});
  $('mask-rectangle').onclick=()=>edit(()=>{const rect=rectanglePixels(['mask-x','mask-y','mask-w','mask-h'].map(id=>$(id).value),selected.width,selected.height);paint.fillStyle='white';paint.fillRect(...rect);});
  $('mask-save').onclick=async()=>{
    try {valid();const c=getContext(),requestKey=key,revision=version;
      const output=document.createElement('canvas');output.width=selected.width;output.height=selected.height;const out=output.getContext('2d');out.fillStyle='black';out.fillRect(0,0,output.width,output.height);out.drawImage(selected,0,0);
      const result=await api('/api/masks',{session:c.session,source_job:c.settings.source_job,source_artifact:c.settings.source_artifact||'image.png',png:output.toDataURL('image/png').split(',')[1]});
      if(requestKey!==currentKey()||revision!==version||c.prompt_id!==getContext().prompt_id)return;
      $('mask_id').value=result.id;$('mask-status').textContent='마스크를 저장했습니다. 이제 부분 수정을 생성할 수 있습니다.';changed();
    }catch(error){$('mask-status').textContent=error.message;}
  };
  for(const id of ['source_job','source_artifact'])$(id).addEventListener('change',()=>{version++;$('mask_id').value='';changed();});
}
