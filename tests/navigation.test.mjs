import test from 'node:test';
import assert from 'node:assert/strict';
import {modelType, setupNavigation, setupStudioTabs, renderWorkspaces} from '../backend/app/web/navigation.mjs';

test('studio tabs hide only their panel, preserve contents and support keyboard focus',()=>{
 const nodes={};let focused='';
 for(const id of ['tab-create','tab-results','studio-create','studio-results'])nodes[id]={hidden:false,content:'draft',setAttribute(k,v){this[k]=v;},focus(){focused=id;}};
 const select=setupStudioTabs({getElementById:id=>nodes[id]});
 assert.equal(nodes['studio-results'].hidden,true);select('results');
 assert.equal(nodes['studio-create'].hidden,true);assert.equal(nodes['tab-results']['aria-selected'],'true');
 assert.equal(nodes['tab-create'].tabIndex,-1);
 let prevented=false;nodes['tab-results'].onkeydown({key:'ArrowLeft',preventDefault(){prevented=true;}});
 assert.ok(prevented);assert.equal(focused,'tab-create');assert.equal(nodes['studio-create'].hidden,false);
 assert.equal(nodes['studio-create'].content,'draft');nodes['tab-create'].onclick();
});

test('model categories use folder roles, including aliases and unknown folders', () => {
  for (const [path, expected] of [
    ['checkpoint/renamed.safetensors','checkpoint'], ['checkpoints/sub/model.safetensors','checkpoint'],
    ['LoRas\\test.safetensors','lora'], ['vae/test.safetensors','vae'],
    ['text_encoders/qwen.safetensors','text_encoder'], ['clip/test.safetensors','text_encoder'],
    ['clip_vision/test.safetensors','clip_vision'], ['upscale_models/test.pth','upscaler'],
    ['detectors/face.onnx','detector'], ['unknown/checkpoint.safetensors','other'],
    ['toString/file.safetensors','other'],
  ]) assert.equal(modelType(path), expected, path);
});

function fixture() {
  const nodes = new Map();
  for (const id of ['layout','toggle-sidebar','sidebar-content','nav-workspaces','nav-models','nav-gallery','workspace-panel','models-panel','gallery-panel']) {
    const attributes = new Map(); const classes = new Set();
    nodes.set(id, {hidden:id === 'models-panel',
      setAttribute:(key,value) => attributes.set(key,value), getAttribute:key => attributes.get(key),
      classList:{toggle:(name,enabled) => enabled ? classes.add(name) : classes.delete(name), contains:name => classes.has(name)},
    });
  }
  return {getElementById:id => nodes.get(id)};
}

test('collapsed navigation can reopen directly into Models, then switch Workspace', () => {
  const doc = fixture(); const values = new Map([['image-maker.sidebar-collapsed','true']]);
  setupNavigation(doc, {getItem:key => values.get(key), setItem:(key,value) => values.set(key,value)});
  const get = id => doc.getElementById(id);
  assert.equal(get('sidebar-content').hidden,true);
  assert.equal(get('toggle-sidebar').getAttribute('aria-expanded'),'false');
  get('nav-models').onclick();
  assert.equal(get('sidebar-content').hidden,false);
  assert.equal(get('workspace-panel').hidden,true);
  assert.equal(get('models-panel').hidden,false);
  assert.equal(get('nav-models').getAttribute('aria-pressed'),'true');
  get('nav-gallery').onclick();
  assert.equal(get('gallery-panel').hidden,false);
  assert.equal(get('models-panel').hidden,true);
  get('nav-workspaces').onclick();
  assert.equal(get('gallery-panel').hidden,true);
  assert.equal(get('models-panel').hidden,true);
  assert.equal(get('workspace-panel').hidden,false);
  get('toggle-sidebar').onclick();
  assert.equal(values.get('image-maker.sidebar-collapsed'),'true');
  assert.equal(get('layout').classList.contains('sidebar-collapsed'),true);
});

test('unavailable preference storage does not break navigation', () => {
  const doc = fixture();
  setupNavigation(doc, {getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}});
  doc.getElementById('toggle-sidebar').onclick();
  doc.getElementById('toggle-sidebar').onclick();
  assert.equal(doc.getElementById('sidebar-content').hidden,false);
});


test('workspace search preserves selection, handles Korean normalization, IDs and empty matches',()=>{
 const select={value:'a',disabled:true,ownerDocument:{createElement(){return {};}},replaceChildren(...nodes){this.nodes=nodes;}};
 const status={};const records=[{ref:'a',title:'현재 작업'},{ref:'2026-11/b',title:'꽃밭 Portrait',archived:true}];
 const snapshot=JSON.stringify(records);
 renderWorkspaces(select,status,records,'a','portrait');
 assert.equal(select.value,'a');assert.equal(select.nodes.length,2);assert.equal(select.nodes[1].textContent,'[숨김] 꽃밭 Portrait');
 assert.match(status.textContent,/1개 검색/);assert.equal(select.disabled,true);
 renderWorkspaces(select,status,records,'a','nothing');
 assert.equal(select.value,'a');assert.equal(select.nodes.length,1);assert.match(status.textContent,/0개/);
 renderWorkspaces(select,status,records,'2026-11/b','꽃밭'.normalize('NFD'));
 assert.equal(select.nodes.length,1);assert.equal(select.value,'2026-11/b');
 renderWorkspaces(select,status,records,'a','2026-11/B');assert.equal(select.nodes.length,2);
 renderWorkspaces(select,status,records,'a','  ');assert.equal(select.nodes.length,2);assert.equal(select.value,'a');
 renderWorkspaces(select,status,records,'missing','');assert.equal(select.value,'2026-11/b');
 assert.equal(JSON.stringify(records),snapshot);
 renderWorkspaces(select,status,[],'a','');assert.equal(select.value,'');assert.equal(select.nodes.length,0);
});

test('workspace cards select safely, show recent changes first and support keyboard focus',()=>{
 let focused=null,changes=0;
 const doc={createElement(){return {setAttribute(k,v){this[k]=v;},append(...children){this.children=children;},focus(){focused=this;}};}};
 const select={value:'a',disabled:false,ownerDocument:doc,replaceChildren(...nodes){this.nodes=nodes;},onchange(){changes++;}};
 const list={ownerDocument:doc,replaceChildren(...nodes){this.nodes=nodes;this.children=nodes;}};
 const records=[{ref:'a',title:'<img onerror=bad>',created_at:'2026-11-01T00:00:00Z'},
                {ref:'b',title:'최근 작업',updated_at:'2026-11-07T00:00:00Z',archived:true}];
 renderWorkspaces(select,{},records,'a','',list);
 const [recent,current]=list.nodes;
 assert.equal(recent.value,'b');assert.equal(current.children[0].textContent,'<img onerror=bad>');
 assert.equal(current['aria-current'],'true');assert.equal(current.tabIndex,0);
 assert.match(recent.children[1].textContent,/숨김/);
 current.onkeydown({key:'Home',preventDefault(){}});
 assert.equal(focused,recent);assert.equal(select.value,'a');assert.equal(changes,0);
 recent.onclick();assert.equal(select.value,'b');assert.equal(changes,1);
 assert.equal(recent['aria-current'],'true');assert.equal(current['aria-current'],'false');
 recent.onclick();assert.equal(changes,1);
 select.disabled=true;current.onclick();assert.equal(select.value,'b');assert.equal(changes,1);
 renderWorkspaces(select,{},records,'b','not found',list);
 assert.equal(list.nodes.length,1);assert.equal(list.nodes[0].value,'b');
 select.disabled=false;const status={};
 renderWorkspaces(select,status,records,'a','최근',list);
 list.nodes.find(item=>item.value==='b').onclick();
 assert.equal(list.nodes.length,1);assert.equal(focused,list.nodes[0]);
 assert.equal(select.value,'b');assert.doesNotMatch(status.textContent,/유지/);
 renderWorkspaces(select,{},[],'b','',list);assert.equal(list.nodes.length,0);
});
