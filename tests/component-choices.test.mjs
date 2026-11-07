import test from 'node:test';
import assert from 'node:assert/strict';
import {componentCompatibility,componentChoices,renderComponentSelect} from '../backend/app/web/pipeline.mjs';

const checkpoint=family=>({available:true,family});
const component=(id,kind,family,extra={})=>({id,name:id,component:kind,profile:{family,...extra}});

test('rejected saved components retain the scanner reason and remain disabled',()=>{
  const model={id:'rejected',name:'style',available:false,reason:'검증 거절 기록이 있는 파일입니다.'};
  const base=checkpoint('sdxl_candidate');
  assert.equal(componentCompatibility(model,base,'model'),model.reason);
  assert.equal(componentCompatibility(model,base,'lora0'),model.reason);
  const rows=componentChoices('lora0',base,[model],model.id);
  assert.deepEqual(rows,[{id:model.id,name:model.name,reason:model.reason}]);
});

test('component eligibility follows image family, encoder slots and detector roles',()=>{
  for(const [raw,family] of [['sd1_candidate','sd1'],['sd1_inpaint_candidate','sd1'],['sd2_candidate','sd2'],['sdxl_candidate','sdxl'],['anima','anima'],['krea2','krea2']]) {
    const base=checkpoint(raw);
    assert.equal(componentCompatibility(component('style','lora',family),base,'lora0'),null);
    assert.ok(componentCompatibility(component('wrong','lora','other'),base,'lora0'));
    const clip=component('encoder','text_encoder',family,{slot:1});
    assert.equal(!!componentCompatibility(clip,base,'clip'),family==='anima');
    assert.equal(!!componentCompatibility({...clip,profile:{family,slot:2}},base,'clip2'),family!=='sdxl');
    assert.equal(componentCompatibility(component('up','upscaler','spandrel'),base,'upscaler'),null);
  }
  const face=component('face','detector','yolov8',{target:'face'});
  assert.equal(componentCompatibility(face,checkpoint('krea2'),'face_detector'),null);
  assert.ok(componentCompatibility(face,checkpoint('sdxl_candidate'),'hand_detector'));
  assert.ok(componentCompatibility(face,null,'face_detector'));
});

test('native select groups preserve an incompatible or missing stored choice',()=>{
  const doc={createElement(tag){return {tag,children:[],append(node){this.children.push(node);}};}};
  const select={ownerDocument:doc,value:'xl',replaceChildren(...children){this.children=children;this.value='';}};
  const models=[component('xl','lora','sdxl'),component('krea','lora','krea2')];
  renderComponentSelect(select,'lora0',checkpoint('krea2'),models,'사용 안 함');
  assert.equal(select.value,'xl');
  const groups=select.children.slice(1);
  assert.equal(groups[0].children[0].value,'krea');
  assert.equal(groups[0].children[0].disabled,false);
  assert.equal(groups[1].children[0].value,'xl');
  assert.equal(groups[1].children[0].disabled,true);
  assert.match(groups[1].children[0].textContent,/SDXL 전용/);
  renderComponentSelect(select,'lora0',checkpoint('sdxl_candidate'),models,'사용 안 함');
  assert.equal(select.children[1].children[0].value,'xl');
  assert.equal(select.children[1].children[0].disabled,false);
  select.value='removed';renderComponentSelect(select,'lora0',checkpoint('sdxl_candidate'),models,'사용 안 함');
  assert.equal(select.value,'removed');
  assert.equal(componentChoices('lora0',checkpoint('sdxl_candidate'),models,'removed').at(-1).reason,'누락 또는 이 슬롯에서 미지원');
  assert.equal(models.length,2);
});
