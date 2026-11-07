import test from 'node:test';
import assert from 'node:assert/strict';
import {promptComparison} from '../backend/app/web/chat.mjs';

const doc={createElement(tag){return {tag,children:[],append(...nodes){this.children.push(...nodes);}};}};
const body=node=>node.textContent??node.children.map(body).join('');
const sections=element=>element.children[1].children;

test('comparison preserves exact multiline text and highlights edits without interpreting HTML',()=>{
  const before='one woman\nblue dress, <img onerror=evil()>',after='one woman\ngreen dress, <img onerror=evil()>';
  const result=promptComparison(doc,before,after,'Positive');
  assert.equal(result.children[0].textContent,'Positive · 변경 있음');
  const [old,next]=sections(result).map(row=>row.children[1]);
  assert.equal(body(old),before);assert.equal(body(next),after);
  assert.equal(old.children[1].tag,'del');assert.equal(next.children[1].tag,'ins');
  assert.equal(next.children[1].textContent,'green');
  assert.equal(next.children[2].textContent,' dress, <img onerror=evil()>');
});

test('insertion, deletion, empty and unknown snapshots remain distinguishable',()=>{
  for(const [before,after] of [['','new'],['old',''],['ab','axb'],['axb','ab'],['😀 old','😃 new']]){
    const result=promptComparison(doc,before,after,'Positive');
    const [old,next]=sections(result).map(row=>row.children[1]);
    assert.equal(body(old),before||'(비어 있음)');assert.equal(body(next),after||'(비어 있음)');
    if(before.startsWith('😀'))assert.ok(old.children[1].textContent.startsWith('😀'));
  }
  const same=promptComparison(doc,'same','same','Negative');
  assert.equal(sections(same).length,1);assert.match(same.children[0].textContent,/변경 없음/);
  const unknown=promptComparison(doc,undefined,'new','Positive');
  assert.equal(sections(unknown).length,1);assert.match(unknown.children[0].textContent,/제안 원문/);
});
