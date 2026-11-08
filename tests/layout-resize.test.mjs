import test from 'node:test';
import assert from 'node:assert/strict';
import {fitPanels} from '../backend/app/web/layout.mjs';

test('dragging either edge preserves the opposite panel and a usable editor',()=>{
  assert.deepEqual(fitPanels(1280,900,390,false,'left'),{left:530,right:390});
  assert.deepEqual(fitPanels(1280,250,900,false,'right'),{left:250,right:670});
  assert.deepEqual(fitPanels(1280,10,10),{left:220,right:320});
  assert.deepEqual(fitPanels(1051,600,700),{left:220,right:471});
  assert.deepEqual(fitPanels(1280,500,800,true,'right'),{left:64,right:800});
  assert.deepEqual(fitPanels(1280,NaN,Infinity),{left:250,right:390});
});
