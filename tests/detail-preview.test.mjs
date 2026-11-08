import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../backend/app/web/app.js',import.meta.url),'utf8');
const start=source.indexOf('    const detailFiles='),end=source.indexOf("    if (job.status === 'SUCCEEDED')",start);
const block=source.slice(start,end);
function render(artifacts){
 const document={createElement:tag=>({tag,children:[],append(...items){this.children.push(...items);}})};
 const article=document.createElement('article');
 vm.runInNewContext(block,{document,metadata:article,job:{id:'job',artifacts},selectedSession:'2026-11/workspace',encodeURIComponent});
 return article;
}
test('detail comparison images open preserved files in a new tab',()=>{
 const article=render(['detail-0-before.png','detail-0-after.png']);
 const pair=article.children[0].children.find(node=>node.className==='detail-comparison');
 assert.equal(pair.children.length,2);
 for(const [i,side] of ['before','after'].entries()){
  const link=pair.children[i].children[0];assert.equal(link.tag,'a');assert.equal(link.target,'_blank');
  assert.equal(link.rel,'noopener noreferrer');assert.match(link.href,new RegExp(`detail-0-${side}\\.png\\?session=2026-11%2Fworkspace`));
  assert.equal(link.children[0].src,link.href);
 }
});
test('interrupted detail generation displays its existing before image without requesting a missing after file',()=>{
 const article=render(['detail-0-before.png']);
 const pair=article.children[0].children.find(node=>node.className==='detail-comparison');
 assert.ok(pair);assert.equal(pair.children.length,1);assert.match(pair.children[0].children[0].href,/before/);
});
