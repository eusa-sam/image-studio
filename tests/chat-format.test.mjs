import test from 'node:test';
import assert from 'node:assert/strict';
import {renderReply} from '../backend/app/web/chat.mjs';
const doc={createElement(tag){return {tag,children:[],attrs:{},textContent:'',append(...nodes){this.children.push(...nodes);},replaceChildren(...nodes){this.children=nodes;},setAttribute(k,v){this.attrs[k]=v;}};}};
const text=node=>node.textContent+node.children.map(text).join('');
test('assistant replies render paragraphs, lists, emphasis and code without executing HTML',()=>{
 const root=doc.createElement('div');
 renderReply(doc,root,'## Changes\n\n- **Dress**: `red`\n- <img src=x onerror=alert(1)>\n\n3. Keep hands\n4. Keep face\n\nPlain\nline');
 assert.deepEqual(root.children.map(n=>n.tag),['h4','ul','ol','p']);
 assert.equal(root.children[1].children.length,2);
 assert.equal(root.children[2].attrs.start,'3');
 assert.equal(text(root.children[3]),'Plain\nline');
 assert.ok(text(root).includes('<img src=x onerror=alert(1)>'));
 const all=n=>[n,...n.children.flatMap(all)];
 assert.equal(all(root).some(n=>['img','script','a'].includes(n.tag)),false);
 assert.ok(all(root).some(n=>n.tag==='strong'&&text(n)==='Dress'));
 assert.ok(all(root).some(n=>n.tag==='code'&&text(n)==='red'));
});
test('stream updates replace partial fences and preserve literal code',()=>{
 const root=doc.createElement('div');
 renderReply(doc,root,'```json\n{"value":"**literal**"}');
 assert.equal(root.children[0].tag,'pre');assert.equal(text(root),' {"value":"**literal**"}'.trim());
 renderReply(doc,root,'```json\n{"value":"**literal**"}\n```\n\nDone **now**');
 assert.deepEqual(root.children.map(n=>n.tag),['pre','p']);
 assert.equal(text(root.children[0]),'{"value":"**literal**"}');
 renderReply(doc,root,'');assert.equal(root.children.length,0);
});
