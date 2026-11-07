import test from 'node:test';
import assert from 'node:assert/strict';
import {setupAppSettings} from '../backend/app/web/settings.mjs';

test('failed reload cannot submit stale settings and reopening recovers',async()=>{
 const nodes=new Map();const doc={getElementById(id){if(!nodes.has(id))nodes.set(id,{value:'',addEventListener(){},showModal(){},close(){}});return nodes.get(id);}};
 const $=id=>doc.getElementById(id);
 let fail=false,posts=0,revision=1;
 setupAppSettings({doc,onSaved:async()=>{},api:async(path,body)=>{
  if(body){posts++;return {...body,revision:body.revision+1};}
  if(fail)throw Error('offline');
  return {revision,model_path:'D:/models',output_path:'D:/outputs',ollama_url:'http://localhost:11434',ollama_endpoint:'/api'};
 }});
 await $('nav-settings').onclick();
 fail=true;await $('nav-settings').onclick();
 await $('application-settings-form').onsubmit({preventDefault(){}});
 await $('settings-test').onclick();
 assert.equal(posts,0);
 assert.equal($('application-settings-fields').disabled,true);
 assert.equal($('settings-close').disabled,false);
 assert.equal($('setting-model_path').value,'');
 assert.match($('application-settings-status').textContent,/offline/);
 fail=false;revision=3;await $('nav-settings').onclick();
 assert.equal($('application-settings-fields').disabled,false);
 await $('application-settings-form').onsubmit({preventDefault(){}});
 assert.equal(posts,1);
});

test('settings dialog saves exact paths, applies only after success, and keeps failed input',async()=>{
 const nodes=new Map();const doc={getElementById(id){if(!nodes.has(id))nodes.set(id,{value:'',addEventListener(){},showModal(){this.open=true;},close(){this.open=false;}});return nodes.get(id);}};
 const $=id=>doc.getElementById(id);
 const initial={schema_version:1,revision:0,model_path:'D:/models',output_path:'D:/outputs',ollama_url:'http://127.0.0.1:11434',ollama_endpoint:'/api'};
 let saved=0,fail=false,posted;
 setupAppSettings({doc,onSaved:async()=>saved++,api:async(path,body)=>{
  if(!body)return initial;
  if(fail)throw Error('busy');posted=body;return {...body,revision:body.revision+1};
 }});
 await $('nav-settings').onclick();assert.equal($('application-settings').open,true);
 $('setting-model_path').value='D:/my models';fail=true;
 await $('application-settings-form').onsubmit({preventDefault(){}});
 assert.equal(saved,0);assert.equal($('setting-model_path').value,'D:/my models');assert.equal($('application-settings-status').textContent,'busy');
 fail=false;await $('application-settings-form').onsubmit({preventDefault(){}});
 assert.equal(saved,1);assert.equal(posted.model_path,'D:/my models');assert.equal(posted.revision,0);
 assert.equal($('application-settings-fields').disabled,false);
 $('settings-close').onclick();assert.equal($('application-settings').open,false);
});
