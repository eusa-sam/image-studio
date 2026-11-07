import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile,readdir} from 'node:fs/promises';

const source=await readFile(new URL('../backend/app/web/boot.js',import.meta.url),'utf8');
test('every shipped frontend file parses as a JavaScript module',async()=>{
  const root=new URL('../backend/app/web/',import.meta.url);
  for(const name of (await readdir(root)).filter(name=>/\.(m?js)$/.test(name))) {
    const code=await readFile(new URL(name,root),'utf8');
    assert.doesNotThrow(()=>new vm.SourceTextModule(code,{identifier:name}),name);
  }
});
test('startup errors remain visible and successful initialization removes the banner',async()=>{
  for(const failure of [null,'module missing','workspace failed']) {
    const status={hidden:false,setAttribute(name,value){this[name]=value;}};
    const events={};
    const context=vm.createContext({document:{getElementById(){return status;}},window:{addEventListener(name,handler){events[name]=handler;}}});
    const script=new vm.Script(source,{importModuleDynamically:async(specifier)=>{
      if(failure==='module missing')throw Error(failure);
      const module=new vm.SyntheticModule(['ready'],function(){this.setExport('ready',failure && specifier.includes('app.js')?Promise.reject(Error(failure)):Promise.resolve());},{context});
      await module.link(()=>{});await module.evaluate();return module;
    }});
    script.runInContext(context);
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(status.hidden,!failure);
    if(failure){assert.equal(status.role,'alert');assert.ok(status.textContent.includes(failure));}
    events.error({message:'later error'});
    assert.equal(status.hidden,false);assert.ok(status.textContent.includes('later error'));
    events.unhandledrejection({reason:Error('lost connection')});
    assert.ok(status.textContent.includes('lost connection'));
  }
});
