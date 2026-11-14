import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {canonicalCatalog, catalogPrefix, catalogPaths} from '../../scripts/canonical-catalog.mjs';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
function fixture() {
  const entries = {'README.md': Buffer.from('distribution readme'), 'plugins/tensorviz/plugin.json': Buffer.from('{}'),
    'plugins/tensorviz/scripts/launch.sh': Buffer.from('#!/bin/sh\n'), 'plugins/tensorviz/runtime-lock.json': Buffer.from('{"url":"https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.1.46/archive"}')};
  for (const name of catalogPaths) entries[name] = Buffer.from(JSON.stringify({name:'tensorviz-standalone',plugins:[{name:'tensorviz',source:name.startsWith('.agents') ? {source:'local',path:'./plugins/tensorviz'} : './plugins/tensorviz'}]}));
  return {entries, publication:{formatVersion:2,version:'0.1.46',sourceRevision:'a'.repeat(40),files:Object.fromEntries(Object.entries(entries).map(([name, bytes])=>[name,hash(bytes)]))}};
}
function update(f, name, bytes) { f.entries[name]=Buffer.from(bytes); f.publication.files[name]=hash(f.entries[name]); }
test('preserves export, history URLs, identity and source-owned paths',()=>{
  const f=fixture(), before=structuredClone(f.publication), result=canonicalCatalog(f.publication,f.entries);
  assert.deepEqual(f.publication,before);
  assert.equal(result['README.md'],undefined); assert.equal(result['plugins/tensorviz/plugin.json'],undefined);
  for(const [name, bytes] of Object.entries(f.entries)) assert.deepEqual(result[`${catalogPrefix}/${name}`].bytes,bytes);
  assert.equal(result[`${catalogPrefix}/plugins/tensorviz/scripts/launch.sh`].mode,'100755');
  for(const name of catalogPaths) {const c=JSON.parse(result[name].bytes);assert.equal(c.name,'tensorviz-standalone');assert.equal(c.plugins[0].name,'tensorviz');assert.equal(typeof c.plugins[0].source==='string'?c.plugins[0].source:c.plugins[0].source.path,`./${catalogPrefix}/plugins/tensorviz`);}
  assert.ok(Object.keys(result).every(name=>catalogPaths.includes(name)||name.startsWith(catalogPrefix+'/')));
});
for(const name of ['../README.md','/absolute','a//b','a/./b','.git/config','a/.GIT/config','a\\b','a:stream']) test(`rejects unsafe path ${name}`,()=>{const f=fixture();update(f,name,'x');assert.throws(()=>canonicalCatalog(f.publication,f.entries));});
test('rejects checksum, provenance, inventory and case collisions',()=>{
  for(const mutate of [f=>f.entries['README.md']=Buffer.from('changed'),f=>f.publication.sourceRevision='main',f=>delete f.entries['README.md'],f=>update(f,'readme.md','x')]) {const f=fixture();mutate(f);assert.throws(()=>canonicalCatalog(f.publication,f.entries));}
});
for(const source of ['../outside','https://github.com/other/repo','./missing',{source:'github',path:'./plugins/tensorviz'}]) test(`rejects unverified catalog source ${JSON.stringify(source)}`,()=>{const f=fixture();update(f,catalogPaths[0],JSON.stringify({plugins:[{name:'tensorviz',source}]}));assert.throws(()=>canonicalCatalog(f.publication,f.entries));});
test('deterministic manifest, no aliasing and no deletion instructions',()=>{const f=fixture(),a=canonicalCatalog(f.publication,f.entries),b=canonicalCatalog(f.publication,f.entries);assert.deepEqual(a,b);f.entries['README.md'].fill(0);assert.notDeepEqual(a[`${catalogPrefix}/README.md`].bytes,f.entries['README.md']);assert.ok(Object.values(a).every(v=>Buffer.isBuffer(v.bytes)));});
