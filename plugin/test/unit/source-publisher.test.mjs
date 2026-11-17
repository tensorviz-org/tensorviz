import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {publishSourceAssets} from '../../scripts/publish-source-release.mjs';
const sha = b=>createHash('sha256').update(b).digest('hex');
const sourceRevision='a'.repeat(40);
function fixture() {
 const bytes=Buffer.from('verified artifact');
 const input={version:'1.0.0',sourceRevision,assets:{'release.json':{bytes,sha256:sha(bytes)}}};
 const state={release:null,assets:[],uploads:0,published:0,tag:sourceRevision,private:false};
 const api={repository:async()=>({full_name:'tensorviz-org/tensorviz',private:state.private,default_branch:'main'}),tag:async()=>({object:{type:'commit',sha:state.tag}}),release:async()=>state.release,create:async body=>(state.release={...body,id:1}),assets:async()=>state.assets,download:async id=>state.assets.find(a=>a.id===id).bytes,upload:async(tag,name,bytes)=>{state.uploads++;state.assets.push({id:1,name,bytes});},publicDownload:async(tag,name)=>state.assets.find(a=>a.name===name).bytes,publish:async()=>{state.published++;state.release.draft=false;}};
 return {input,state,api};
}
test('publishes verified source assets once; immutable retry does no writes',async()=>{
 const {input,state,api}=fixture();await publishSourceAssets(input,api); await publishSourceAssets(input,api);
 assert.equal(state.uploads,1);assert.equal(state.published,1);
});
for(const [name,mutate] of [
 ['private repository',f=>f.state.private=true],
 ['wrong source tag',f=>f.state.tag='b'.repeat(40)],
 ['prelaunch version',f=>f.input.version='0.1.47'],
 ['path traversal',f=>f.input.assets['../evil']=f.input.assets['release.json']],
 ['checksum mismatch',f=>f.input.assets['release.json'].sha256='0'.repeat(64)],
 ['different release source',f=>f.state.release={tag_name:'v1.0.0',target_commitish:'b'.repeat(40),draft:true,id:1}],
 ['published missing asset',f=>f.state.release={tag_name:'v1.0.0',target_commitish:sourceRevision,draft:false,id:1}],
 ['extra hosted asset',f=>f.state.assets=[{id:2,name:'unexpected',bytes:Buffer.from('x')}]],
 ['mismatched hosted bytes',f=>f.state.assets=[{id:2,name:'release.json',bytes:Buffer.from('x')}]],
]) test(`refuses ${name}`,async()=>{const f=fixture();mutate(f);await assert.rejects(publishSourceAssets(f.input,f.api));assert.equal(f.state.published,0);assert.equal(f.state.uploads,0);});
test('upload corruption never exposes draft',async()=>{const f=fixture();f.api.upload=async(t,name)=>f.state.assets.push({id:1,name,bytes:Buffer.from('corrupt')});await assert.rejects(publishSourceAssets(f.input,f.api));assert.equal(f.state.published,0);});
test('concurrent moved tag blocks publishing',async()=>{const f=fixture();const upload=f.api.upload;f.api.upload=async(...a)=>{await upload(...a);f.state.tag='b'.repeat(40);};await assert.rejects(publishSourceAssets(f.input,f.api));assert.equal(f.state.published,0);});

import {githubTransport} from '../../scripts/publish-source-release.mjs';
test('transport discovers existing draft on tags API404 and pins GitHub hostname',()=>{
 const calls=[];const command=(tool,args)=>{calls.push(args);if(args.some(a=>a==='releases/tags/v1.0.0'||a.endsWith('/releases/tags/v1.0.0'))){const e=Error('404');e.stderr='HTTP 404';throw e;}return JSON.stringify([[{id:7,tag_name:'v1.0.0',draft:true}]]);};
 const found=githubTransport('/tmp',{command}).release('v1.0.0');assert.equal(found.id,7);assert.equal(calls.length,2);assert.ok(calls.every(a=>a[1]==='--hostname'&&a[2]==='github.com'));
});
test('transport rejects ambiguous duplicate drafts',()=>{
 const command=(tool,args)=>{if(args.some(a=>a.endsWith('/releases/tags/v1.0.0'))){const e=Error();e.stderr='HTTP 404';throw e;}return JSON.stringify([[{tag_name:'v1.0.0'},{tag_name:'v1.0.0'}]]);};
 assert.throws(()=>githubTransport('/tmp',{command}).release('v1.0.0'),/Duplicate/);
});
test('transport does not treat authentication error as absence',()=>{const command=()=>{const e=Error('denied');e.stderr='HTTP 403';throw e;};assert.throws(()=>githubTransport('/tmp',{command}).release('v1.0.0'),/denied/);});

test('public bytes must match before reporting success',async()=>{const f=fixture();f.api.publicDownload=async()=>Buffer.from('bad');await assert.rejects(publishSourceAssets(f.input,f.api),/Public asset verification/);assert.equal(f.state.published,1);});
test('public unavailable is retryable without overwriting published bytes',async()=>{const f=fixture(),download=f.api.publicDownload;f.api.publicDownload=async()=>{throw Error('unavailable');};await assert.rejects(publishSourceAssets(f.input,f.api));f.api.publicDownload=download;await publishSourceAssets(f.input,f.api);assert.equal(f.state.uploads,1);assert.equal(f.state.published,1);});

test('obsolete callback-authorized candidate publisher is not exported',async()=>{
 const exports=await import('../../scripts/publish-source-release.mjs');
 assert.equal(Object.hasOwn(exports,'publishSourceCandidate'),false);
});
