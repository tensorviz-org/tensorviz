import test from 'node:test';
import assert from 'node:assert/strict';
import {githubPublicationJournal} from '../../scripts/github-publication-journal.mjs';
const identity={version:'1.0.0',sourceRevision:'a'.repeat(40),vsixSha256:'b'.repeat(64)};
function fixture(){
 const state={release:null,assets:[],uploads:0,corrupt:false,fail:false};
 const transport={repository:async()=>({full_name:'tensorviz-org/tensorviz',private:false}),
  release:async()=>state.release,create:async body=>(state.release={...body,id:1}),assets:async()=>state.assets,
  upload:async(tag,name,bytes)=>{state.uploads++;if(state.fail)throw Error('upload unavailable');state.assets.push({id:state.assets.length+1,name,bytes});},
  download:async id=>state.corrupt?Buffer.from('corrupt'):state.assets.find(a=>a.id===id).bytes};
 return {state,transport,journal:githubPublicationJournal(identity,{transport})};
}
test('a verified durable intent exists before returning permission to submit',async()=>{
 const f=fixture();await f.journal.intent('vscode-marketplace',identity);
 assert.equal(f.state.release.draft,true);assert.equal(f.state.release.target_commitish,identity.sourceRevision);
 assert.equal(JSON.parse(f.state.assets[0].bytes).state,'submission-started');
 await f.journal.intent('open-vsx',identity);assert.equal(f.state.uploads,2);
});
test('a new process finds an earlier uncertain intent and refuses another submission',async()=>{
 const f=fixture();await f.journal.intent('vscode-marketplace',identity);
 const retried=githubPublicationJournal(identity,{transport:f.transport});
 await assert.rejects(retried.intent('vscode-marketplace',identity),/pending or uncertain/);assert.equal(f.state.uploads,1);
});
for(const [name,mutate] of [
 ['corrupt readback',f=>f.state.corrupt=true],['unavailable storage',f=>f.state.fail=true],
 ['published journal',f=>f.state.release={draft:false}],['wrong source',f=>f.state.release={draft:true,tag_name:'wrong',target_commitish:'c'.repeat(40)}],
])test(`submission is blocked by ${name}`,async()=>{const f=fixture();mutate(f);await assert.rejects(f.journal.intent('docs',identity));});
test('only known channels and exact source identity can create an intent',async()=>{
 const f=fixture();await assert.rejects(f.journal.intent('../escape',identity));
 await assert.rejects(f.journal.intent('docs',{...identity,sourceRevision:'c'.repeat(40)}));assert.equal(f.state.uploads,0);
});
