import test from 'node:test';
import assert from 'node:assert/strict';
import {releaseChannels} from '../../scripts/release-channels.mjs';

const identity={version:'1.0.0',sourceRevision:'a'.repeat(40),vsixSha256:'b'.repeat(64),docsSha256:'c'.repeat(64)};
const receipt={...identity,verified:true};
function fixture(inspect) {
  const calls={inspect:0,publish:0,sleeps:[]};
  const channels=Object.fromEntries(['github','vscode-marketplace','open-vsx','docs'].map(name=>[name,{inspect:async()=>receipt,publish:async()=>assert.fail(`Unexpected ${name} publication`)}]));
  channels['open-vsx']={inspect:async()=>inspect(++calls.inspect),publish:async()=>{calls.publish++;}};
  const options={sleep:async milliseconds=>{calls.sleeps.push(milliseconds);}};
  return {channels,calls,options};
}

test('successful upload polls transient public absence without another publication',async()=>{
  const f=fixture(attempt=>attempt<4?null:receipt);
  const result=await releaseChannels(identity,f.channels,f.options);
  assert.equal(result.complete,true);
  assert.equal(f.calls.publish,1);
  assert.equal(f.calls.inspect,4);
  assert.deepEqual(f.calls.sleeps,[5000,5000]);
});

test('public absence times out after thirteen verification reads and sixty seconds',async()=>{
  const f=fixture(()=>null);
  const result=await releaseChannels(identity,f.channels,f.options);
  assert.equal(result.complete,false);
  assert.equal(result.channels['open-vsx'].status,'blocked');
  assert.equal(result.channels.docs.status,'verified');
  assert.equal(f.calls.publish,1);
  assert.equal(f.calls.inspect,14); // Initial inspection plus thirteen post-upload reads.
  assert.equal(f.calls.sleeps.length,12);
  assert.equal(f.calls.sleeps.reduce((sum,value)=>sum+value,0),60000);
});

test('conflicting verified bytes stop polling immediately after publication',async()=>{
  const f=fixture(attempt=>attempt===1?null:{...receipt,vsixSha256:'d'.repeat(64)});
  const result=await releaseChannels(identity,f.channels,f.options);
  assert.equal(result.channels['open-vsx'].status,'blocked');
  assert.equal(f.calls.publish,1);
  assert.equal(f.calls.inspect,2);
  assert.deepEqual(f.calls.sleeps,[]);
});

test('verification exceptions stop immediately without retrying a read or upload',async()=>{
  const f=fixture(attempt=>{if(attempt===1)return null;throw Error('Registry package differs from admitted immutable VSIX');});
  const result=await releaseChannels(identity,f.channels,f.options);
  assert.match(result.channels['open-vsx'].error,/differs from admitted/);
  assert.equal(f.calls.publish,1);
  assert.equal(f.calls.inspect,2);
  assert.deepEqual(f.calls.sleeps,[]);
});

test('failed submission never starts a verification polling loop',async()=>{
  const f=fixture(()=>null);
  f.channels['open-vsx'].publish=async()=>{f.calls.publish++;throw Error('Uncertain upload result');};
  const result=await releaseChannels(identity,f.channels,f.options);
  assert.match(result.channels['open-vsx'].error,/Uncertain upload/);
  assert.equal(f.calls.publish,1);
  assert.equal(f.calls.inspect,1);
  assert.deepEqual(f.calls.sleeps,[]);
});
