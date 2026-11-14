import {test} from 'node:test';
import assert from 'node:assert/strict';
import {waitForInstalledMcp} from './unix-mcp-readiness.mjs';
const state = (runtimeStatus, tools = {}, toolsError = null) => ({data:[{name:'tensorviz', runtimeStatus, tools, toolsError}]});
const ready = state('connected', {open_graph:{name:'open_graph'}});
test('polls startup then reuses named thread server', async()=>{
 let n=0,clock=0;const params=[];
 const result=await waitForInstalledMcp(async p=>{params.push(p);return n++ ? ready : state('starting');},{now:()=>clock,sleep:async ms=>{clock+=ms;}});
 assert.equal(result.name,'tensorviz');assert.equal(params.length,2);assert.equal(params[1].serverName,'tensorviz');assert.ok(params[1].remainingMs<30000);
});
for(const status of ['failed','cancelled','disabled','authenticationRequired'])test(`does not retry ${status}`,async()=>{
 let count=0;await assert.rejects(waitForInstalledMcp(async()=>{count++;return state(status);}));assert.equal(count,1);
});
test('does not retry toolsError even when state starting',async()=>{let count=0;await assert.rejects(waitForInstalledMcp(async()=>{count++;return state('starting',{},'certificate rejected');}),/certificate rejected/);assert.equal(count,1);});
test('bounded timeout on persistent empty catalog',async()=>{let clock=0,count=0;await assert.rejects(waitForInstalledMcp(async()=>{count++;return state('starting');},{timeoutMs:1000,intervalMs:250,now:()=>clock,sleep:async ms=>{clock+=ms;}}),/within 30 seconds/);assert.equal(clock,1000);assert.equal(count,4);});
test('a late ready response cannot bypass deadline',async()=>{let clock=0;await assert.rejects(waitForInstalledMcp(async()=>{clock=30001;return ready;},{now:()=>clock}),/deadline/);});

test('connected empty catalog is not silently retried',async()=>{let count=0;await assert.rejects(waitForInstalledMcp(async()=>{count++;return state('connected');}),/terminal state/);assert.equal(count,1);});
test('ready server resolves immediately',async()=>{let count=0;const result=await waitForInstalledMcp(async p=>{assert.equal(p.serverName,'tensorviz');count++;return ready;});assert.equal(result.name,'tensorviz');assert.equal(count,1);});
for(const pending of [null,'notStarted'])test(`pending ${pending} can become ready`,async()=>{let count=0,clock=0;await waitForInstalledMcp(async()=>count++?ready:state(pending),{now:()=>clock,sleep:async ms=>{clock+=ms;}});assert.equal(count,2);});
test('different server name fails',async()=>{await assert.rejects(waitForInstalledMcp(async()=>({data:[{...ready.data[0],name:'other'}]})),/name changed/);});
