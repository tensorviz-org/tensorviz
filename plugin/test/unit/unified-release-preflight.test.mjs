import test from 'node:test';
import assert from 'node:assert/strict';
import {unifiedReleasePreflight} from '../../scripts/unified-release-preflight.mjs';
import {unifiedVersionFiles} from '../../scripts/unified-source-versions.mjs';
import {unifiedWorkflow} from '../../scripts/actions-release-admission.mjs';
function fixture(){
 const source='a'.repeat(40),head='b'.repeat(40),repository='tensorviz-org/tensorviz';
 const input={source,head,publish:true,candidateRunId:'123',repository,ref:'refs/heads/main',event:'workflow_dispatch'};
 const policy={repository,version:'1.0.0',enabled:true,channels:{enabled:true},evidence:{method:'github-actions',workflow:unifiedWorkflow}};
 const run={head_sha:source,path:unifiedWorkflow,head_branch:'main',repository:{full_name:repository},head_repository:{full_name:repository},event:'workflow_dispatch',status:'completed',conclusion:'success'};
 const calls=[];const command=(tool,args)=>{calls.push([tool,...args]);if(tool==='gh')return JSON.stringify(run);
  if(args[0]==='merge-base')return '';
  const file=args[1].split(':')[1];if(file.endsWith('public-launch-policy.json'))return JSON.stringify(policy);
  assert.ok(unifiedVersionFiles.includes(file));return JSON.stringify({version:'1.0.0',packages:{'':{version:'1.0.0'}}});};
 return {input,policy,run,command,calls};
}
test('build binds current main and publication selects a successful exact-source candidate',()=>{
 const f=fixture();assert.equal(unifiedReleasePreflight(f.input,{command:f.command}).source,f.input.source);
 assert.deepEqual(unifiedReleasePreflight({...f.input,publish:false,source:'ignored'},{command:f.command}),{source:f.input.head,version:'1.0.0',publish:false,candidateRunId:''});
 assert.ok(f.calls.every(c=>!c.includes('push')&&!c.includes('commit')));
});
for(const [name,mutate] of [
 ['fork request',f=>f.input.repository='fork/tensorviz'],['branch request',f=>f.input.ref='refs/heads/feature'],
 ['PR request',f=>f.input.event='pull_request'],['mutable source',f=>f.input.source='main'],
 ['missing run',f=>f.input.candidateRunId=''],['disabled policy',f=>f.policy.enabled=false],
 ['forked candidate',f=>f.run.head_repository.full_name='fork/tensorviz'],['stale candidate',f=>f.run.head_sha=f.input.head],
 ['untrusted builder',f=>f.run.path='untrusted.yml'],['unfinished candidate',f=>f.run.status='in_progress'],
])test(`preflight blocks ${name}`,()=>{const f=fixture();mutate(f);assert.throws(()=>unifiedReleasePreflight(f.input,{command:f.command}));});
