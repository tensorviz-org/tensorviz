import test from 'node:test';
import assert from 'node:assert/strict';
import {unifiedReleasePreflight} from '../../scripts/unified-release-preflight.mjs';
import {unifiedVersionFiles} from '../../scripts/unified-source-versions.mjs';
import {unifiedWorkflow} from '../../scripts/actions-release-admission.mjs';
function fixture(version='1.0.0'){
 const source='a'.repeat(40),head='b'.repeat(40),repository='tensorviz-org/tensorviz';
 const input={source,head,publish:true,candidateRunId:'123',repository,ref:'refs/heads/main',event:'workflow_dispatch'};
 const policy={schemaVersion:1,repository,version,previousVersion:version==='1.0.1'?'1.0.0':'0.1.48',previousCatalogCommit:version==='1.0.1'?'664d5615d71c10b5d9f84695c4c40011dddbdc7f':'8b0b539953c5abe693803c15854d953a00bcd5fe',enabled:true,channels:{enabled:true},evidence:{method:'github-actions',workflow:unifiedWorkflow}};
 const documents=Object.fromEntries(unifiedVersionFiles.map(file=>[file,{version,packages:{'':{version}}}]));
 const run={head_sha:source,path:unifiedWorkflow,head_branch:'main',repository:{full_name:repository},head_repository:{full_name:repository},event:'workflow_dispatch',status:'completed',conclusion:'success'};
 const calls=[];const command=(tool,args)=>{calls.push([tool,...args]);if(tool==='gh')return JSON.stringify(run);
  if(args[0]==='merge-base')return '';
  const file=args[1].split(':')[1];if(file.endsWith('public-launch-policy.json'))return JSON.stringify(policy);
  assert.ok(unifiedVersionFiles.includes(file));return JSON.stringify(documents[file]);};
 return {input,policy,run,command,calls,documents};
}
test('build binds current main and publication selects a successful exact-source candidate',()=>{
 const f=fixture();assert.equal(unifiedReleasePreflight(f.input,{command:f.command}).source,f.input.source);
 assert.deepEqual(unifiedReleasePreflight({...f.input,publish:false,source:'ignored'},{command:f.command}),{source:f.input.head,version:'1.0.0',publish:false,candidateRunId:''});
 assert.ok(f.calls.every(c=>!c.includes('push')&&!c.includes('commit')));
});
test('the reviewed patch release selects 1.0.1 while retaining exact source and migration pins',()=>{
 const f=fixture('1.0.1');
 assert.equal(unifiedReleasePreflight(f.input,{command:f.command}).version,'1.0.1');
 f.documents['plugin/package.json'].version='1.0.0';
 assert.throws(()=>unifiedReleasePreflight(f.input,{command:f.command}),/Source version mismatch/);
});
for(const [name,mutate] of [
 ['fork request',f=>f.input.repository='fork/tensorviz'],['branch request',f=>f.input.ref='refs/heads/feature'],
 ['PR request',f=>f.input.event='pull_request'],['mutable source',f=>f.input.source='main'],
 ['missing run',f=>f.input.candidateRunId=''],['disabled policy',f=>f.policy.enabled=false],
 ['unknown release',f=>f.policy.version='1.0.2'],['wrong previous version',f=>f.policy.previousVersion='0.1.47'],
 ['wrong previous catalog',f=>f.policy.previousCatalogCommit='c'.repeat(40)],
 ['forked candidate',f=>f.run.head_repository.full_name='fork/tensorviz'],['stale candidate',f=>f.run.head_sha=f.input.head],
 ['untrusted builder',f=>f.run.path='untrusted.yml'],['unfinished candidate',f=>f.run.status='in_progress'],
])test(`preflight blocks ${name}`,()=>{const f=fixture();mutate(f);assert.throws(()=>unifiedReleasePreflight(f.input,{command:f.command}));});
