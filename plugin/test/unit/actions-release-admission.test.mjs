import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {contentHash} from '../../scripts/source-release-admission.mjs';
import {requiredReleaseJobs,unifiedWorkflow,validateActionsReleaseRun,verifyActionsReleaseAdmission} from '../../scripts/actions-release-admission.mjs';
function fixture(){
 const sourceRevision='a'.repeat(40),digest='b'.repeat(64),repository='tensorviz-org/tensorviz';
 const manifest={schemaVersion:1,repository,version:'1.0.0',sourceRevision,build:{workflow:unifiedWorkflow,runId:123,runAttempt:1},files:{}};
 const policy={schemaVersion:1,enabled:true,repository,version:'1.0.0',previousVersion:'0.1.48',previousCatalogCommit:'8b0b539953c5abe693803c15854d953a00bcd5fe',evidence:{method:'github-actions',workflow:unifiedWorkflow}};
 const run={id:123,run_attempt:1,head_sha:sourceRevision,repository:{full_name:repository},head_repository:{full_name:repository},path:unifiedWorkflow,event:'workflow_dispatch',head_branch:'main',status:'completed',conclusion:'success'};
 const jobs=requiredReleaseJobs.map(name=>({name,conclusion:'success'}));
 const platforms=['darwin-arm64','darwin-x64','linux-x64','linux-arm64','win32-x64','win32-arm64'];
 const release={version:'1.0.0',sourceRevision,files:Object.fromEntries(platforms.map(p=>[`tensorviz-runtime-1.0.0-${p}.tar.gz`,digest])),publication:{runtime:{runtimes:Object.fromEntries(platforms.map(p=>[p,{sha256:digest}]))}}};
 const migrationReceipts=['darwin-arm64','linux-x64'].map(platform=>({schemaVersion:1,fromVersion:'0.1.48',toVersion:'1.0.0',previousExport:policy.previousCatalogCommit,sourceRevision,platform,runtimeSha256:digest,clients:Object.fromEntries(['codex','claude'].map((n,i)=>[n,{version:['0.165.0','2.1.308'][i],upgrade:true,rollback:true,nativeCapture:true,preservedSettings:true}]))}));
 const admission={schemaVersion:2,method:'github-actions',runId:123,runAttempt:1,migrationReceipts};
 const bytes=Buffer.from(JSON.stringify(manifest));
 const proof={verificationResult:{statement:{subject:[{digest:{sha256:contentHash(bytes)}}],predicate:{runDetails:{metadata:{invocationId:`https://github.com/${repository}/actions/runs/123/attempts/1`}}}}}};
 const calls=[];const command=(tool,args)=>{calls.push(args);assert.equal(tool,'gh');if(args[0]==='attestation'){
  assert.deepEqual(fs.readFileSync(args[2]),bytes);assert.ok(args.includes('--deny-self-hosted-runners'));
  assert.equal(args[args.indexOf('--source-digest')+1],sourceRevision);assert.equal(args[args.indexOf('--signer-digest')+1],sourceRevision);
  assert.equal(args[args.indexOf('--signer-workflow')+1],`${repository}/${unifiedWorkflow}`);return JSON.stringify([proof]);}
  return JSON.stringify(args[3].includes('/jobs?')?{total_count:jobs.length,jobs}:run);};
 return {manifest,policy,run,jobs,release,admission,bytes,proof,calls,command};
}
test('live trusted run, all native/migration jobs and signed exact-byte provenance are mandatory',async()=>{
 const f=fixture();const result=await verifyActionsReleaseAdmission(f.manifest,f.bytes,f.admission,f.policy,f.release,{command:f.command});
 assert.equal(result.jobs,13);assert.equal(result.runId,123);assert.equal(f.calls.length,3);
});
for(const [name,mutate] of [
 ['disabled policy',f=>f.policy.enabled=false],['wrong workflow',f=>f.run.path='untrusted.yml'],
 ['unknown policy schema',f=>f.policy.schemaVersion=2],
 ['unreviewed future migration',f=>{f.policy.version='1.1.0';f.manifest.version='1.1.0';}],
 ['wrong manifest repository',f=>f.manifest.repository='fork/tensorviz'],
 ['fork',f=>f.run.head_repository.full_name='fork/tensorviz'],['PR event',f=>f.run.event='pull_request'],
 ['branch',f=>f.run.head_branch='feature'],['stale source',f=>f.run.head_sha='c'.repeat(40)],
 ['stale attempt',f=>f.run.run_attempt=2],['failed run',f=>f.run.conclusion='failure'],
 ['skipped native job',f=>f.jobs.find(j=>j.name==='native-win32-arm64').conclusion='skipped'],
 ['missing migration',f=>f.jobs.pop()],['duplicate job',f=>f.jobs.push(f.jobs[0])],
])test(`admission rejects ${name}`,()=>{const f=fixture();mutate(f);assert.throws(()=>validateActionsReleaseRun(f.manifest,f.admission,f.policy,f.run,f.jobs));});
test('claimed success never substitutes for genuine migration receipt validation',async()=>{
 const f=fixture();f.admission.migrationReceipts[0].clients.claude.rollback=false;
 await assert.rejects(verifyActionsReleaseAdmission(f.manifest,f.bytes,f.admission,f.policy,f.release,{command:f.command}),/Incomplete installed-client/);
 assert.equal(f.calls.length,2);
});
test('migration evidence must describe the admitted release',async()=>{
 const f=fixture();f.release.version='1.1.0';
 await assert.rejects(verifyActionsReleaseAdmission(f.manifest,f.bytes,f.admission,f.policy,f.release,{command:f.command}),/Migration release identity/);
 assert.equal(f.calls.length,2);
});
test('an attestation from another invocation or inventory cannot authorize upload',async()=>{
 for(const mutate of [f=>f.proof.verificationResult.statement.predicate.runDetails.metadata.invocationId+='2',f=>f.proof.verificationResult.statement.subject[0].digest.sha256='0'.repeat(64)]){
  const f=fixture();mutate(f);await assert.rejects(verifyActionsReleaseAdmission(f.manifest,f.bytes,f.admission,f.policy,f.release,{command:f.command}),/Signed provenance/);
 }
});
