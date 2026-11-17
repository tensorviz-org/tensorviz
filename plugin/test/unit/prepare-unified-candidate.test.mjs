import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {contentHash} from '../../scripts/source-release-admission.mjs';
import {snapshotCandidateVsix} from '../../scripts/prepare-unified-candidate.mjs';

async function fixture(t, identity={name:'tensorviz-vscode',publisher:'tensorviz',version:'1.0.0'}) {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'unified-vsix-snapshot-'));
  t.after(()=>fs.rm(root,{recursive:true,force:true}));
  await fs.mkdir(path.join(root,'extension'));
  await fs.writeFile(path.join(root,'extension/package.json'),JSON.stringify(identity));
  const source=path.join(root,'source.vsix'),destination=path.join(root,'snapshot.vsix');
  execFileSync('zip',['-q',source,'extension/package.json'],{cwd:root});
  return {root,source,destination};
}
test('candidate metadata and checksum are read from the same copied VSIX bytes',async t=>{
  const f=await fixture(t),original=await fs.readFile(f.source),copyFile=fs.copyFile;
  try {
    fs.copyFile=async(...args)=>{await copyFile(...args);await fs.writeFile(f.source,'replaced after capture');};
    const digest=await snapshotCandidateVsix(f.source,f.destination,'1.0.0');
    assert.equal(digest,contentHash(original));assert.deepEqual(await fs.readFile(f.destination),original);
  }finally{fs.copyFile=copyFile;}
});
for(const change of [{name:'wrong'},{publisher:'wrong'},{version:'0.1.48'}])test(`rejects substituted package ${JSON.stringify(change)}`,async t=>{
  const f=await fixture(t,{name:'tensorviz-vscode',publisher:'tensorviz',version:'1.0.0',...change});
  await assert.rejects(snapshotCandidateVsix(f.source,f.destination,'1.0.0'),/identity mismatch/);
});
test('linked input and an existing destination cannot replace admitted artifacts',async t=>{
  const f=await fixture(t),link=path.join(f.root,'linked.vsix');await fs.symlink(f.source,link);
  await assert.rejects(snapshotCandidateVsix(link,f.destination,'1.0.0'),/regular file/);
  await fs.writeFile(f.destination,'preserve');
  await assert.rejects(snapshotCandidateVsix(f.source,f.destination,'1.0.0'),{code:'EEXIST'});
  assert.equal(await fs.readFile(f.destination,'utf8'),'preserve');
});
