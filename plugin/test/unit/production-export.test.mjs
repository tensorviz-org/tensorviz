import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {exportPublic,integrationFiles} from '../../scripts/export-public.mjs';
import {auditIntegration} from '../../scripts/release-candidate.mjs';
import {verifyBootstrap} from '../../scripts/bootstrap-provenance.mjs';
import {pluginVersion} from '../../scripts/release-version.mjs';
import {nativePlatforms} from '../../scripts/native-platforms.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const version=pluginVersion();
const entry=platform=>({formatVersion:1,version,platform,pythonABI:nativePlatforms[platform].abi,sha256:'a'.repeat(64),bytes:1,url:'https://example.com/a.tar.gz'});
function pe(arch){const b=Buffer.alloc(128);b.write('MZ');b.writeUInt32LE(64,60);b.write('PE\0\0',64,'binary');b.writeUInt16LE(arch==='x64'?0x8664:0xaa64,68);return b;}
test('production export includes closed Windows Codex catalogs alongside unchanged Unix clients',async()=>{
 const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'production-export-'));
 try{
  const lock={formatVersion:2,version,runtimes:Object.fromEntries(['linux-x64','darwin-x64','linux-arm64','win32-x64','win32-arm64'].map(p=>[p,entry(p)]))};
  const inputs={};for(const arch of ['x64','arm64']){const binary=path.join(tmp,`${arch}.exe`),notices=path.join(tmp,`${arch}.md`);await fs.writeFile(binary,pe(arch));await fs.writeFile(notices,'Node and pkg notices');inputs[arch]={binary,notices,receipt:{formatVersion:1,arch,version,sourceRevision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),sha256:hash(pe(arch)),bytes:128,noticesSha256:hash('Node and pkg notices')}};}
  const out=path.join(tmp,'catalog');await exportPublic(out,{runtimeLock:lock,windowsBootstraps:inputs});
  const publication=await auditIntegration(out);assert.deepEqual(Object.keys(publication.files).sort(),integrationFiles(lock).sort());
  const market=JSON.parse(await fs.readFile(path.join(out,'.agents/plugins/marketplace.json')));assert.deepEqual(market.plugins.map(p=>p.name),['tensorviz','tensorviz-windows-x64','tensorviz-windows-arm64']);
  const claude=JSON.parse(await fs.readFile(path.join(out,'.claude-plugin/marketplace.json')));assert.equal(claude.plugins.length,1);
  await fs.writeFile(path.join(out,'plugins/tensorviz-windows-x64/scripts/tensorviz-bootstrap.exe'),'tamper');await assert.rejects(auditIntegration(out),/checksum/);
  await assert.rejects(exportPublic(path.join(tmp,'missing'),{runtimeLock:lock}),/requires/);
 }finally{await fs.rm(tmp,{recursive:true,force:true});}
});
test('bootstrap verification rejects stale source, wrong PE architecture and modified bytes',async()=>{
 const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'bootstrap-provenance-'));
 try{const arch='x64',sourceRevision='b'.repeat(40),binary=path.join(tmp,'tensorviz-bootstrap-x64.exe'),notices=path.join(tmp,'tensorviz-bootstrap-x64.NOTICES.md'),bytes=pe(arch);await fs.writeFile(binary,bytes);await fs.writeFile(notices,'notices');await fs.writeFile(`${binary}.json`,JSON.stringify({formatVersion:1,arch,version,sourceRevision,sha256:hash(bytes),bytes:bytes.length,noticesSha256:hash('notices')}));
 const archive=path.join(tmp,'runtime.tar.gz');await verifyBootstrap(archive,{arch,version,sourceRevision});await assert.rejects(verifyBootstrap(archive,{arch,version,sourceRevision:'c'.repeat(40)}),/provenance/);await fs.writeFile(binary,pe('arm64'));await assert.rejects(verifyBootstrap(archive,{arch,version,sourceRevision}),/architecture/);
 }finally{await fs.rm(tmp,{recursive:true,force:true});}
});
test('six-platform assembly produces and verifies Windows standalone and combined catalog ZIPs',async()=>{
 const {execFileSync}=await import('node:child_process');
 const {assembleNativeRelease}=await import('../../scripts/assemble-native-release.mjs');
 const {verifyReleaseCandidate}=await import('../../scripts/release-candidate.mjs');
 const sourceRevision=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
 const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'six-native-fixture-'));
 try {
  const archives=[];
  for(const platform of ['darwin-arm64','darwin-x64','linux-x64','linux-arm64','win32-x64','win32-arm64']){
   const native=path.join(tmp,platform);await fs.mkdir(native);const host=platform.startsWith('win32')?'tensorviz-host.exe':'tensorviz-host';await fs.writeFile(path.join(native,host),'synthetic native fixture, not executable evidence');
   const manifest={version,sourceRevision,platform,pythonABI:nativePlatforms[platform].abi,coreSourceIncluded:false,rendererDistributionCoverage:'owner-confirmed',workingTreeChanged:false,files:{[host]:hash(await fs.readFile(path.join(native,host)))}};
   await fs.writeFile(path.join(native,'release-manifest.json'),JSON.stringify(manifest));
   const archive=path.join(tmp,`tensorviz-runtime-${version}-${platform}.tar.gz`);execFileSync('tar',['-czf',archive,'-C',native,host,'release-manifest.json']);const bytes=await fs.readFile(archive);
   await fs.writeFile(`${archive}.json`,JSON.stringify({...entry(platform),bytes:bytes.length,sha256:hash(bytes),url:`https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v${version}/${path.basename(archive)}`}));archives.push(archive);
   if(platform.startsWith('win32-')){const arch=platform.slice(6),binary=path.join(tmp,`tensorviz-bootstrap-${arch}.exe`),bootstrap=pe(arch);await fs.writeFile(binary,bootstrap);await fs.writeFile(path.join(tmp,`tensorviz-bootstrap-${arch}.NOTICES.md`),'notices');await fs.writeFile(`${binary}.json`,JSON.stringify({formatVersion:1,arch,version,sourceRevision,sha256:hash(bootstrap),bytes:bootstrap.length,noticesSha256:hash('notices')}));}
  }
  const directory=await assembleNativeRelease(path.join(tmp,'integration'),archives);const checked=await verifyReleaseCandidate(directory,{publishing:true});assert.equal(checked.archives.length,6);assert.ok(checked.files[`tensorviz-windows-x64-${version}.zip`]);assert.ok(checked.files[`tensorviz-windows-arm64-${version}.zip`]);
  const integration=path.join(directory,'integration'), receiptName='plugins/tensorviz-windows-x64/bootstrap-provenance.json';
  const receipt=JSON.parse(await fs.readFile(path.join(integration,receiptName)));receipt.sourceRevision='f'.repeat(40);
  const altered=JSON.stringify(receipt);await fs.writeFile(path.join(integration,receiptName),altered);
  const publication=JSON.parse(await fs.readFile(path.join(integration,'PUBLICATION.json')));publication.files[receiptName]=hash(altered);await fs.writeFile(path.join(integration,'PUBLICATION.json'),JSON.stringify(publication));
  await assert.rejects(verifyReleaseCandidate(directory),/bootstrap provenance/);
 }finally{await fs.rm(tmp,{recursive:true,force:true});}
});
