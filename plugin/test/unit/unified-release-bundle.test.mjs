import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {generateKeyPairSync,sign} from 'node:crypto';
import {verifyUnifiedBundle} from '../../scripts/unified-release-bundle.mjs';
import {contentHash,inventoryHash,sourceReleasePlatforms,sourceReleaseClients,sourceReleaseMatrix} from '../../scripts/source-release-admission.mjs';
async function fixture() {
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'unified-bundle-test-'));
 await fs.mkdir(path.join(root,'assets'));await fs.mkdir(path.join(root,'plugin'));await fs.mkdir(path.join(root,'zip/extension'),{recursive:true});
 const version='1.0.0',sourceRevision='a'.repeat(40),vsix=`tensorviz-${version}.vsix`;
 await fs.writeFile(path.join(root,'zip/extension/package.json'),JSON.stringify({name:'tensorviz',publisher:'tensorviz',version}));
 execFileSync('zip',['-q',path.join(root,'assets',vsix),'extension/package.json'],{cwd:path.join(root,'zip')});
 const release={version,sourceRevision,files:{'plugin.zip':contentHash(Buffer.from('dummy mocked plugin'))}};
 await fs.writeFile(path.join(root,'plugin/release.json'),JSON.stringify(release));
 await fs.copyFile(path.join(root,'plugin/release.json'),path.join(root,'assets/plugin-release.json'));
 await fs.writeFile(path.join(root,'assets/plugin.zip'),'dummy mocked plugin');
 await fs.writeFile(path.join(root,'assets/extension-provenance.json'),JSON.stringify({schemaVersion:1,version,sourceRevision,sha256:contentHash(await fs.readFile(path.join(root,'assets',vsix)))}));
 const files={};for(const name of await fs.readdir(path.join(root,'assets')))files[name]=contentHash(await fs.readFile(path.join(root,'assets',name)));
 const manifest={schemaVersion:1,repository:'tensorviz-org/tensorviz',version,sourceRevision,files};
 const keys=generateKeyPairSync('ed25519');
 const policy={schemaVersion:1,enabled:true,repository:manifest.repository,version,previousVersion:'0.1.46',previousCatalogCommit:'b'.repeat(40),extensionName:'tensorviz',extensionPublisher:'tensorviz',clientVersions:Object.fromEntries(sourceReleaseClients.map(c=>[c,'1.2.3'])),reviewers:{test:keys.publicKey.export({type:'spki',format:'pem'})}};
 const envelopes=sourceReleaseMatrix.map(({platform,client})=>{
 const receipt={schemaVersion:1,platform,client,clientVersion:'1.2.3',sourceRevision,version,inventorySha256:inventoryHash(files),fromVersion:policy.previousVersion,previousCatalogCommit:policy.previousCatalogCommit,fromRepository:client==='extension'?'tensorviz-org/tensorviz':'tensorviz-org/tensorviz-plugins',toRepository:manifest.repository,logSha256:'c'.repeat(64),completedAt:'2026-11-14T20:00:00Z',checks:Object.fromEntries(['cleanInstall','upgrade','rollback','preservedSettings','nativeCapture','noTorchStartup','missingTorchGuidance','sourceMigration'].map(k=>[k,true]))};
 const bytes=Buffer.from(JSON.stringify(receipt));return {keyId:'test',payload:bytes.toString('base64'),signature:sign(null,bytes,keys.privateKey).toString('base64')};
 });
 await fs.writeFile(path.join(root,'unified-release.json'),JSON.stringify(manifest));await fs.writeFile(path.join(root,'admission.json'),JSON.stringify(envelopes));
 return {root,manifest,policy,release,options:{verifyCandidate:async()=>release}};
}
test('combined admission validates real ZIP metadata, signed inventory and exact plugin bytes (plugin verifier mocked)',async()=>{const f=await fixture();try{const result=await verifyUnifiedBundle(f.root,f.policy,f.options);assert.equal(result.admission.rows,20);assert.equal(Object.keys(result.assets).length,6);}finally{await fs.rm(f.root,{recursive:true,force:true});}});
for(const [name,mutate] of [
 ['unexpected file',f=>fs.writeFile(path.join(f.root,'assets/extra'),'x')],
 ['tampered VSIX',f=>fs.appendFile(path.join(f.root,'assets/tensorviz-1.0.0.vsix'),'tampered')],
 ['wrong package identity',async f=>f.policy.extensionName='wrong'],
 ['omitted plugin bytes',async f=>f.release.files.other='d'.repeat(64)],
 ['wrong source',async f=>f.release.sourceRevision='d'.repeat(40)],
 ])test(`combined admission rejects ${name}`,async()=>{const f=await fixture();try{await mutate(f);await assert.rejects(verifyUnifiedBundle(f.root,f.policy,f.options));}finally{await fs.rm(f.root,{recursive:true,force:true});}});

test('manifest and admission uploads retain validated bytes despite input replacement',async()=>{
 const f=await fixture();
 try {
  const originalManifest=await fs.readFile(path.join(f.root,'unified-release.json'));
  const originalAdmission=await fs.readFile(path.join(f.root,'admission.json'));
  f.options.verifyCandidate=async()=>{
    await fs.writeFile(path.join(f.root,'unified-release.json'),JSON.stringify({...f.manifest,schemaVersion:999}));
    await fs.writeFile(path.join(f.root,'admission.json'),'[]');
    return f.release;
  };
  const result=await verifyUnifiedBundle(f.root,f.policy,f.options);
  assert.deepEqual(result.assets['unified-release.json'].bytes,originalManifest);
  assert.deepEqual(result.assets['admission.json'].bytes,originalAdmission);
  assert.equal(JSON.parse(result.assets['unified-release.json'].bytes).schemaVersion,1);
 } finally {await fs.rm(f.root,{recursive:true,force:true});}
});
test('VSIX metadata inspection uses captured hash-verified bytes after original path replacement',async()=>{
 const f=await fixture(),readFile=fs.readFile;
 const vsix=path.join(f.root,'assets/tensorviz-1.0.0.vsix');
 try {
  const original=await readFile(vsix);let replaced=false;
  fs.readFile=async function(file,...args){
    const bytes=await readFile(file,...args);
    if(file===vsix&&!replaced){replaced=true;await fs.writeFile(vsix,'not a zip anymore');}
    return bytes;
  };
  const result=await verifyUnifiedBundle(f.root,f.policy,f.options);
  assert.equal(replaced,true);assert.deepEqual(result.assets['tensorviz-1.0.0.vsix'].bytes,original);
  assert.equal(await readFile(vsix,'utf8'),'not a zip anymore');
 } finally {fs.readFile=readFile;await fs.rm(f.root,{recursive:true,force:true});}
});
test('plugin manifest changing its verified identity is rejected',async()=>{
 const f=await fixture();try{f.options.verifyCandidate=async()=>({...f.release,files:{'different.zip':'f'.repeat(64)}});await assert.rejects(verifyUnifiedBundle(f.root,f.policy,f.options),/Plugin manifest changed/);}finally{await fs.rm(f.root,{recursive:true,force:true});}
});

test('plugin manifest coverage metadata cannot change during verification',async()=>{
 const f=await fixture();try{
  const original={...f.release,rendererDistributionCoverage:'unconfirmed'};
  await fs.writeFile(path.join(f.root,'plugin/release.json'),JSON.stringify(original));
  f.options.verifyCandidate=async()=>({...f.release,rendererDistributionCoverage:'owner-confirmed'});
  await assert.rejects(verifyUnifiedBundle(f.root,f.policy,f.options),/Plugin manifest changed/);
 }finally{await fs.rm(f.root,{recursive:true,force:true});}
});
