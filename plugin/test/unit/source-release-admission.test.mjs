import test from 'node:test';
import assert from 'node:assert/strict';
import {generateKeyPairSync,sign} from 'node:crypto';
import {verifySourceAdmission,sourceReleasePlatforms,sourceReleaseClients,sourceReleaseMatrix,inventoryHash} from '../../scripts/source-release-admission.mjs';
import {distributionRepositoryForVersion,runtimeReleaseURL} from '../../scripts/release-version.mjs';
const {publicKey,privateKey}=generateKeyPairSync('ed25519');
const stamp=r=>{const bytes=Buffer.from(JSON.stringify(r));return {keyId:'reviewer',payload:bytes.toString('base64'),signature:sign(null,bytes,privateKey).toString('base64')};};
function fixture() {
 const manifest={version:'1.0.0',sourceRevision:'a'.repeat(40),files:{'artifact.zip':'b'.repeat(64)}};
 const policy={schemaVersion:1,enabled:true,repository:'tensorviz-org/tensorviz',version:'1.0.0',previousVersion:'0.1.46',previousCatalogCommit:'c'.repeat(40),clientVersions:Object.fromEntries(sourceReleaseClients.map(c=>[c,'1.2.3'])),reviewers:{reviewer:publicKey.export({type:'spki',format:'pem'})}};
 const receipts=sourceReleaseMatrix.map(({platform,client})=>({schemaVersion:1,platform,client,clientVersion:'1.2.3',sourceRevision:manifest.sourceRevision,version:manifest.version,inventorySha256:inventoryHash(manifest.files),fromVersion:policy.previousVersion,previousCatalogCommit:policy.previousCatalogCommit,fromRepository:client==='extension'?'tensorviz-org/tensorviz':'tensorviz-org/tensorviz-plugins',toRepository:'tensorviz-org/tensorviz',logSha256:'d'.repeat(64),completedAt:'2026-11-14T20:00:00Z',checks:Object.fromEntries(['cleanInstall','upgrade','rollback','preservedSettings','nativeCapture','noTorchStartup','missingTorchGuidance','sourceMigration'].map(k=>[k,true]))}));
 return {manifest,policy,receipts};
}
test('20 signed exact-artifact receipts cover every OS/arch/client',()=>{const f=fixture();assert.equal(verifySourceAdmission(f.manifest,f.receipts.map(stamp),f.policy).rows,20);});
for(const [name,mutate] of [
 ['disabled policy',f=>f.policy.enabled=false], ['missing Windows evidence',f=>f.receipts.pop()],
 ['duplicate evidence',f=>f.receipts[0]=f.receipts[1]], ['wrong source',f=>f.receipts[0].sourceRevision='e'.repeat(40)],
 ['different inventory',f=>f.manifest.files['artifact.zip']='e'.repeat(64)], ['different previous catalog',f=>f.receipts[0].previousCatalogCommit='e'.repeat(40)],
 ['missing rollback',f=>f.receipts[0].checks.rollback=false], ['missing log',f=>delete f.receipts[0].logSha256],
 ['wrong client version',f=>f.receipts[0].clientVersion='2.0.0'], ['old repo destination',f=>f.receipts[0].toRepository='tensorviz-org/tensorviz-plugins']
])test(`refuses ${name}`,()=>{const f=fixture();mutate(f);assert.throws(()=>verifySourceAdmission(f.manifest,f.receipts.map(stamp),f.policy));});
test('tampered signed payload rejected',()=>{const f=fixture(),e=f.receipts.map(stamp);e[0].payload=Buffer.from('{}').toString('base64');assert.throws(()=>verifySourceAdmission(f.manifest,e,f.policy));});
test('unknown reviewer rejected',()=>{const f=fixture(),e=f.receipts.map(stamp);e[0].keyId='stranger';assert.throws(()=>verifySourceAdmission(f.manifest,e,f.policy));});
test('historic 0.x URLs unchanged; public versions use core only',()=>{assert.equal(distributionRepositoryForVersion('0.1.46'),'tensorviz-org/tensorviz-plugins');assert.equal(distributionRepositoryForVersion('1.0.0'),'tensorviz-org/tensorviz');assert.match(runtimeReleaseURL('1.0.0','win32-arm64'),/^https:\/\/github.com\/tensorviz-org\/tensorviz\/releases\/download\/v1.0.0\//);assert.throws(()=>distributionRepositoryForVersion('../evil'));assert.throws(()=>runtimeReleaseURL('1.0.0','../evil'));});
