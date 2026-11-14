import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {contentHash,verifySourceAdmission} from './source-release-admission.mjs';
const safe=name=>/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name);
const readJson=async p=>JSON.parse(await fs.readFile(p,'utf8'));

/** Complete bundle contract: plugin/ is an audited legacy-shaped candidate whose
 * NEW runtime URLs name core; assets/ holds the single combined upload inventory.
 * Manifest/proof envelopes bind all channels before any GitHub release mutation.
 */
export async function verifyUnifiedBundle(directory,policy,{verifyCandidate}={}) {
  if(!verifyCandidate) ({verifyReleaseCandidate:verifyCandidate}=await import('./release-candidate.mjs'));
  const manifest=await readJson(path.join(directory,'unified-release.json'));
  const envelopes=await readJson(path.join(directory,'admission.json'));
  if(manifest.schemaVersion!==1 || manifest.repository!=='tensorviz-org/tensorviz' || !manifest.files || typeof manifest.files!=='object') throw Error('Invalid unified manifest');
  const release=await verifyCandidate(path.join(directory,'plugin'),{publishing:true});
  if(release.version!==manifest.version||release.sourceRevision!==manifest.sourceRevision) throw Error('Combined source/version mismatch');
  const assets=Object.create(null);
  const names=Object.keys(manifest.files).sort();
  if(JSON.stringify((await fs.readdir(path.join(directory,'assets'))).sort())!==JSON.stringify(names)) throw Error('Combined asset inventory differs');
  for(const name of names) {
    if(!safe(name)||! /^[a-f0-9]{64}$/.test(manifest.files[name])) throw Error('Unsafe asset inventory');
    const file=path.join(directory,'assets',name);
    if(!(await fs.lstat(file)).isFile()) throw Error('Asset is not a regular file');
    const bytes=await fs.readFile(file);
    if(contentHash(bytes)!==manifest.files[name]) throw Error('Combined checksum mismatch');
    assets[name]={bytes,sha256:manifest.files[name]};
  }
  const pluginFiles={...release.files,'plugin-release.json':contentHash(await fs.readFile(path.join(directory,'plugin','release.json')))};
  for(const [name,sha] of Object.entries(pluginFiles)) if(manifest.files[name]!==sha) throw Error('Combined inventory omits or changes audited plugin bytes');
  const vsix=`tensorviz-${manifest.version}.vsix`;
  const allowed=[...Object.keys(pluginFiles),vsix,'extension-provenance.json'].sort();
  if(JSON.stringify(names)!==JSON.stringify(allowed)) throw Error('Unexpected combined release asset; update closed inventory explicitly');
  const extension=JSON.parse(assets['extension-provenance.json'].bytes);
  if(extension.schemaVersion!==1||extension.sourceRevision!==manifest.sourceRevision||extension.version!==manifest.version||extension.sha256!==manifest.files[vsix]) throw Error('Extension source provenance mismatch');
  // Do not extract an archive to disk. Read only the exact VSIX package metadata.
  const pkg=JSON.parse(execFileSync('unzip',['-p',path.join(directory,'assets',vsix),'extension/package.json'],{encoding:'utf8',maxBuffer:1024*1024}));
  if(pkg.version!==manifest.version || pkg.name!==policy.extensionName || pkg.publisher!==policy.extensionPublisher || !policy.extensionName || !policy.extensionPublisher) throw Error('VSIX identity/version mismatch');
  const admission=verifySourceAdmission(manifest,envelopes,policy);
  for(const name of ['unified-release.json','admission.json']) {
    const bytes=await fs.readFile(path.join(directory,name));assets[name]={bytes,sha256:contentHash(bytes)};
  }
  return {...manifest,assets,admission};
}
