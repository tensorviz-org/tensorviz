import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {isDeepStrictEqual} from 'node:util';
import {contentHash,verifySourceAdmission} from './source-release-admission.mjs';
const safe=name=>/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name);

/** Complete bundle contract: plugin/ is an audited legacy-shaped candidate whose
 * NEW runtime URLs name core; assets/ holds the single combined upload inventory.
 * Manifest/proof envelopes bind all channels before any GitHub release mutation.
 */
export async function verifyUnifiedBundle(directory,policy,{verifyCandidate}={}) {
  if(!verifyCandidate) ({verifyReleaseCandidate:verifyCandidate}=await import('./release-candidate.mjs'));
  // Retain the exact bytes validated and signed; never reopen them for upload.
  const manifestBytes=await fs.readFile(path.join(directory,'unified-release.json'));
  const admissionBytes=await fs.readFile(path.join(directory,'admission.json'));
  const pluginManifestBytes=await fs.readFile(path.join(directory,'plugin','release.json'));
  const manifest=JSON.parse(manifestBytes), envelopes=JSON.parse(admissionBytes);
  const pluginManifest=JSON.parse(pluginManifestBytes);
  if(manifest.schemaVersion!==1 || manifest.repository!=='tensorviz-org/tensorviz' || !manifest.files || typeof manifest.files!=='object') throw Error('Invalid unified manifest');
  const release=await verifyCandidate(path.join(directory,'plugin'),{publishing:true});
  if(Object.entries(pluginManifest).some(([key,value])=>!Object.hasOwn(release,key)||!isDeepStrictEqual(value,release[key]))) throw Error('Plugin manifest changed during validation');
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
  const pluginFiles={...release.files,'plugin-release.json':contentHash(pluginManifestBytes)};
  for(const [name,sha] of Object.entries(pluginFiles)) if(manifest.files[name]!==sha) throw Error('Combined inventory omits or changes audited plugin bytes');
  const vsix=`tensorviz-${manifest.version}.vsix`;
  const allowed=[...Object.keys(pluginFiles),vsix,'extension-provenance.json'].sort();
  if(JSON.stringify(names)!==JSON.stringify(allowed)) throw Error('Unexpected combined release asset; update closed inventory explicitly');
  const extension=JSON.parse(assets['extension-provenance.json'].bytes);
  if(extension.schemaVersion!==1||extension.sourceRevision!==manifest.sourceRevision||extension.version!==manifest.version||extension.sha256!==manifest.files[vsix]) throw Error('Extension source provenance mismatch');
  // Inspect the very same verified bytes that will be published, not a mutable
  // caller path. mkdtemp is private; only exact package metadata is read, not extracted.
  const snapshot=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-vsix-admission-'));
  let pkg;
  try {
    const snapshotFile=path.join(snapshot,'verified.vsix');
    await fs.writeFile(snapshotFile,assets[vsix].bytes,{flag:'wx',mode:0o600});
    pkg=JSON.parse(execFileSync('unzip',['-p',snapshotFile,'extension/package.json'],{encoding:'utf8',maxBuffer:1024*1024}));
  } finally { await fs.rm(snapshot,{recursive:true,force:true}); }
  if(pkg.version!==manifest.version || pkg.name!==policy.extensionName || pkg.publisher!==policy.extensionPublisher || !policy.extensionName || !policy.extensionPublisher) throw Error('VSIX identity/version mismatch');
  const admission=verifySourceAdmission(manifest,envelopes,policy);
  assets['unified-release.json']={bytes:manifestBytes,sha256:contentHash(manifestBytes)};
  assets['admission.json']={bytes:admissionBytes,sha256:contentHash(admissionBytes)};
  return {...manifest,assets,admission};
}
