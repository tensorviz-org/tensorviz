// Offline admission of reviewed, signed exact-artifact local test evidence.
import {createHash,createPublicKey,verify} from 'node:crypto';
export const sourceReleasePlatforms=Object.freeze(['darwin-arm64','darwin-x64','linux-arm64','linux-x64','win32-arm64','win32-x64']);
export const sourceReleaseClients=Object.freeze(['extension','codex','claude','cursor']);
// Windows Claude/Cursor are not currently advertised. Do not invent their support.
export const sourceReleaseMatrix=Object.freeze(sourceReleasePlatforms.flatMap(platform=>sourceReleaseClients.filter(client=>!platform.startsWith('win32-')||['extension','codex'].includes(client)).map(client=>({platform,client}))));
export const contentHash=bytes=>createHash('sha256').update(bytes).digest('hex');
export const inventoryHash=files=>contentHash(Buffer.from(JSON.stringify(Object.fromEntries(Object.entries(files).sort(([a],[b])=>a.localeCompare(b))))));
export function verifySourceAdmission(manifest,envelopes,policy) {
  if(policy?.schemaVersion!==1 || policy.enabled!==true || policy.repository!=='tensorviz-org/tensorviz') throw Error('Reviewed launch policy is not enabled');
  if(!/^[a-f0-9]{40}$/.test(manifest.sourceRevision??'') || !/^[1-9]\d*\.\d+\.\d+$/.test(manifest.version??'')) throw Error('Invalid source release');
  if(policy.version!==manifest.version || !/^[a-f0-9]{40}$/.test(policy.previousCatalogCommit??'') || !/^\d+\.\d+\.\d+$/.test(policy.previousVersion??'')) throw Error('Pinned migration policy mismatch');
  const rows=sourceReleaseMatrix.map(({platform,client})=>`${platform}/${client}`);
  if(!Array.isArray(envelopes)||envelopes.length!==rows.length) throw Error('Every supported native client/architecture needs signed evidence');
  const seen=new Set(), inventory=inventoryHash(manifest.files);
  for(const envelope of envelopes) {
    const publicKey=policy.reviewers?.[envelope.keyId];
    if(typeof publicKey!=='string' || typeof envelope.payload!=='string' || typeof envelope.signature!=='string') throw Error('Unknown evidence reviewer');
    const bytes=Buffer.from(envelope.payload,'base64');
    const key=createPublicKey(publicKey);
    if(key.asymmetricKeyType!=='ed25519' || !verify(null,bytes,key,Buffer.from(envelope.signature,'base64'))) throw Error('Invalid evidence signature');
    const receipt=JSON.parse(bytes.toString('utf8')), row=`${receipt.platform}/${receipt.client}`;
    if(!rows.includes(row)||seen.has(row)) throw Error('Unexpected or duplicate evidence');
    seen.add(row);
    if(receipt.schemaVersion!==1 || receipt.sourceRevision!==manifest.sourceRevision || receipt.version!==manifest.version
      || receipt.inventorySha256!==inventory || receipt.fromVersion!==policy.previousVersion || receipt.previousCatalogCommit!==policy.previousCatalogCommit
      || receipt.fromRepository!==(receipt.client==='extension'?'tensorviz-org/tensorviz':'tensorviz-org/tensorviz-plugins') || receipt.toRepository!=='tensorviz-org/tensorviz'
      || receipt.clientVersion!==policy.clientVersions?.[receipt.client] || typeof receipt.clientVersion!=='string'
      || !/^[a-f0-9]{64}$/.test(receipt.logSha256??'') || !Number.isFinite(Date.parse(receipt.completedAt))) throw Error('Evidence provenance mismatch');
    for(const check of ['cleanInstall','upgrade','rollback','preservedSettings','nativeCapture','noTorchStartup','missingTorchGuidance','sourceMigration']) {
      if(receipt.checks?.[check]!==true) throw Error(`Incomplete evidence: ${row}/${check}`);
    }
  }
  return {rows:seen.size,inventorySha256:inventory};
}
