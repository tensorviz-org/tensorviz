// Launch-only CLI. Policy, signatures, source ancestry and local evidence gate all channels.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {admitUnifiedBundle,publishSourceAssets,githubTransport} from './publish-source-release.mjs';
import {productionReleaseAdapters,publicationJournal,lockedCLI} from './production-release-adapters.mjs';
import {releaseChannels} from './release-channels.mjs';
import {contentHash} from './source-release-admission.mjs';
import {githubPublicationJournal} from './github-publication-journal.mjs';

export async function inspectUnifiedGitHub(bundle,transport) {
  const tag=`v${bundle.version}`,repo=await transport.repository(),ref=await transport.tag(tag);
  if(repo.full_name!=='tensorviz-org/tensorviz'||repo.private!==false||repo.archived||ref.object?.type!=='commit'||ref.object.sha!==bundle.sourceRevision)throw Error('Canonical published source identity mismatch');
  const release=await transport.release(tag);if(!release||release.draft)return null;
  if(release.tag_name!==tag||release.target_commitish!==bundle.sourceRevision)throw Error('Published GitHub provenance mismatch');
  const assets=await transport.assets(release.id);
  if(new Set(assets.map(a=>a.name)).size!==assets.length||JSON.stringify(assets.map(a=>a.name).sort())!==JSON.stringify(Object.keys(bundle.assets).sort()))throw Error('Published GitHub inventory mismatch');
  for(const item of assets)if(contentHash(await transport.publicDownload(tag,item.name))!==bundle.assets[item.name].sha256)throw Error('Public GitHub asset checksum mismatch');
  return {version:bundle.version,sourceRevision:bundle.sourceRevision,verified:true};
}
export async function publishUnifiedChannels(directory,sourceRoot,reviewedSource,journalRoot) {
  const {bundle,policy}=await admitUnifiedBundle(directory,sourceRoot,reviewedSource);
  const config=policy.channels;
  if(config?.enabled!==true||config.publisher!==policy.extensionPublisher||config.extension!==policy.extensionName)throw Error('Reviewed channel policy not enabled or identity differs');
  const {installReviewedReleaseTools}=await import('./release-tool-install.mjs');
  const installed=await installReviewedReleaseTools(sourceRoot,reviewedSource,config.tools);
  const tools=installed.tools;
  const identity={version:bundle.version,sourceRevision:bundle.sourceRevision,vsixSha256:bundle.files[`tensorviz-${bundle.version}.vsix`],docsSha256:bundle.docs.sha256};
  const snapshot=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-channels-'));
  try {
    for(const [name,asset]of Object.entries(bundle.assets))await fs.writeFile(path.join(snapshot,name),asset.bytes,{flag:'wx',mode:0o600});
    const transport=githubTransport(snapshot);
    const github={inspect:()=>inspectUnifiedGitHub(bundle,transport),publish:()=>publishSourceAssets(bundle,transport)};
    const journal=journalRoot==='github-draft' ? githubPublicationJournal(identity)
      : publicationJournal(path.join(path.resolve(journalRoot),bundle.version,bundle.sourceRevision));
    const adapters=productionReleaseAdapters({identity,vsixBytes:bundle.assets[`tensorviz-${bundle.version}.vsix`].bytes,docs:bundle.docs,config,tools,journal,github,
      credentials:{VSCE_PAT:process.env.VSCE_PAT,OVSX_PAT:process.env.OVSX_PAT,GOOGLE_APPLICATION_CREDENTIALS:process.env.GOOGLE_APPLICATION_CREDENTIALS,githubOidcRequestUrl:process.env.ACTIONS_ID_TOKEN_REQUEST_URL,githubOidcRequestToken:process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN}});
    const result=await releaseChannels(identity,adapters);
    if(journalRoot!=='github-draft') {
      await fs.mkdir(path.join(path.resolve(journalRoot),bundle.version,bundle.sourceRevision),{recursive:true});
      await fs.writeFile(path.join(path.resolve(journalRoot),bundle.version,bundle.sourceRevision,'channel-result.json'),JSON.stringify(result,null,2)+'\n',{mode:0o600});
    }
    return result;
  } finally {await fs.rm(snapshot,{recursive:true,force:true});await installed.cleanup();}
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  if(process.argv.length!==5)throw Error('Usage: publish-unified-channels.mjs BUNDLE SOURCE_ROOT DURABLE_JOURNAL_ROOT');
  const result=await publishUnifiedChannels(path.resolve(process.argv[2]),path.resolve(process.argv[3]),process.env.TENSORVIZ_REVIEWED_SOURCE_SHA,process.argv[4]==='github-draft'?'github-draft':path.resolve(process.argv[4]));
  console.log(JSON.stringify(result));if(!result.complete)process.exitCode=1;
}
