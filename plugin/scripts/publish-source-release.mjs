// Additive source-tag publisher. Never creates/moves refs, edits main, or deletes assets.
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const repository = 'tensorviz-org/tensorviz';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

/** Dependency-injected transport makes all mutation semantics locally testable. */
export async function publishSourceAssets({version, sourceRevision, assets}, transport) {
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version) || Number(version.split('.')[0]) < 1 || !/^[a-f0-9]{40}$/.test(sourceRevision)) throw Error('Invalid public source release provenance');
  const names = Object.keys(assets).sort();
  if (!names.length || names.length > 95) throw Error('Invalid asset inventory');
  for (const name of names) {
    const asset = assets[name];
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) || !Buffer.isBuffer(asset.bytes) || digest(asset.bytes) !== asset.sha256) throw Error('Invalid asset or checksum');
  }
  const tag = `v${version}`;
  const assertSource = async () => {
    const repo = await transport.repository();
    if (repo.full_name !== repository || repo.private !== false || repo.archived || repo.default_branch !== 'main') throw Error('Canonical repository is not ready for public release');
    const ref = await transport.tag(tag);
    // Deliberately require a lightweight tag; annotated tags need an explicit verified peel implementation.
    if (ref.object?.type !== 'commit' || ref.object.sha !== sourceRevision) throw Error('Source tag does not match reviewed commit');
  };
  await assertSource();
  let release = await transport.release(tag);
  if (release && (release.tag_name !== tag || release.target_commitish !== sourceRevision)) throw Error('Existing release provenance differs');
  if (!release) release = await transport.create({tag_name:tag, target_commitish:sourceRevision, draft:true, prerelease:false, name:`TensorViz ${version}`, body:`Source revision: ${sourceRevision}\nAll artifacts are pinned in unified-release.json.\n`});
  const verifyHosted = async (requireAll) => {
    const hosted = await transport.assets(release.id);
    const seen = new Set();
    for (const item of hosted) {
      if (!Object.hasOwn(assets,item.name) || seen.has(item.name)) throw Error('Unexpected or duplicate hosted asset');
      seen.add(item.name);
      if (digest(await transport.download(item.id)) !== assets[item.name].sha256) throw Error('Hosted bytes differ; never overwrite');
    }
    if (requireAll && seen.size !== names.length) throw Error('Hosted assets are incomplete');
    return seen;
  };
  const seen = await verifyHosted(false);
  for (const name of names) if (!seen.has(name)) {
    if (!release.draft) throw Error('Published release is incomplete; never mutate');
    await transport.upload(tag,name,assets[name].bytes);
  }
  // Re-read uploaded bytes before exposing draft. A partial failure is safely retryable.
  await verifyHosted(true);
  await assertSource();
  if (release.draft) await transport.publish(release.id);
  await assertSource();
  for(const name of names) if(digest(await transport.publicDownload(tag,name))!==assets[name].sha256) throw Error('Public asset verification failed; retry original bundle');
  return {repository,tag,sourceRevision,assets:names};
}

export async function publishSourceCandidate(directory, {transport, reviewedSource, approvedMigration} = {}) {
  if (process.env.TENSORVIZ_PUBLIC_CUTOVER !== 'true') throw Error('Public cutover is not enabled');
  if (process.env.TLDRAW_DISTRIBUTION_CONFIRMED !== 'true') throw Error('Renderer distribution approval required');
  const {verifyReleaseCandidate} = await import('./release-candidate.mjs');
  const release = await verifyReleaseCandidate(directory,{publishing:true});
  if (release.sourceRevision !== reviewedSource || !/^[a-f0-9]{40}$/.test(reviewedSource ?? '')) throw Error('Reviewed source mismatch');
  // Migration validator is deliberately mandatory; no generic boolean waiver.
  if (typeof approvedMigration !== 'function') throw Error('Canonical-client migration validator not wired; activation blocked');
  await approvedMigration(release);
  const files = {...release.files, 'release.json':digest(await fs.readFile(path.join(directory,'release.json')))};
  const assets = Object.create(null);
  for (const [name,sha256] of Object.entries(files)) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name)) throw Error('Unsafe candidate asset path');
    const filename = path.join(directory,name);
    if (!(await fs.lstat(filename)).isFile()) throw Error('Asset must be a regular file');
    assets[name] = {bytes:await fs.readFile(filename),sha256};
  }
  return publishSourceAssets({...release,assets},transport);
}

// Transport performs only additive release operations; never source/catalog mutations.
export function githubTransport(directory, {command=execFileSync} = {}) {
  const run = args => command('gh',args,{encoding:'utf8',maxBuffer:1024*1024*8});
  const api = (suffix,args=[]) => JSON.parse(run(['api','--hostname','github.com',`repos/${repository}/${suffix}`,...args]));
  return {
    repository:()=>api(''), tag:tag=>api(`git/ref/tags/${tag}`),
    release:tag=> {
      try { return api(`releases/tags/${tag}`); } catch(e) { if (!/HTTP 404\b/.test(String(e.stderr))) throw e; }
      const matches=api('releases?per_page=100',['--paginate','--slurp']).flat().filter(item=>item.tag_name===tag);
      if(matches.length>1) throw Error('Duplicate release drafts for source tag');
      return matches[0] ?? null;
    },
    create:body=>api('releases',['--method','POST',...Object.entries(body).flatMap(([k,v])=>['-F',`${k}=${v}`])]),
    assets:id=>api(`releases/${id}/assets?per_page=100`,['--paginate','--slurp']).flat(),
    download:id=>command('gh',['api','--hostname','github.com',`repos/${repository}/releases/assets/${id}`,'-H','Accept: application/octet-stream'],{maxBuffer:512*1024*1024}),
    upload:async(tag,name,bytes)=> {
      const filename=path.join(directory,name);
      // Ensure CLI uploads exactly the bytes already validated, without clobber support.
      if (!(await fs.lstat(filename)).isFile()) throw Error('Upload must be a regular file');
      if (digest(await fs.readFile(filename))!==digest(bytes)) throw Error('Asset changed before upload');
      run(['release','upload',tag,filename,'--repo',repository]);
    },
    publicDownload:async(tag,name)=> {
      const response=await fetch(`https://github.com/${repository}/releases/download/${tag}/${name}`,{signal:AbortSignal.timeout(300000)});
      if(!response.ok) throw Error('Public asset not available');
      return Buffer.from(await response.arrayBuffer());
    },
    publish:id=>api(`releases/${id}`,['--method','PATCH','-F','draft=false']),
  };
}
/** Policy and versions come from the exact reviewed Git object, never bundle input. */
export async function publishUnifiedBundle(directory, sourceRoot, reviewedSource) {
  if(process.env.TENSORVIZ_PUBLIC_CUTOVER!=='true' || process.env.TLDRAW_DISTRIBUTION_CONFIRMED!=='true') throw Error('Explicit cutover and renderer approval required');
  if(!/^[a-f0-9]{40}$/.test(reviewedSource??'')) throw Error('Exact reviewed source required');
  const git=args=>execFileSync('git',args,{cwd:sourceRoot,encoding:'utf8',maxBuffer:4*1024*1024});
  if(git(['rev-parse','HEAD']).trim()!==reviewedSource || git(['status','--porcelain','--untracked-files=no']).trim()) throw Error('Clean reviewed checkout required');
  const policy=JSON.parse(git(['show',`${reviewedSource}:plugin/release/public-launch-policy.json`]));
  const {validateUnifiedSourceVersions,unifiedVersionFiles}=await import('./unified-source-versions.mjs');
  const documents=Object.fromEntries(unifiedVersionFiles.map(file=>[file,JSON.parse(git(['show',`${reviewedSource}:${file}`]))]));
  validateUnifiedSourceVersions(documents,policy.version);
  const comparison=JSON.parse(execFileSync('gh',['api','--hostname','github.com',`repos/${repository}/compare/${reviewedSource}...main`],{encoding:'utf8'}));
  if(!['ahead','identical'].includes(comparison.status)||comparison.merge_base_commit?.sha!==reviewedSource) throw Error('Reviewed source is not canonical main history');
  const {verifyUnifiedBundle}=await import('./unified-release-bundle.mjs');
  const bundle=await verifyUnifiedBundle(directory,policy);
  if(bundle.sourceRevision!==reviewedSource) throw Error('Bundle differs from reviewed source');
  // Upload private verified snapshots rather than reopening mutable source artifacts.
  const snapshot=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-source-assets-'));
  try {
    for(const [name,asset] of Object.entries(bundle.assets)) await fs.writeFile(path.join(snapshot,name),asset.bytes,{flag:'wx',mode:0o600});
    return await publishSourceAssets(bundle,githubTransport(snapshot));
  } finally { await fs.rm(snapshot,{recursive:true,force:true}); }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if(process.argv.length!==4) throw Error('Usage: publish-source-release.mjs BUNDLE_DIRECTORY SOURCE_ROOT');
  console.log(JSON.stringify(await publishUnifiedBundle(path.resolve(process.argv[2]),path.resolve(process.argv[3]),process.env.TENSORVIZ_REVIEWED_SOURCE_SHA)));
}
