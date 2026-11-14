// Prepare a reviewable additive catalog commit, never advance main or bypass reviews.
import {canonicalCatalog,catalogPrefix,catalogPaths} from './canonical-catalog.mjs';
import {createHash} from 'node:crypto';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
export async function prepareCanonicalCatalogCommit({publication,entries,expectedMain},api) {
  if(!/^[a-f0-9]{40}$/.test(expectedMain??'')) throw Error('Expected canonical main SHA required');
  if(!/^[1-9]\d*\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(publication?.version??'')) throw Error('Canonical catalog advancement is launch-only');
  const output=canonicalCatalog(publication,entries);
  const info=await api.repository();
  if(info.full_name!=='tensorviz-org/tensorviz') throw Error('Wrong canonical repository');
  const head=await api.main();
  if(head.sha!==expectedMain) throw Error('Main changed; rebuild/review catalog plan');
  if(await api.isAncestor(publication.sourceRevision,head.sha)!==true) throw Error('Catalog build source is not in canonical main history');
  const previous=await api.catalogInventory(head.sha);
  // No delete-by-omission. Removed generated paths require explicit reviewed cleanup.
  for(const file of previous) if(!Object.hasOwn(output,file)) throw Error(`Stale catalog path needs explicit review: ${file}`);
  const tree=[];
  for(const [file,value] of Object.entries(output)) {
    if(!(file.startsWith(`${catalogPrefix}/`)||catalogPaths.includes(file))) throw Error('Catalog escaped designated source namespace');
    const blob=await api.blob(value.bytes);
    if(typeof blob.sha!=='string'||!/^[a-f0-9]{40}$/.test(blob.sha)) throw Error('Invalid Git blob identity');
    tree.push({path:file,mode:value.mode,type:'blob',sha:blob.sha});
  }
  const newTree=await api.tree({base_tree:head.tree,tree});
  const commit=await api.commit({tree:newTree.sha,parents:[head.sha],message:`Prepare canonical catalog ${publication.version} [skip ci]\n\nBuild source: ${publication.sourceRevision}\nCatalog manifest SHA256: ${sha(output[`${catalogPrefix}/CANONICAL.json`].bytes)}\n`});
  if((await api.main()).sha!==expectedMain) throw Error('Main changed during preparation; do not create/merge branch');
  // Caller may open a PR on this commit. NEVER update main or source release tag here.
  return {commit:commit.sha,parent:head.sha,sourceRevision:publication.sourceRevision,files:tree.map(e=>e.path),requiresReview:true};
}
