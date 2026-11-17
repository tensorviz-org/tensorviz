// Channel adapter contract: authenticated publishing is deliberately kept outside
// this pure orchestration module. No generic retry may overwrite published bytes.
import {setTimeout as wait} from 'node:timers/promises';
export async function releaseChannels({version,sourceRevision,vsixSha256,docsSha256},channels,{sleep=wait}={}) {
  if(!/^[1-9]\d*\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)||!/^[a-f0-9]{40}$/.test(sourceRevision)||(!/^[a-f0-9]{64}$/.test(vsixSha256)||!/^[a-f0-9]{64}$/.test(docsSha256))) throw Error('Exact unified release identity required');
  const names=['github','vscode-marketplace','open-vsx','docs'];
  if(JSON.stringify(Object.keys(channels).sort())!==JSON.stringify(names.sort())) throw Error('Every release channel requires an explicit adapter');
  const results={};
  // GitHub source artifacts must be publicly verified before marketplaces advertise.
  for(const name of ['github','vscode-marketplace','open-vsx','docs']) {
    const adapter=channels[name];
    try {
      let receipt=await adapter.inspect(version);
      if(!receipt) {
        await adapter.publish({version,sourceRevision,vsixSha256,docsSha256});
        // Registries may accept an upload before their public read API sees it.
        // Poll absence up to thirteen times with 60 seconds of total backoff.
        // Never repeat a submission, exception, or conflicting-content check.
        for(let attempt=0;attempt<13;attempt++) {
          receipt=await adapter.inspect(version);
          if(receipt!==null||attempt===12) break;
          await sleep(5000);
        }
      }
      if(!receipt||receipt.version!==version||receipt.sourceRevision!==sourceRevision||receipt.verified!==true
        || (['vscode-marketplace','open-vsx'].includes(name)&&receipt.vsixSha256!==vsixSha256)||(name==='docs'&&receipt.docsSha256!==docsSha256)) throw Error('Published channel provenance/content not verified');
      results[name]={status:'verified',receipt};
    } catch(error) {
      results[name]={status:'blocked',error:String(error.message??error)};
      // Avoid advertising a release whose canonical artifacts are unavailable.
      if(name==='github') break;
    }
  }
  return {version,sourceRevision,complete:names.every(name=>results[name]?.status==='verified'),channels:results};
}
