import {distributionRepositoryForVersion,releaseVersion} from './release-version.mjs';

/** Keep historical 0.x rendering byte-identical; public source starts at 1.x. */
export function renderPublicDocument(template,version) {
  releaseVersion(version);
  let text=template.replaceAll('{{VERSION}}',version);
  if(Number(version.split('.')[0])===0) return text;
  text=text.replace(/tensorviz-org\/tensorviz-plugins(?:\/releases\/(?:download|tag)\/v[^/\s)]+)?/g, match => {
    const historical=match.match(/\/releases\/(?:download|tag)\/v(.+)$/);
    if(historical && historical[1]!==version) return match;
    return match.replace('tensorviz-org/tensorviz-plugins',distributionRepositoryForVersion(version));
  });
  text=text.replace('public releases and the installation catalog stay in `tensorviz-org/tensorviz`.',
    'source, public releases and the installation catalog are maintained in `tensorviz-org/tensorviz`.');
  text=text.replace('The public distribution repository contains only the audited integration,\nnot the private source history.',
    'The canonical repository contains TensorViz source and an audited integration catalog under\n`distribution/plugin-catalog/`. The catalog ZIP contains only the integration export.');
  text=text.replace('directory and use that absolute path instead. Never register the private source\nrepository as a marketplace.',
    'directory and use that absolute path instead. Do not substitute GitHub\'s source ZIP for the complete catalog ZIP.');
  text=text.replace('in the public distribution repository. Tags identify audited export commits;\nPUBLICATION.json binds each export to its reviewed private source SHA. It never\nreplaces source main or copies private Git history.',
    'in the canonical source repository. Release tags identify the reviewed source commit;\nPUBLICATION.json binds the generated integration to that build source. A subsequent\ncatalog commit may reference it without changing the immutable source release tag.\nThe catalog is a non-destructive overlay, never a replacement of source main.');
  text=text.replace('The public distribution repository and its version tags contain the audited\nintegration catalog. The private `tensorviz` source repository is not an\ninstallation catalog. The catalog ZIP provides the same complete export.',
    'The canonical repository exposes root client indexes pointing to the audited\n`distribution/plugin-catalog/` tree. Its release tags identify reviewed source,\nnot a distribution-only export. Use the complete catalog ZIP for the local\ninstallation commands below, not a GitHub source archive.');
  return text;
}
