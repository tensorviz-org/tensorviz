// Release-specific installed-client evidence; never a generic bypass switch.
export const migration45 = Object.freeze({from:'0.1.44',to:'0.1.45',previousExport:'6fff2d501cbe037e2d90b252ee484a446c37bd61'});
export const migration46 = Object.freeze({from:'0.1.45',to:'0.1.46',previousExport:'1b5da75b72a38c903deb7ea749eda71996284986'});
export const migration47 = Object.freeze({from:'0.1.46',to:'0.1.47',previousExport:'3d311272047e649a9ba9829c6e1b21ca95fc0655'});
export const releaseMigrations = Object.freeze({[migration45.to]:migration45,[migration46.to]:migration46,[migration47.to]:migration47});
export function verifyReleaseMigration({previous,initialMain,release,receipts}) {
  const migration=releaseMigrations[release.version];
  if(!migration) return;
  const retry=previous.version===migration.to && previous.sourceRevision===release.sourceRevision;
  if(!retry && (previous.version!==migration.from || initialMain!==migration.previousExport)) throw new Error('Release migration requires the pinned previous catalog or an identical-source retry.');
  const targets=['darwin-arm64','darwin-x64','linux-x64','linux-arm64','win32-x64','win32-arm64'];
  const entries=release.publication?.runtime?.runtimes;
  if(!entries || JSON.stringify(Object.keys(entries).sort())!==JSON.stringify(targets.sort())) throw new Error('Release requires all six validated native targets.');
  if(!Array.isArray(receipts)||receipts.length!==2) throw new Error('Both installed-client migration receipts are required.');
  for(const platform of ['darwin-arm64','linux-x64']) {
    const matches=receipts.filter(receipt=>receipt.platform===platform);
    if(matches.length!==1) throw new Error(`Exactly one migration receipt required: ${platform}`);
    const receipt=matches[0], archive=`tensorviz-runtime-${migration.to}-${platform}.tar.gz`;
    if(receipt.schemaVersion!==1 || receipt.fromVersion!==migration.from || receipt.toVersion!==migration.to
      || receipt.sourceRevision!==release.sourceRevision || !/^[a-f0-9]{40}$/.test(release.sourceRevision??'')
      || receipt.previousExport!==migration.previousExport || !/^[a-f0-9]{64}$/.test(receipt.runtimeSha256??'')
      || receipt.runtimeSha256!==release.files[archive] || receipt.runtimeSha256!==entries[platform].sha256)
      throw new Error(`Migration receipt provenance mismatch: ${platform}`);
    for(const [client,version] of Object.entries({codex:'0.165.0',claude:'2.1.308'})) {
      const evidence=receipt.clients?.[client];
      if(evidence?.version!==version || ['upgrade','rollback','nativeCapture','preservedSettings'].some(check=>evidence[check]!==true)) throw new Error(`Incomplete installed-client migration: ${platform}/${client}`);
    }
  }
}
