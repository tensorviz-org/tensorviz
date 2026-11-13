// One-time version-series transition. This is not a general downgrade switch.
export const alignment = Object.freeze({
  from: '0.3.110', to: '0.1.43',
  previousExport: '1a3c3618082603030b5ef68b793d9834b8bb7f3d',
});
export function verifyAlignment({ previous, initialMain, release, extensionVersion, receipts }) {
  if (previous.version !== alignment.from || release.version !== alignment.to
      || initialMain !== alignment.previousExport || extensionVersion !== alignment.to) {
    throw new Error('Refusing to roll the public catalog back to an older release. Only the exact audited version-series transition is eligible.');
  }
  if (!Array.isArray(receipts) || receipts.length !== 2) throw new Error('Both native migration receipts are required.');
  for (const platform of ['darwin-arm64', 'linux-x64']) {
    const matches = receipts.filter(receipt => receipt.platform === platform);
    if (matches.length !== 1) throw new Error(`Exactly one migration receipt is required for ${platform}.`);
    const receipt = matches[0];
    const archive = `tensorviz-runtime-${alignment.to}-${platform}.tar.gz`;
    if (receipt.schemaVersion !== 1 || receipt.fromVersion !== alignment.from || receipt.toVersion !== alignment.to
        || receipt.sourceRevision !== release.sourceRevision || receipt.previousExport !== initialMain
        || !/^[a-f0-9]{64}$/.test(receipt.runtimeSha256 ?? '') || receipt.runtimeSha256 !== release.files[archive]) {
      throw new Error(`Migration receipt provenance mismatch: ${platform}.`);
    }
    for (const [client, version] of Object.entries({ codex: '0.165.0', claude: '2.1.308' })) {
      const evidence = receipt.clients?.[client];
      if (evidence?.version !== version || ['upgrade', 'rollback', 'nativeCapture', 'preservedSettings'].some(check => evidence[check] !== true)) {
        throw new Error(`Incomplete installed-client migration: ${platform}/${client}.`);
      }
    }
  }
}
