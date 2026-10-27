import { readFileSync } from 'node:fs';

export const distributionRepository = 'tensorviz-org/tensorviz';

export function releaseVersion(base, runNumber) {
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(base)) throw new Error('Invalid plugin version.');
  if (runNumber === undefined) return base;
  if (!/^[1-9]\d*$/.test(String(runNumber)) || !Number.isSafeInteger(Number(runNumber))) throw new Error('Invalid release run number.');
  const [major, minor, patch] = base.split('.');
  const next = Number(patch) + Number(runNumber);
  if (!Number.isSafeInteger(next)) throw new Error('Release version overflow.');
  return `${major}.${minor}.${next}`;
}

export function pluginVersion() {
  const { version } = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  return releaseVersion(process.env.TENSORVIZ_RELEASE_VERSION ?? version);
}
