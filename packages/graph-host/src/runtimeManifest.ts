/** The canonical pins for the managed backend runtime — release, interpreter version,
 *  per-platform asset and SHA-256.
 *
 *  One file of data, two readers: the packaged extension downloads with it
 *  (runtimeAcquire.ts), and `ide/scripts/fetch-runtime.sh` stages the same interpreter into
 *  the fork's bundle. They cannot silently disagree because
 *  extension/src/test/unit/runtimeManifest.test.ts parses the script's pin block and asserts
 *  equality in the ordinary unit run.
 *
 *  Runtime URLs and hashes are code-reviewed release pins — never workspace configuration,
 *  never mutated at runtime, and never "updated" from a downloaded checksum sidecar. The
 *  project publishes no sidecar (checked 2026-09-05; see fetch-runtime.sh's header), so a
 *  fetched checksum would only ever prove the file we just downloaded is the file we just
 *  downloaded. */
import * as fs from 'node:fs';
import * as path from 'node:path';

export interface RuntimePlatformPin {
  readonly triple: string;
  /** Where the interpreter is inside the extracted tree, e.g. `bin/python3`. */
  readonly interpreter: string;
  readonly sha256: string;
}

export interface RuntimeManifest {
  readonly release: string;
  readonly python: string;
  readonly flavour: string;
  readonly baseUrl: string;
  /** Streaming ceiling for the archive, in bytes. A pin this size fails loudly instead of
   *  letting a hostile or broken server stream forever. */
  readonly maxDownloadBytes: number;
  readonly platforms: Readonly<Record<string, RuntimePlatformPin>>;
}

const SHA256 = /^[0-9a-f]{64}$/;

function field<T>(value: T | undefined | null | '', what: string): T {
  if (value === undefined || value === null || value === '') {
    throw new Error(`runtime-manifest.json: missing ${what}`);
  }
  return value;
}

/** Read and validate the manifest that ships beside the extension. Throws: a packaged
 *  extension with a corrupt pin file is a build failure, not a user-recoverable state. */
export function loadRuntimeManifest(extensionPath: string, manifestPath?: string): RuntimeManifest {
  const file = manifestPath ?? path.join(extensionPath, 'runtime-manifest.json');
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`runtime-manifest.json could not be read at ${file}: ${String(err)}`);
  }
  const manifest = raw as Partial<RuntimeManifest>;
  const baseUrl = field(manifest.baseUrl, 'baseUrl');
  if (!baseUrl.startsWith('https://')) {
    throw new Error(`runtime-manifest.json: baseUrl must be HTTPS, got "${baseUrl}"`);
  }
  const platforms = field(manifest.platforms, 'platforms');
  const names = Object.keys(platforms);
  if (names.length === 0) {
    throw new Error('runtime-manifest.json: platforms is empty');
  }
  for (const name of names) {
    const pin = platforms[name];
    field(pin.triple, `platforms.${name}.triple`);
    field(pin.interpreter, `platforms.${name}.interpreter`);
    if (!SHA256.test(pin.sha256 ?? '')) {
      throw new Error(`runtime-manifest.json: platforms.${name}.sha256 is not a sha256 hex pin`);
    }
  }
  const maxDownloadBytes = field(manifest.maxDownloadBytes, 'maxDownloadBytes');
  if (!Number.isInteger(maxDownloadBytes) || maxDownloadBytes <= 0) {
    throw new Error('runtime-manifest.json: maxDownloadBytes must be a positive integer');
  }
  return manifest as RuntimeManifest;
}

/** This machine's key into the manifest's platform table, or undefined when the combination
 *  is one tensorViz does not pin a runtime for. */
export function runtimePlatformKey(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string | undefined {
  const key = `${platform}-${arch}`;
  return key === 'darwin-arm64' || key === 'darwin-x64' || key === 'linux-x64' || key === 'linux-arm64' || key === 'win32-x64' || key === 'win32-arm64' ? key : undefined;
}

export type RuntimeAssetSelection =
  | { ok: true; platform: string; pin: RuntimePlatformPin; asset: string; url: string; sha256: string }
  | { ok: false; message: string };

/** Pick this machine's pinned asset, or decline clearly. The refusal is the message users on
 *  unsupported hosts should see — "tensorViz cannot set up its runtime here" — not a generic
 *  download error. */
export function selectRuntimeAsset(
  manifest: RuntimeManifest,
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): RuntimeAssetSelection {
  const key = runtimePlatformKey(platform, arch);
  const supported = Object.keys(manifest.platforms).join(', ');
  if (!key || !manifest.platforms[key]) {
    return {
      ok: false,
      message:
        `tensorViz's managed runtime is not available for ${platform}-${arch}. ` +
        `The pinned runtime is published for: ${supported}. The graph page can still open a ` +
        'source-only projection, or you can point the trace settings at your own interpreter.',
    };
  }
  const pin = manifest.platforms[key];
  const asset =
    `cpython-${manifest.python}+${manifest.release}-${pin.triple}-${manifest.flavour}.tar.gz`;
  // The '+' in the version is literal in the asset name and must be percent-encoded in the
  // URL — the same construction fetch-runtime.sh performs with sed.
  return {
    ok: true,
    platform: key,
    pin,
    asset,
    url: `${manifest.baseUrl}/${manifest.release}/${asset.replace(/\+/g, '%2B')}`,
    sha256: pin.sha256,
  };
}
