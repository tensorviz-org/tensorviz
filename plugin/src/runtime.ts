import { createHash } from 'node:crypto';
import { createReadStream, promises as fs } from 'node:fs';
import path from 'node:path';
import { downloadVerified, extractPythonRuntime, verifyRuntimeInterpreter } from '@tensorviz/graph-host/runtimeAcquire';
import { loadRuntimeManifest, selectRuntimeAsset } from '@tensorviz/graph-host/runtimeManifest';
import { runEnvironmentHelper } from '@tensorviz/graph-host/projectEnvironmentCore';
import { ApprovalGate } from './approval';
import { installStandaloneDependencies } from './dependencies';
import { withResourceLock } from './locks';
import { digest, HostError } from './workspace';
import { ownedTemporaryDirectory } from '@tensorviz/graph-host/ownedTemporary';

async function exists(file: string): Promise<boolean> {
  try { await fs.access(file); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error; }
}
async function checksum(file: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

/** Shared tool service and separately mutable capture environments per checkout. */
export class ManagedRuntime {
  private activity?: { signal: AbortSignal; detail?: string };
  get progress(): string | undefined { return this.activity?.signal.aborted ? undefined : this.activity?.detail; }
  constructor(private readonly backend: string, private readonly storage: string, private readonly approvals: ApprovalGate) {}
  availability(): Record<string, unknown> {
    return { available: process.platform === 'darwin' && process.arch === 'arm64', platform: `${process.platform}-${process.arch}`,
      selection: { managedRuntime: true }, detail: 'Managed CPU environments currently target macOS Apple Silicon. Provisioning uses your saved TensorViz permission for verified downloads and dependency installation.' };
  }

  async provision(root: string, signal: AbortSignal): Promise<string> {
    return this.provisionRole(root, signal, 'capture');
  }

  async provisionService(root: string, signal: AbortSignal): Promise<string> {
    return this.provisionRole(root, signal, 'service');
  }

  private async location(root: string, role: 'capture' | 'service') {
    const manifest = loadRuntimeManifest(path.dirname(this.backend));
    const asset = selectRuntimeAsset(manifest);
    if (!asset.ok) throw new HostError('MANAGED_RUNTIME_UNSUPPORTED', asset.message);
    const requirementText = await fs.readFile(path.join(this.backend, role === 'service' ? 'requirements-standalone.txt' : 'requirements-capture.txt'), 'utf8');
    const packages = requirementText.split(/\r?\n/).map(line => line.trim()).filter(line => line && !line.startsWith('#'))
      .map(line => role === 'capture' && line === 'torch' ? 'torch==2.14.1' : line);
    const key = digest(JSON.stringify([asset.sha256, packages])).slice(0, 24);
    const directory = role === 'service' ? path.join(this.storage, 'service', key) : path.join(this.storage, 'managed', digest(root), key);
    const prefix = path.join(directory, 'venv');
    const interpreter = path.join(prefix, 'bin', 'python');
    const owner = JSON.stringify({ ...(role === 'service' ? { role } : { workspaceRoot: root }), runtime: asset.sha256, requirements: digest(JSON.stringify(packages)) });
    const record = path.join(directory, 'owner.json'), ready = path.join(directory, 'ready.json');
    return { manifest, asset, packages, directory, prefix, interpreter, owner, record, ready };
  }

  async readyCapture(root: string): Promise<string | undefined> {
    if (!this.availability().available) return undefined;
    const { ready, record, owner, interpreter } = await this.location(root, 'capture');
    return await exists(ready) && await exists(record) && await fs.readFile(record, 'utf8') === owner
      && await fs.readFile(ready, 'utf8') === owner && await exists(interpreter) ? interpreter : undefined;
  }

  private async provisionRole(root: string, signal: AbortSignal, role: 'capture' | 'service'): Promise<string> {
    signal.throwIfAborted();
    const activity = { signal, detail: undefined as string | undefined };
    this.activity = activity;
    const report = (detail: string) => { if (this.activity === activity && !signal.aborted) activity.detail = detail; };
    if (!this.availability().available) throw new HostError('MANAGED_RUNTIME_UNSUPPORTED', 'Managed plugin environments currently support macOS Apple Silicon. Select a compatible existing Python on this platform.');
    const { manifest, asset, packages, directory, prefix, interpreter, owner, record, ready } = await this.location(root, role);
    if (await exists(ready) && await exists(record) && await fs.readFile(record, 'utf8') === owner
      && await fs.readFile(ready, 'utf8') === owner && await exists(interpreter)) return interpreter;
    await this.approvals.approve({ operation: 'managed_runtime', workspaceRoot: root, interpreter,
      detail: role === 'service' ? 'Prepare the TensorViz-owned tool service and installer using verified Python and the listed libraries. This environment is shared by plugin workspaces; server libraries are not added to your model Python.'
        : 'Download this pinned Python, verify its SHA-256, and create a private CPU capture environment for this exact checkout under your TensorViz permission. Verified Python downloads are shared; installed project packages are not. Existing environments and project files are preserved.',
      facts: { url: asset.url, sha256: asset.sha256, python: manifest.python, maxDownloadBytes: manifest.maxDownloadBytes,
        destination: prefix, requirements: packages } }, signal);
    try {
      return await withResourceLock(directory, signal, async check => {
        await fs.mkdir(directory, { recursive: true, mode: 0o700 });
        if (await exists(record)) {
          if (await fs.readFile(record, 'utf8') !== owner) throw new HostError('MANAGED_RUNTIME_CHANGED', 'The managed environment ownership record changed. Preserve it and select an existing environment.');
        } else await fs.writeFile(record, owner, { flag: 'wx', mode: 0o600 });
        const shared = path.join(this.storage, 'python', asset.sha256);
        const base = path.join(shared, asset.pin.interpreter);
        await withResourceLock(shared, signal, async checkDownload => {
          if (await exists(path.join(shared, 'verified.json')) && await fs.readFile(path.join(shared, 'verified.json'), 'utf8') === asset.sha256
            && await exists(base)) return;
          report('Downloading and verifying the pinned Python.');
          const archive = path.join(this.storage, 'downloads', `${asset.sha256}.tar.gz`);
          if (!await exists(archive) || (await fs.stat(archive)).size > manifest.maxDownloadBytes || await checksum(archive) !== asset.sha256) {
            const downloaded = await downloadVerified({ url: asset.url, sha256: asset.sha256, maxBytes: manifest.maxDownloadBytes,
              destPath: archive, signal, report });
            if (!downloaded.ok) throw new HostError('RUNTIME_DOWNLOAD_FAILED', downloaded.message);
          }
          check(); checkDownload();
          const temporary = await ownedTemporaryDirectory(path.join(this.storage, 'temporary'), 'tensorviz-runtime-');
          const staging = path.join(temporary.directory, 'runtime');
          try {
            const extracted = await extractPythonRuntime({ archive, destDir: staging, expectedInterpreter: asset.pin.interpreter });
            if (!extracted.ok) throw new HostError('RUNTIME_EXTRACTION_FAILED', extracted.message);
            check(); checkDownload();
            const verified = await verifyRuntimeInterpreter(path.join(staging, asset.pin.interpreter), manifest.python);
            if (!verified.ok) throw new HostError('RUNTIME_VERIFICATION_FAILED', verified.why);
            await fs.writeFile(path.join(staging, 'verified.json'), asset.sha256, { flag: 'wx', mode: 0o600 });
            if (await exists(shared)) throw new HostError('RUNTIME_CACHE_INCOMPLETE', 'An incomplete shared runtime is present. Preserve or remove that TensorViz-owned cache directory before retrying.');
            check(); checkDownload();
            await fs.mkdir(path.dirname(shared), { recursive: true, mode: 0o700 });
            await fs.rename(staging, shared);
          } finally { await temporary.dispose(); }
        });
        check();
        if (!await exists(interpreter)) {
          report(role === 'service' ? 'Creating the TensorViz tool service environment.' : 'Creating the checkout’s private capture environment.');
          // Only an incomplete prefix below this verified ownership record is replaced.
          await fs.rm(prefix, { recursive: true, force: true });
          const created = await runEnvironmentHelper(base, this.backend, ['create-environment', prefix], () => signal.aborted, true, true);
          if (!created.ok) throw new HostError('RUNTIME_CREATION_FAILED', created.message ?? 'Environment creation did not complete. Retry to resume provisioning.');
        }
        check(); report('Preparing the exact package installation plan.');
        await installStandaloneDependencies({ workspaceRoot: root, python: interpreter, prefix, backendDir: this.backend,
          storage: path.join(this.storage, 'temporary'), packages, signal, approvals: this.approvals,
          current: async () => { check(); return await fs.readFile(record, 'utf8') === owner; } });
        const inspected = await runEnvironmentHelper(interpreter, this.backend, [role === 'service' ? 'inspect' : 'inspect-capture'], () => signal.aborted, true, true);
        check();
        if (!inspected.ok || inspected.missing?.length || inspected.incompatible?.length) throw new HostError('RUNTIME_INCOMPLETE', 'The managed environment still has missing or incompatible requirements. Inspect it and resume dependency repair.');
        await fs.writeFile(ready, owner, { mode: 0o600 });
        return interpreter;
      });
    } finally { if (this.activity === activity) this.activity = undefined; }
  }
}
