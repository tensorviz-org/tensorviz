import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import lockfile from 'proper-lockfile';
import { digest, HostError } from './workspace';

/** All hosts of this OS user serialize mutations of the same canonical resource. */
export async function withResourceLock<T>(resource: string, signal: AbortSignal, action: (check: () => void) => Promise<T>): Promise<T> {
  signal.throwIfAborted();
  const directory = path.join(os.tmpdir(), `tensorviz-locks-${process.getuid?.() ?? os.userInfo().username}`);
  await fs.mkdir(directory, { recursive: true, mode: 0o700 });
  const stat = await fs.lstat(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new HostError('LOCK_UNAVAILABLE', 'The TensorViz resource lock directory is unavailable.');
  let compromised = false;
  let release: () => Promise<void>;
  try {
    release = await lockfile.lock(path.join(directory, digest(resource)), {
      realpath: false, stale: 30_000, update: 10_000, retries: 0,
      onCompromised: () => { compromised = true; },
    });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ELOCKED') throw new HostError('RESOURCE_BUSY', 'Another TensorViz host is changing this resource. Retry after it finishes; a crashed owner expires within 30 seconds.');
    throw error;
  }
  const check = () => {
    signal.throwIfAborted();
    if (compromised) throw new HostError('LOCK_CHANGED', 'The resource lock changed. Inspect the current files before retrying.');
  };
  try { check(); return await action(check); }
  finally { await release(); }
}
