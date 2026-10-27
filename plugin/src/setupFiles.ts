import { constants, promises as fs } from 'node:fs';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import type { PreparedGraphSetup } from '@tensorviz/graph-host/graphOpening';
import { digest, HostError, relativeInside } from './workspace';
import { withResourceLock } from './locks';

export interface SetupSnapshot { text: string; identity: string | null }
const changed = () => new HostError('SETUP_CHANGED', 'Setup files changed while this operation was being prepared. Reopen setup to preserve the current files.');

async function destination(root: string, file: string): Promise<string> {
  if (file !== 'tensorviz.toml' && !/^\.tensorviz\/setup_[A-Za-z_][A-Za-z0-9_]*_[a-f0-9]{12}\.py$/.test(file)) {
    throw new HostError('INVALID_DESTINATION', 'Only tensorviz.toml and a generated setup helper can be written.');
  }
  let absolute = root;
  const parts = file.split('/');
  for (const [index, part] of parts.entries()) {
    absolute = path.join(absolute, part);
    try {
      const stat = await fs.lstat(absolute);
      if (stat.isSymbolicLink() || (index === parts.length - 1 ? !stat.isFile() || stat.nlink !== 1 : !stat.isDirectory())) {
        throw new HostError('INVALID_DESTINATION', 'Setup paths must be ordinary workspace files without symbolic or hard links.');
      }
      relativeInside(root, await fs.realpath(absolute));
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  }
  return absolute;
}

export async function readSetup(root: string, file = 'tensorviz.toml'): Promise<SetupSnapshot> {
  const absolute = await destination(root, file);
  let handle;
  try { handle = await fs.open(absolute, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0)); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { text: '', identity: null }; throw error; }
  try {
    const before = await handle.stat();
    if (before.size > 262144 || !before.isFile() || before.nlink !== 1) throw new HostError('SETUP_LIMIT', 'The setup file is not an ordinary bounded file.');
    const buffer = Buffer.alloc(262145);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    const text = new TextDecoder('utf-8', { fatal: true }).decode(buffer.subarray(0, bytesRead));
    const after = await fs.lstat(await destination(root, file));
    const identity = (stat: typeof before) => `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`;
    if (bytesRead > 262144 || before.size !== bytesRead || identity(before) !== identity(after)) throw changed();
    return { text, identity: identity(after) };
  } finally { await handle.close(); }
}

export const sameSetup = (a: SetupSnapshot, b: SetupSnapshot): boolean => a.identity === b.identity && a.text === b.text;

/** Atomic config publication, after user approval and final source/config checks. */
export async function saveSetup(root: string, before: SetupSnapshot, prepared: PreparedGraphSetup,
  signal: AbortSignal, current: () => Promise<boolean>): Promise<void> {
  await withResourceLock(path.join(root, 'tensorviz.toml'), signal, async check => {
    const checkCurrent = async () => {
      check();
      if (!await current() || !sameSetup(before, await readSetup(root))) throw changed();
      check();
    };
    await checkCurrent();
    if (prepared.factory) {
      const file = await destination(root, prepared.factory.file);
      const previous = await readSetup(root, prepared.factory.file);
      if (previous.identity !== null && previous.text !== prepared.factory.source) {
        throw new HostError('HELPER_CHANGED', 'The generated helper was edited. Preserve it and select it as an existing factory.');
      }
      if (previous.identity === null) {
        await fs.mkdir(path.dirname(file), { recursive: true });
        await destination(root, prepared.factory.file);
        await checkCurrent();
        await fs.writeFile(file, prepared.factory.source, { flag: 'wx', mode: 0o644 });
      }
    }
    const file = await destination(root, 'tensorviz.toml');
    const temporary = path.join(root, `.tensorviz-setup-${randomUUID()}.tmp`);
    try {
      const mode = before.identity === null ? 0o644 : (await fs.stat(file)).mode & 0o777;
      await fs.writeFile(temporary, prepared.config, { flag: 'wx', mode });
      await checkCurrent();
      await fs.rename(temporary, file);
      const written = await readSetup(root);
      if (digest(written.text) !== digest(prepared.config)) throw changed();
    } finally { await fs.rm(temporary, { force: true }); }
  });
}
