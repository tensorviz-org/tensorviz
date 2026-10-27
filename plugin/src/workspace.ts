import { createHash } from 'node:crypto';
import { constants, promises as fs } from 'node:fs';
import path from 'node:path';
import { ENUMERATION_ASK, SOURCE_SET_LIMITS, type SourceIO } from '@tensorviz/graph-host';
import { GraphOpeningError } from '@tensorviz/graph-host/graphOpening';

export class HostError extends GraphOpeningError {
  constructor(readonly code: string, message: string) { super(message); }
}

export const digest = (value: string | Buffer): string => createHash('sha256').update(value).digest('hex');
const excluded = new Set(['.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__', '.tensorviz']);

/** Identity is the actual checkout directory, never the launcher's cwd or a branch. */
export async function workspaceRoot(value?: string): Promise<string> {
  if (!value || !path.isAbsolute(value)) throw new HostError('WORKSPACE_REQUIRED', 'Provide the absolute path of the exact project or Git worktree in workspaceRoot.');
  const root = await fs.realpath(value).catch(() => undefined);
  if (!root || !(await fs.stat(root)).isDirectory()) throw new HostError('WORKSPACE_UNAVAILABLE', 'The selected workspace directory is unavailable.');
  return root;
}

export function relativeInside(root: string, file: string): string {
  const relative = path.relative(root, file);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new HostError('OUTSIDE_WORKSPACE', 'Select a file inside this exact workspace.');
  }
  return relative.split(path.sep).join('/');
}

export async function savedFile(root: string, value: string): Promise<{ file: string; text: string; digest: string }> {
  const requested = path.resolve(root, value);
  relativeInside(root, requested);
  const actual = await fs.realpath(requested);
  const file = relativeInside(root, actual);
  if (!file.endsWith('.py')) throw new HostError('PYTHON_FILE_REQUIRED', 'Select a saved Python file.');
  const handle = await fs.open(actual, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    const before = await handle.stat();
    if (!before.isFile() || before.size > SOURCE_SET_LIMITS.maxSourceBytes) throw new HostError('SOURCE_LIMIT', 'The selected source exceeds the supported file limit.');
    const buffer = Buffer.alloc(SOURCE_SET_LIMITS.maxSourceBytes + 1);
    let count = 0;
    while (count < buffer.length) {
      const read = await handle.read(buffer, count, buffer.length - count, null);
      if (!read.bytesRead) break;
      count += read.bytesRead;
    }
    const bytes = buffer.subarray(0, count);
    const after = await handle.stat();
    if (bytes.length > SOURCE_SET_LIMITS.maxSourceBytes || before.size !== after.size || before.mtimeMs !== after.mtimeMs
      || await fs.realpath(requested) !== actual) throw new HostError('SOURCE_CHANGED', 'Source changed while being read. Retry after saving.');
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    return { file, text, digest: digest(bytes) };
  } finally { await handle.close(); }
}

/** Enumeration and reads share the engine's bounds; symlink directories are not followed. */
export function sourceIO(root: string): SourceIO {
  return {
    openDocument: () => undefined,
    read: async file => (await savedFile(root, file)).text,
    list: async () => {
      const files: string[] = [], pending = [''];
      let examined = 0;
      while (pending.length && files.length < ENUMERATION_ASK && examined < 20_000) {
        const relative = pending.pop()!;
        const directory = await fs.opendir(path.join(root, relative));
        for await (const entry of directory) {
          if (++examined >= 20_000 || files.length >= ENUMERATION_ASK) break;
          if (excluded.has(entry.name) || entry.isSymbolicLink()) continue;
          const file = relative ? `${relative}/${entry.name}` : entry.name;
          if (entry.isDirectory()) pending.push(file);
          else if (entry.isFile() && entry.name.endsWith('.py')) files.push(file);
        }
      }
      // The core treats an invalid candidate as an explicit truncation admission.
      if (pending.length || examined >= 20_000) files.push('SOURCE_ENUMERATION_TRUNCATED');
      return files;
    },
  };
}
