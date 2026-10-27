/** Saved binary dependencies participate in review without becoming text documents. */
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { executionPathSchema } from '@tensorviz/graph-host/executionSpec';

type Resource = { path: string; digest: string };
const MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024;
const MAX_TOTAL_BYTES = 2144 * 1024 * 1024;
const sameObject = (a: fs.BigIntStats, b: fs.BigIntStats) => a.dev === b.dev && a.ino === b.ino && a.mode === b.mode;
const sameFile = (a: fs.BigIntStats, b: fs.BigIntStats) => sameObject(a, b) && a.size === b.size
  && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs;

export class ReviewResourceError extends Error {
  constructor() { super('A tracked resource is changed, unsaved, unavailable or outside the review limits. Validate the saved setup again.'); }
}

/** Hash once, then pin physical identity and change timestamps through the final
 * synchronous writer guard. No deserialization, workspace import or resource write. */
export async function reviewResourceGuard(workspace: string, resources: Resource[], signal: AbortSignal,
  isDirty: (absolute: string) => boolean): Promise<() => boolean> {
  const requested = resources.map(item => ({ ...item }));
  if (!requested.length) return () => !signal.aborted;
  try {
    if (requested.length > 128 || new Set(requested.map(item => item.path)).size !== requested.length) throw new Error();
    const root = await fs.promises.realpath(workspace);
    const directories = new Map<string, fs.BigIntStats>();
    const files = new Map<string, fs.BigIntStats>();
    const dirty = (file: string) => isDirty(path.join(root, file)) || isDirty(path.join(workspace, file));
    const current = () => {
      try {
        return !signal.aborted && fs.realpathSync(workspace) === root
          && [...directories].every(([file, before]) => {
            const now = fs.lstatSync(file, { bigint: true });
            return now.isDirectory() && sameObject(before, now);
          }) && requested.every(item => {
            const before = files.get(item.path), now = fs.lstatSync(path.join(root, item.path), { bigint: true });
            return !!before && !dirty(item.path) && now.isFile() && sameFile(before, now);
          });
      } catch { return false; }
    };
    let total = 0;
    const buffer = Buffer.allocUnsafe(64 * 1024);
    for (const item of requested) {
      executionPathSchema.parse(item.path);
      if (!/^[a-f0-9]{64}$/.test(item.digest) || signal.aborted || dirty(item.path)) throw new Error();
      let directory = root;
      for (const segment of ['', ...item.path.split('/').slice(0, -1)]) {
        directory = path.join(directory, segment);
        const info = await fs.promises.lstat(directory, { bigint: true });
        if (!info.isDirectory() || directories.has(directory) && !sameObject(directories.get(directory)!, info)) throw new Error();
        directories.set(directory, info);
      }
      const absolute = path.join(root, item.path);
      const info = await fs.promises.lstat(absolute, { bigint: true });
      if (!info.isFile() || info.size > BigInt(MAX_FILE_BYTES) || total + Number(info.size) > MAX_TOTAL_BYTES) throw new Error();
      // A file swapped for a FIFO between lstat and open must not block review.
      const handle = await fs.promises.open(absolute, fs.constants.O_RDONLY
        | (fs.constants.O_NOFOLLOW ?? 0) | (fs.constants.O_NONBLOCK ?? 0));
      try {
        if (!sameFile(info, await handle.stat({ bigint: true }))) throw new Error();
        const hash = createHash('sha256');
        let bytes = 0;
        for (;;) {
          if (signal.aborted || dirty(item.path)) throw new Error();
          const read = await handle.read(buffer, 0, buffer.length, null);
          if (!read.bytesRead) break;
          bytes += read.bytesRead; total += read.bytesRead;
          if (bytes > MAX_FILE_BYTES || total > MAX_TOTAL_BYTES || bytes > Number(info.size)) throw new Error();
          hash.update(buffer.subarray(0, read.bytesRead));
        }
        if (bytes !== Number(info.size) || hash.digest('hex') !== item.digest
          || !sameFile(info, await handle.stat({ bigint: true }))) throw new Error();
      } finally { await handle.close(); }
      files.set(item.path, info);
    }
    if (!current()) throw new Error();
    return current;
  } catch { throw new ReviewResourceError(); }
}
