import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { executionPathSchema, type ExecutionSpec } from '@tensorviz/graph-host/executionSpec';

export class ExecutionFileError extends Error {}
export interface ExecutionFileIdentity { file: string; realPath: string; digest: string; bytes: number }

/** Resolve only explicit request files; never follow a path outside this workspace. */
export async function executionFileIdentities(workspace: string, spec: ExecutionSpec, signal?: AbortSignal): Promise<ExecutionFileIdentity[]> {
  const requested = [spec.setup && { file: spec.setup.file, max: 64 * 1024 * 1024 },
    'fixture' in spec.inputs && { file: spec.inputs.fixture, max: 64 * 1024 * 1024 },
    spec.checkpoint && { file: spec.checkpoint.file, max: 2 * 1024 * 1024 * 1024 }].filter(Boolean) as Array<{ file: string; max: number }>;
  if (!requested.length) return [];
  const root = await fs.promises.realpath(workspace);
  const identities: ExecutionFileIdentity[] = [];
  for (const request of requested) {
    executionPathSchema.parse(request.file);
    try {
      const realPath = await fs.promises.realpath(path.join(root, request.file));
      const relative = path.relative(root, realPath);
      if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error('outside workspace');
      const stat = await fs.promises.stat(realPath);
      if (!stat.isFile() || stat.size > request.max) throw new Error('size or type');
      const hash = createHash('sha256');
      let bytes = 0;
      const stream = fs.createReadStream(realPath);
      const abort = () => { stream.destroy(new Error('Execution file read cancelled.')); };
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) abort();
      try {
        for await (const chunk of stream) {
          bytes += (chunk as Buffer).length;
          if (bytes > request.max) throw new Error('size changed');
          hash.update(chunk as Buffer);
        }
      } finally {
        signal?.removeEventListener('abort', abort);
        stream.destroy();
      }
      identities.push({ file: request.file, realPath, digest: hash.digest('hex'), bytes });
    } catch {
      throw new ExecutionFileError(`Cannot read execution file ${request.file}. It must be a regular file inside this workspace and within the documented size limit.`);
    }
  }
  return identities;
}


/** Verify identities of the exact workspace bytes compiled by the worker. */
export async function verifyImportedSources(workspace: string, value: unknown,
  isDirty: (absoluteFile: string) => Promise<boolean>, signal?: AbortSignal): Promise<void> {
  if (!Array.isArray(value) || value.length === 0 || value.length > 4096) {
    throw new ExecutionFileError('Imported Python source evidence is unavailable. The run cannot validate the current source.');
  }
  const root = await fs.promises.realpath(workspace);
  let total = 0;
  for (const entry of value) {
    if (!entry || typeof entry.file !== 'string' || !/^[a-f0-9]{64}$/.test(entry.digest) ||
        !Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || (total += entry.bytes) > 64 * 1024 * 1024) {
      throw new ExecutionFileError('Imported Python source evidence is invalid or exceeds its limit.');
    }
    executionPathSchema.parse(entry.file);
    const lexical = path.join(root, entry.file);
    const real = await fs.promises.realpath(lexical);
    const relative = path.relative(root, real);
    if (path.isAbsolute(relative) || relative === '..' || relative.startsWith(`..${path.sep}`)) {
      throw new ExecutionFileError('Imported Python source is outside this workspace.');
    }
    executionPathSchema.parse(relative.split(path.sep).join('/'));
    if (real !== entry.realPath || signal?.aborted || await isDirty(lexical) || (real !== lexical && await isDirty(real))) {
      throw new ExecutionFileError('Imported Python source changed or has unsaved edits. Save it and rerun; retained evidence describes the earlier execution only.');
    }
    const stat = await fs.promises.stat(real);
    if (!stat.isFile() || stat.size !== entry.bytes) throw new ExecutionFileError('Imported Python source changed during execution. Save and rerun.');
    const hash = createHash('sha256');
    const stream = fs.createReadStream(real);
    let count = 0;
    try {
      for await (const chunk of stream) {
        count += (chunk as Buffer).length;
        if (signal?.aborted || count > entry.bytes) throw new ExecutionFileError('Imported Python source changed or verification was cancelled.');
        hash.update(chunk as Buffer);
      }
    } finally { stream.destroy(); }
    if (hash.digest('hex') !== entry.digest) throw new ExecutionFileError('Imported Python source changed during execution. Save and rerun.');
  }
}
