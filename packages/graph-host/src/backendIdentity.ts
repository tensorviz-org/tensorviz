/** Content identity of the Python backend a worker runs (execution cache, F1).
 *
 * The worker's envelope is produced by the backend's capture code, so a cached result
 * is only valid for the exact backend bytes that produced it. A staged backend (the
 * VSIX) carries `backend-identity.json`, written by `tooling/runtime/stage-backend.mjs`
 * at package time. A development checkout has no such file; its identity is the hash of
 * every `*.py` under the backend directory, computed once per process and directory.
 * Either way, any backend edit gives a different digest. */
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import * as path from 'node:path';

export const BACKEND_IDENTITY_FILE = 'backend-identity.json';

/** Directories never part of the running backend in a development checkout: the ones
 *  `runtime/backend-profiles.json` keeps out of every staged backend (tests, evals,
 *  scripts, reports, caches), plus virtualenvs. The worker never imports them. */
const SKIPPED = new Set(['tests', 'evals', 'scripts', 'reports', '__pycache__', '.runtime', '.pytest_cache', '.mypy_cache',
  'node_modules', '.git', '.venv', 'venv']);

/** sha256 over sorted `relpath \0 sha256(bytes) \n` lines. Shared with the stage script's
 *  format so the meaning of a digest does not depend on who computed it. */
export function digestFiles(root: string, files: string[]): string {
  const hash = createHash('sha256');
  for (const relative of [...files].sort()) {
    const content = createHash('sha256').update(readFileSync(path.join(root, relative))).digest('hex');
    hash.update(`${relative.split(path.sep).join('/')}\0${content}\n`);
  }
  return hash.digest('hex');
}

/** Every `*.py` below `root`, relative, skipping the directories above and dot directories. */
export function backendPythonFiles(root: string): string[] {
  const found: string[] = [];
  const walk = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (!SKIPPED.has(entry.name) && !entry.name.startsWith('.')) walk(path.join(directory, entry.name));
      } else if (entry.isFile() && entry.name.endsWith('.py')) {
        found.push(path.relative(root, path.join(directory, entry.name)));
      }
    }
  };
  walk(root);
  return found;
}

const staged = new Map<string, string>();
const source = new Map<string, { signature: string; digest: string }>();

/** `staged:<digest>` from the shipped identity file, or `source:<digest>` for a checkout.
 *
 * Cached per process. A staged backend is immutable (it is replaced only by an extension
 * update, which restarts the host), so its identity is read once. A checkout can be
 * edited while the host runs, so its cached digest is reused only while the file list,
 * sizes and mtimes are unchanged (a stat pass, ~1-3 ms); otherwise it is re-hashed. */
export function backendDigest(backendDir: string): string {
  const known = staged.get(backendDir);
  if (known) return known;
  try {
    const identity = JSON.parse(readFileSync(path.join(backendDir, BACKEND_IDENTITY_FILE), 'utf8')) as { format?: number; digest?: string };
    if (identity.format === 1 && typeof identity.digest === 'string' && /^[a-f0-9]{64}$/.test(identity.digest)) {
      const digest = `staged:${identity.digest}`;
      staged.set(backendDir, digest);
      return digest;
    }
  } catch { /* development checkout: no identity file */ }
  const files = backendPythonFiles(backendDir).sort();
  const signature = createHash('sha256');
  for (const relative of files) {
    const stat = statSync(path.join(backendDir, relative));
    signature.update(`${relative}\0${stat.size}\0${stat.mtimeMs}\n`);
  }
  const key = signature.digest('hex');
  const cached = source.get(backendDir);
  if (cached?.signature === key) return cached.digest;
  const digest = `source:${digestFiles(backendDir, files)}`;
  source.set(backendDir, { signature: key, digest });
  return digest;
}

/** Tests only: forget cached digests so an edited fixture backend is re-hashed. */
export function resetBackendDigestCache(): void { staged.clear(); source.clear(); }
