/** The workspace Python source set behind the multi-file projection
 *  (CURRENT_TASK.md "P0 Reliable Multi-File Model Graphs", step 4).
 *
 * The contract splits the job exactly here: the BACKEND never gains workspace
 * file access for this feature, so the extension enumerates the candidate set,
 * reads every entry the way the user is looking at it — an open dirty buffer
 * wins over whatever is on disk — digests each, and posts the lot to
 * `/api/ide/multifile/project`. This module owns the rules worth testing and
 * the thin VS Code adapter:
 *
 *  - which files count (every `.py` under the target's folder that is not
 *    `.git`, `__pycache__`, a virtualenv, or `node_modules`);
 *  - the budgets are the backend's own ceilings (services/ide/multifile.py):
 *    at most `maxSources`, each under `maxSourceBytes`, totalling under
 *    `maxTotalBytes`. Hitting any of them is HONEST: the request goes out
 *    with `truncated: true` and the backend echoes the admission as a
 *    SOURCE_SET_TRUNCATED diagnostic, so a partial set never looks complete;
 *  - the root is always the first entry and never budgeted out: without it
 *    the request is refused outright, so it is the one file that MUST be in;
 *  - digests are sha256 of exactly the text sent; entries keep their VS Code
 *    document version when the text came from an open buffer.
 *
 * Read-only promise: `list` enumerates, `read` reads; nothing here writes.
 */

import { createHash } from 'node:crypto';
import * as path from 'node:path';

import { IDE_PROJECTION_PROTOCOL_VERSION } from '@tensorviz/contracts/protocol';
import type { WorkerEnvelope } from './traceWorkerCore';

/** The backend's own ceilings; posting past them is a coded refusal there. */
export const SOURCE_SET_LIMITS = {
  maxSources: 256,
  maxSourceBytes: 1_000_000,
  maxTotalBytes: 8_000_000,
} as const;

/** How far enumeration asks beyond the budget, so "there was more" is VISIBLE
 *  (truncated) instead of silently becoming "the first N paths on disk". */
export const ENUMERATION_ASK = SOURCE_SET_LIMITS.maxSources * 4 + 1;

export interface SourceEntry {
  /** Normalised workspace-relative POSIX path. */
  file: string;
  source: string;
  /** sha256 of `source`, lowercase hex. */
  digest: string;
  /** VS Code document version — present only when a live buffer supplied the text. */
  document_version?: number;
}

export interface SourceSet {
  /** Root first, local import hints next, then remaining candidates sorted by path. */
  entries: SourceEntry[];
  /** True when ANY candidate was skipped by a budget or enumeration was capped. */
  truncated: boolean;
}

/** The host answers this module needs, injectable so the rules are testable
 *  without an editor (the VS Code adapter below is the production one, a
 *  deliberate mirror of extension.ts's `vscodeSnapshotSource`). */
export interface SourceIO {
  /** Every candidate's workspace-relative POSIX path. */
  list(): Promise<string[]>;
  /** The text of a path `list` returned. Never called for a file an open
   *  buffer has: `openDocument` answers first. */
  read(relativePath: string): Promise<string>;
  /** The open document for a path, or undefined when the file is on disk only. */
  openDocument(relativePath: string): { getText(): string; readonly version: number } | undefined;
}

export function sha256Text(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

export function isNormalizedPythonPath(relativePath: string): boolean {
  if (
    relativePath === '' ||
    relativePath.includes('\\') ||
    relativePath.startsWith('/') ||
    relativePath.startsWith('~') ||
    /^[A-Za-z]:/.test(relativePath) ||
    !relativePath.endsWith('.py')
  ) {
    return false;
  }
  return relativePath
    .split('/')
    .every((segment) => segment !== '' && segment !== '.' && segment !== '..');
}

/** Read-only admission hints, not Python binding resolution. Only paths in the
 * supplied inventory can gain priority; the backend still parses and resolves
 * the actual bindings. Dynamic imports and unsupported spellings stay unresolved. */
function localImportPaths(source: string, file: string, inventory: ReadonlySet<string>): string[] {
  const found = new Set<string>();
  const add = (module: string) => {
    const dots = module.match(/^\.+/)?.[0].length ?? 0;
    const parent = file.split('/').slice(0, -1);
    if (dots > parent.length + 1) return undefined;
    const parts = dots ? [...parent.slice(0, parent.length - dots + 1), ...module.slice(dots).split('.').filter(Boolean)] : module.split('.');
    const prefix = parts.join('/');
    for (const candidate of [`${prefix}.py`, `${prefix}/__init__.py`]) if (inventory.has(candidate)) found.add(candidate);
    for (let i = 1; i < parts.length; i++) {
      const init = `${parts.slice(0, i).join('/')}/__init__.py`;
      if (inventory.has(init)) found.add(init);
    }
    return prefix;
  };
  // Ignore comments and string bodies so documentation cannot consume the
  // dependency budget. These are conservative import hints, never an AST claim.
  const code = source.replace(/"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*/g,
    value => value.replace(/[^\n]/g, ' '));
  for (const match of code.matchAll(/^[ \t]*(?:from[ \t]+([.\w]+)[ \t]+import[ \t]+([^\n]*)|import[ \t]+([^\n]+))/gm)) {
    if (match[1]) {
      const prefix = add(match[1]);
      if (prefix !== undefined) for (const item of match[2].split(',')) {
        const name = item.trim().split(/\s+as\s+/)[0];
        if (/^[A-Za-z_]\w*$/.test(name)) add(`${prefix.replaceAll('/', '.')}.${name}`);
      }
    } else for (const item of match[3].split(',')) {
      const module = item.trim().split(/\s+as\s+/)[0];
      if (/^[A-Za-z_]\w*(?:\.\w+)*$/.test(module)) add(module);
    }
  }
  return [...found].sort();
}

/** Assemble the request's source set: root and its local imports first, budgets
 *  enforced with the truncation admission. Deterministic output — the same
 *  workspace reads to the same body. */
export async function collectSourceSet(rootRel: string, io: SourceIO): Promise<SourceSet> {
  if (!isNormalizedPythonPath(rootRel)) {
    throw new Error(`invalid workspace-relative Python root: ${rootRel}`);
  }
  const listed = await io.list();
  const seen = new Set<string>();
  const candidates: string[] = [];
  let truncated = listed.length >= ENUMERATION_ASK;
  for (const rel of listed) {
    if (!isNormalizedPythonPath(rel)) {
      truncated = true;
      continue;
    }
    if (rel === rootRel || seen.has(rel)) {
      continue;
    }
    seen.add(rel);
    candidates.push(rel);
  }
  candidates.sort();
  const inventory = new Set([rootRel, ...candidates]);
  const ordered = [rootRel], scheduled = new Set(ordered);
  let nextCandidate = 0;

  const entries: SourceEntry[] = [];
  let totalBytes = 0;
  while (ordered.length || nextCandidate < candidates.length) {
    if (!ordered.length) {
      const next = candidates[nextCandidate++];
      if (scheduled.has(next)) continue;
      scheduled.add(next); ordered.push(next);
    }
    const rel = ordered.shift()!;
    const isRoot = rel === rootRel;
    if (!isRoot) {
      if (entries.length >= SOURCE_SET_LIMITS.maxSources) {
        truncated = true;
        break;
      }
    }
    // An open buffer is the user's file; the disk answer is for files nobody
    // is looking at — the same rule the root itself follows.
    let open: ReturnType<SourceIO['openDocument']>;
    let text: string;
    try {
      open = io.openDocument(rel);
      text = open ? open.getText() : await io.read(rel);
    } catch (error) {
      if (isRoot) {
        throw error;
      }
      // A candidate can disappear after findFiles returns. Keep the useful
      // projection and admit that its source set was incomplete.
      truncated = true;
      continue;
    }
    const bytes = Buffer.byteLength(text, 'utf8');
    if (!isRoot) {
      if (bytes > SOURCE_SET_LIMITS.maxSourceBytes) {
        truncated = true;
        continue;
      }
      if (totalBytes + bytes > SOURCE_SET_LIMITS.maxTotalBytes) {
        truncated = true;
        break;
      }
    }
    totalBytes += bytes;
    entries.push({
      file: rel,
      source: text,
      digest: sha256Text(text),
      ...(open ? { document_version: open.version } : {}),
    });
    for (const dependency of localImportPaths(text, rel, inventory)) if (!scheduled.has(dependency)) {
      scheduled.add(dependency); ordered.push(dependency);
    }
  }
  return { entries, truncated };
}

/** The body `POST /api/ide/multifile/project` takes, exactly as
 *  backend/services/ide/multifile.py's `MultifileProjectionRequest` declares it
 *  (docs/ide-multifile-projection-contract.md). */
export interface MultifileProjectionBody {
  protocol_version: number;
  root: {
    file: string;
    cls: string;
    source_digest: string;
    document_version?: number;
    measured_input_shape?: number[];
  };
  sources: SourceEntry[];
  truncated: boolean;
  trace_result?: WorkerEnvelope;
}

/**
 * Assemble one projection request: the whole candidate source set, rooted at the
 * class being drawn.
 *
 * ONE builder for both surfaces, deliberately. The embedded canvas and the local
 * browser page draw the same class out of the same workspace, and the second copy
 * of these six fields is precisely how they drifted: the canvas spent a release
 * posting to the single-buffer route while the page used this one, so the same
 * model at the same moment drew twelve layers in the editor and sixteen in the
 * browser. A field added here reaches both, or neither.
 *
 * `source` is the root buffer as the caller is looking at it — the collector
 * re-reads it through `io` (an open document wins over disk), so what is digested
 * here and what is sent are the same text.
 */
export async function multifileProjectionBody(options: {
  io: SourceIO;
  /** Workspace-relative POSIX path of the class's own file. */
  file: string;
  cls: string;
  source: string;
  /** VS Code's version for the root buffer, when one is open. */
  documentVersion?: number;
  /** What a previous trace measured this class's input to be. The backend ranks it
   *  BELOW `# tensorviz: input=`, so it is a hint and never an override. */
  measuredInputShape?: number[];
  /** The executed tier's worker envelope; absent leaves the request on the AST tier. */
  trace?: WorkerEnvelope;
}): Promise<MultifileProjectionBody> {
  const sources = await collectSourceSet(options.file, options.io);
  return {
    protocol_version: IDE_PROJECTION_PROTOCOL_VERSION,
    root: {
      file: options.file,
      cls: options.cls,
      source_digest: sha256Text(options.source),
      ...(options.documentVersion !== undefined
        ? { document_version: options.documentVersion }
        : {}),
      ...(options.measuredInputShape ? { measured_input_shape: options.measuredInputShape } : {}),
    },
    sources: sources.entries,
    truncated: sources.truncated,
    ...(options.trace ? { trace_result: options.trace } : {}),
  };
}

/** `fsPath` as a workspace-relative POSIX path under `folder`, or undefined when it
 *  is not inside it — the spelling every source entry, span and dependency uses, so
 *  "is this edit part of the graph on screen" is one comparison and not three. */
export function relativeInside(folder: string, fsPath: string): string | undefined {
  const rel = path.relative(folder, fsPath);
  if (rel === '' || rel.startsWith('..') || path.isAbsolute(rel)) {
    return undefined;
  }
  return rel.split(path.sep).join('/');
}

