import { isHostFile, matchesHostFile } from './workspaceFiles';
/** The project-wide Python source set, and the manifest/bodies handshake.
 *
 * `workspaceSources.ts` enumerates one model's dependency closure and posts every
 * byte of it. That is right for one class and wrong for a project twice over: the
 * backend's multi-file endpoint caps admission at 256 entries, and re-uploading the
 * tree on every keystroke is the dominant cost once the tree is a repository.
 *
 * So this module speaks a different shape to `/api/ide/project/index`:
 *
 *  - a **manifest** of `{file, digest}` for every admitted file — the complete
 *    picture, cheaply, so the backend can tell an unreached file from an absent one;
 *  - **bodies** only for digests the backend has not already seen. It answers
 *    `need_bodies` with the paths it lacks, we send exactly those, and after the
 *    first index a keystroke ships one body.
 *
 * Two rules it keeps that the older enumerator gets wrong:
 *
 *  - **`.gitignore` is honoured.** `vscode.workspace.findFiles` never applies it —
 *    `search.useIgnoreFiles` governs the search UI, not this API — and passing an
 *    explicit `exclude` additionally overrides `files.exclude`. Ignore files are
 *    therefore read and applied here, directly. Without it, indexing a repo with a
 *    `data/` or `wandb/` directory means enumerating thousands of irrelevant files.
 *  - **Truncation is admitted, never hidden.** Hitting the ceiling sets `truncated`,
 *    which reaches the index and the banner, so a partial project never looks whole.
 *
 * Read-only: this module enumerates, reads and digests. It writes nothing.
 */

import { createHash } from 'node:crypto';
import * as path from 'node:path';
import * as vscode from 'vscode';

/** Matches `research/admission.py`. Manifest entries are cheap, bodies are not. */
export const PROJECT_LIMITS = {
  maxManifestEntries: 10_000,
  maxBodies: 512,
  maxBodyBytes: 1_000_000,
  maxTotalBodyBytes: 16_000_000,
} as const;

/** Asks beyond the ceiling so "there was more" is visible rather than silent. */
const ENUMERATION_ASK = PROJECT_LIMITS.maxManifestEntries + 1;

/** Directories no project graph can ever mean, independent of ignore files. */
const ALWAYS_SKIP = new Set([
  '.git',
  '.hg',
  '.svn',
  '__pycache__',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
  '.tox',
  'node_modules',
  '.venv',
  'venv',
  'env',
  '.env',
  'site-packages',
  '.ipynb_checkpoints',
  '.tensorviz',
]);

export interface ManifestEntry {
  file: string;
  digest: string;
}

export interface BodyEntry {
  file: string;
  source: string;
}

export interface ProjectSourceSet {
  manifest: ManifestEntry[];
  /** Path → text, for every admitted file. Bodies are chosen from this on demand. */
  texts: Map<string, string>;
  truncated: boolean;
  /** How many files enumeration saw before any budget applied. */
  discovered: number;
}

export interface ProjectSourceIO {
  list(): Promise<string[]>;
  read(relativePath: string): Promise<string>;
  openDocument(relativePath: string): Pick<vscode.TextDocument, 'getText'> | undefined;
  /** Contents of an ignore file, or undefined when it does not exist. */
  readIgnore(relativePath: string): Promise<string | undefined>;
}

export function sha256Text(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

/** One `.gitignore` pattern, reduced to what we can honour without a full engine. */
interface IgnoreRule {
  negated: boolean;
  directoryOnly: boolean;
  anchored: boolean;
  regex: RegExp;
}

/** Translate a gitignore glob into a regex.
 *
 * Deliberately a subset: `*`, `**`, `?`, leading `!`, trailing `/`, and a leading
 * `/` anchor. Character classes and the rarer corners are not handled — a pattern we
 * cannot read is skipped rather than approximated, because a wrong exclusion hides a
 * file the user can see in their editor, which is the one failure this whole design
 * is trying to avoid.
 */
export function compileIgnoreRule(rawLine: string): IgnoreRule | undefined {
  let line = rawLine.trim();
  if (!line || line.startsWith('#')) {
    return undefined;
  }
  if (line.includes('[') || line.includes(']')) {
    return undefined;
  }
  const negated = line.startsWith('!');
  if (negated) {
    line = line.slice(1);
  }
  const directoryOnly = line.endsWith('/');
  if (directoryOnly) {
    line = line.slice(0, -1);
  }
  const anchored = line.startsWith('/') || line.slice(0, -1).includes('/');
  if (line.startsWith('/')) {
    line = line.slice(1);
  }
  if (!line) {
    return undefined;
  }

  let pattern = '';
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '*') {
      if (line[index + 1] === '*') {
        pattern += '.*';
        index += 1;
        if (line[index + 1] === '/') {
          index += 1;
        }
      } else {
        pattern += '[^/]*';
      }
    } else if (char === '?') {
      pattern += '[^/]';
    } else if ('\\^$.|+()[]{}'.includes(char)) {
      pattern += `\\${char}`;
    } else {
      pattern += char;
    }
  }
  return {
    negated,
    directoryOnly,
    anchored,
    regex: new RegExp(anchored ? `^${pattern}(/.*)?$` : `(^|/)${pattern}(/.*)?$`),
  };
}

export function parseIgnore(contents: string): IgnoreRule[] {
  const rules: IgnoreRule[] = [];
  for (const line of contents.split(/\r?\n/)) {
    const rule = compileIgnoreRule(line);
    if (rule) {
      rules.push(rule);
    }
  }
  return rules;
}

/** Last matching rule wins, which is git's own precedence — so a `!keep.py` after a
 *  `*.py` genuinely re-includes it. */
export function isIgnored(relativePath: string, rules: readonly IgnoreRule[]): boolean {
  let ignored = false;
  for (const rule of rules) {
    if (rule.regex.test(relativePath)) {
      ignored = !rule.negated;
    }
  }
  return ignored;
}

export function hasSkippedSegment(relativePath: string): boolean {
  return relativePath.split('/').some((segment) => ALWAYS_SKIP.has(segment));
}

/** Enumerate, filter, read and digest — the pure half, driven by an injected IO. */
export async function collectProjectSources(io: ProjectSourceIO): Promise<ProjectSourceSet> {
  const ignoreText = await io.readIgnore('.gitignore');
  const rules = ignoreText ? parseIgnore(ignoreText) : [];

  const listed = await io.list();
  const discovered = listed.length;
  const candidates = listed
    .filter((rel) => !hasSkippedSegment(rel))
    .filter((rel) => !isIgnored(rel, rules))
    .sort();

  const manifest: ManifestEntry[] = [];
  const texts = new Map<string, string>();
  let truncated = listed.length >= ENUMERATION_ASK;

  for (const rel of candidates) {
    if (manifest.length >= PROJECT_LIMITS.maxManifestEntries) {
      truncated = true;
      break;
    }
    let text: string;
    const open = io.openDocument(rel);
    if (open) {
      // A dirty buffer is what the user is looking at, so it wins over disk.
      text = open.getText();
    } else {
      try {
        text = await io.read(rel);
      } catch {
        // A file that vanished between listing and reading is not an error worth
        // failing the whole index over; it simply is not there any more.
        continue;
      }
    }
    if (Buffer.byteLength(text, 'utf8') > PROJECT_LIMITS.maxBodyBytes) {
      // Too large to send as a body, so it cannot be in the manifest either — the
      // backend would ask for a body we can never supply.
      truncated = true;
      continue;
    }
    texts.set(rel, text);
    manifest.push({ file: rel, digest: sha256Text(text) });
  }

  return { manifest, texts, truncated, discovered };
}

/** Bodies for exactly the paths the backend said it lacked, within budget.
 *
 * Returns the ones that did not fit as `deferred`, so a very large first index
 * proceeds over several rounds instead of failing — each round admits more, and the
 * caller loops until `need_bodies` stops coming back.
 */
export function bodiesFor(
  set: ProjectSourceSet,
  wanted: readonly string[],
): { bodies: BodyEntry[]; deferred: string[] } {
  const bodies: BodyEntry[] = [];
  const deferred: string[] = [];
  let totalBytes = 0;

  for (const rel of wanted) {
    const source = set.texts.get(rel);
    if (source === undefined) {
      continue;
    }
    const bytes = Buffer.byteLength(source, 'utf8');
    if (bodies.length >= PROJECT_LIMITS.maxBodies || totalBytes + bytes > PROJECT_LIMITS.maxTotalBodyBytes) {
      deferred.push(rel);
      continue;
    }
    totalBytes += bytes;
    bodies.push({ file: rel, source });
  }
  return { bodies, deferred };
}

/** The production IO. Enumeration deliberately passes NO `exclude` glob: supplying
 *  one overrides `files.exclude` rather than adding to it, and ignore files are
 *  applied by `collectProjectSources` instead. */
export function vscodeProjectSourceIO(folder: string): ProjectSourceIO {
  const root = vscode.Uri.file(folder);
  const fsPathOf = (relativePath: string) => path.join(folder, relativePath);
  return {
    async list() {
      const found = await vscode.workspace.findFiles(
        new vscode.RelativePattern(root, '**/*.py'),
        undefined,
        ENUMERATION_ASK,
      );
      return found.filter(isHostFile).map((uri) => path.relative(folder, uri.fsPath).split(path.sep).join('/'));
    },
    async read(relativePath) {
      return Buffer.from(
        await vscode.workspace.fs.readFile(vscode.Uri.file(fsPathOf(relativePath))),
      ).toString('utf8');
    },
    openDocument(relativePath) {
      const fsPath = fsPathOf(relativePath);
      return vscode.workspace.textDocuments.find((document) => matchesHostFile(document.uri, fsPath));
    },
    async readIgnore(relativePath) {
      try {
        return Buffer.from(
          await vscode.workspace.fs.readFile(vscode.Uri.file(fsPathOf(relativePath))),
        ).toString('utf8');
      } catch {
        return undefined;
      }
    },
  };
}
