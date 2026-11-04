import * as path from 'node:path';
import * as fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { reviewGit, resolveReviewRevision } from './gitReviewSources';
import { isNormalizedPythonPath, sha256Text, SOURCE_SET_LIMITS, SourceSet } from './workspaceSourceCore';

export interface ReviewSnapshot {
  label: string;
  /** Repository-relative paths mapped to immutable blob/content identities. */
  files: ReadonlyMap<string, string>;
  read(file: string): Promise<string>;
  truncated?: boolean;
  commit?: string;
  unavailable?: ReadonlySet<string>;
  working?: boolean;
}
export interface ReviewModelRoot { file: string; cls: string; line: number; endLine: number; project: string; dependencies?: string[] }
export interface ReviewModel {
  id: string; label: string; status: 'modified' | 'added' | 'removed' | 'renamed' | 'unresolved';
  before?: ReviewModelRoot; after?: ReviewModelRoot; affectedFiles: string[];
}
export interface ReviewSession {
  title: string; before: ReviewSnapshot; after: ReviewSnapshot;
  folder?: string;
  preferredFile?: string;
  preferredClass?: string;
  preferredLocation?: { file: string; side: 'before' | 'after'; line: number };
  refresh(): Promise<ReviewSession>;
  updated?(): Promise<boolean>;
}
interface IndexedModel { file: string; cls: string; line: number; endLine: number; fingerprint: string; dependencies: string[] }
interface IndexedRoot extends ReviewModelRoot { fingerprint: string; dependencies: string[] }
export interface ReviewInventory {
  models: ReviewModel[]; changedFiles: string[]; notes: { side: 'before' | 'after'; file?: string; detail: string }[];
  sources: { before: Map<string, SourceSet>; after: Map<string, SourceSet> };
}

export function reviewCoverage(inventory: ReviewInventory, model?: ReviewModel): { notes: string[]; repositoryNotes: string[] } {
  const relevant = new Set([...inventory.changedFiles, ...[model?.before, model?.after].flatMap(root => root ? [root.file, ...root.dependencies ?? []] : [])]);
  const grouped = new Map<string, { note: ReviewInventory['notes'][number]; sides: Set<string> }>();
  for (const note of inventory.notes) {
    const key = JSON.stringify([note.file, note.detail]);
    const entry = grouped.get(key) ?? { note, sides: new Set<string>() };
    entry.sides.add(note.side); grouped.set(key, entry);
  }
  const notes: string[] = [], repositoryNotes: string[] = [];
  for (const { note, sides } of grouped.values()) {
    const label = sides.size === 2 ? 'Before and after' : sides.has('before') ? 'Before' : 'After';
    const target = !note.file || relevant.has(note.file) ? notes : repositoryNotes;
    target.push(`${label}: ${note.file ? `${note.file}: ` : ''}${note.detail}`);
  }
  return { notes, repositoryNotes };
}
const excluded = new Set(['.git', '.venv', 'venv', 'node_modules', '__pycache__', '.tox']);
export const reviewPythonPath = (file: string) => isNormalizedPythonPath(file) && !file.split('/').some(part => excluded.has(part));
const reviewPath = (file: string) => reviewPythonPath(file) || ['tensorviz.toml', 'pyproject.toml', 'setup.cfg'].includes(path.posix.basename(file));

/** Project boundaries are read from the revision, not today's working tree. */
export function snapshotProject(file: string, files: ReadonlyMap<string, string>): string {
  for (let directory = path.posix.dirname(file); directory !== '.'; directory = path.posix.dirname(directory)) {
    if (['tensorviz.toml', 'pyproject.toml', 'setup.py', 'setup.cfg'].some(name => files.has(`${directory}/${name}`))
      || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(path.posix.basename(directory))) return directory;
  }
  return '';
}

export async function snapshotSources(snapshot: ReviewSnapshot, project: string, priority: string[]): Promise<SourceSet> {
  const prefix = project ? `${project}/` : '';
  const files = [...snapshot.files.keys()].filter(file => reviewPythonPath(file) && snapshotProject(file, snapshot.files) === project)
    .sort((a, b) => Number(priority.includes(b)) - Number(priority.includes(a)) || a.localeCompare(b));
  const entries: SourceSet['entries'] = [];
  let bytes = 0, truncated = !!snapshot.truncated;
  for (const file of files) {
    if (entries.length >= SOURCE_SET_LIMITS.maxSources) { truncated = true; break; }
    if (snapshot.unavailable?.has(file)) { truncated = true; continue; }
    const source = await snapshot.read(file), size = Buffer.byteLength(source);
    if (size > SOURCE_SET_LIMITS.maxSourceBytes || bytes + size > SOURCE_SET_LIMITS.maxTotalBytes) { truncated = true; continue; }
    entries.push({ file: file.slice(prefix.length), source, digest: sha256Text(source) });
    bytes += size;
  }
  return { entries, truncated };
}

export async function discoverReview(session: ReviewSession, endpoint: { origin: string; token: string } | {
  request(body: unknown): Promise<{ models: IndexedModel[]; notes: { file: string; detail: string }[] }>;
}): Promise<ReviewInventory> {
  const changedFiles = [...new Set([...session.before.files.keys(), ...session.after.files.keys()])]
    .filter(file => session.before.files.get(file) !== session.after.files.get(file)).sort();
  const notes: ReviewInventory['notes'] = [], sources: ReviewInventory['sources'] = { before: new Map(), after: new Map() };
  const roots: Record<'before' | 'after', IndexedRoot[]> = { before: [], after: [] };
  for (const side of ['before', 'after'] as const) {
    const snapshot = session[side];
    // Scan all bounded project inventories: a shared dependency can affect a
    // model whose own file did not change. Never use just changed class names.
    const projects = [...new Set([...snapshot.files.keys()].filter(reviewPythonPath).map(file => snapshotProject(file, snapshot.files)))].sort();
    if (snapshot.truncated) notes.push({ side, detail: 'Repository inventory is incomplete.' });
    for (const project of projects) {
      const set = await snapshotSources(snapshot, project, changedFiles);
      sources[side].set(project, set);
      if (set.truncated) notes.push({ side, detail: `${project || 'Repository'} source discovery is partial (source budget reached).` });
      const body = { protocol_version: 1, truncated: set.truncated, sources: set.entries };
      const result = 'request' in endpoint ? await endpoint.request(body) : await (async () => {
        const response = await fetch(`${endpoint.origin}/api/ide/review/index`, { method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': endpoint.token },
          body: JSON.stringify(body), signal: AbortSignal.timeout(30_000) });
        if (!response.ok) throw new Error(`Model discovery failed (${response.status}).`);
        return await response.json() as { models: IndexedModel[]; notes: { file: string; detail: string }[] };
      })();
      const prefix = project ? `${project}/` : '';
      notes.push(...result.notes.map(note => ({ side, file: prefix + note.file, detail: note.detail })));
      roots[side].push(...result.models.map(model => ({ ...model, project, file: prefix + model.file, dependencies: model.dependencies.map(file => prefix + file) })));
    }
  }
  const key = (root: IndexedRoot) => `${root.file}::${root.cls}`;
  const before = new Map(roots.before.map(root => [key(root), root])), after = new Map(roots.after.map(root => [key(root), root]));
  const pairs: [IndexedRoot | undefined, IndexedRoot | undefined][] = [];
  for (const [id, left] of before) if (after.has(id)) { pairs.push([left, after.get(id)]); before.delete(id); after.delete(id); }
  // Only unique exact declaration/import fingerprints can establish renames.
  for (const [id, left] of before) {
    const leftMatches = [...before.values()].filter(root => root.fingerprint === left.fingerprint);
    const rightMatches = [...after.values()].filter(root => root.fingerprint === left.fingerprint);
    if (leftMatches.length === 1 && rightMatches.length === 1 && left.project === rightMatches[0].project) {
      pairs.push([left, rightMatches[0]]); before.delete(id); after.delete(key(rightMatches[0]));
    }
  }
  pairs.push(...[...before.values()].map(root => [root, undefined] as [IndexedRoot, undefined]),
    ...[...after.values()].map(root => [undefined, root] as [undefined, IndexedRoot]));
  const models = pairs.flatMap(([left, right]): ReviewModel[] => {
    const affectedFiles = changedFiles.filter(file => left?.dependencies.includes(file) || right?.dependencies.includes(file));
    if (!affectedFiles.length && left && right) return [];
    const root = right ?? left!;
    const missingSide = !left ? 'before' : !right ? 'after' : undefined;
    const parseBlocked = !!missingSide && notes.some(note => note.side === missingSide && note.detail.startsWith('could not parse ')
      && !!note.file && (note.file === root.file || root.dependencies.includes(note.file)));
    const incomplete = missingSide && (session[missingSide].truncated || sources[missingSide].get(root.project)?.truncated
      || notes.some(note => note.side === missingSide && note.file === root.file));
    const status = incomplete || parseBlocked ? 'unresolved' : !left ? 'added' : !right ? 'removed' : key(left) !== key(right) ? 'renamed' : 'modified';
    return [{ id: sha256Text(`${left ? key(left) : ''}\0${right ? key(right) : ''}`),
      label: status === 'renamed' ? `${left!.cls} → ${right!.cls}` : root.cls, status,
      before: left, after: right, affectedFiles }];
  });
  models.sort((a, b) => Number(b.affectedFiles.includes(session.preferredFile ?? '')) - Number(a.affectedFiles.includes(session.preferredFile ?? ''))
    || (a.after ?? a.before)!.file.localeCompare((b.after ?? b.before)!.file));
  return { models, changedFiles, notes, sources };
}

export async function gitSnapshot(folder: string, ref: string): Promise<ReviewSnapshot> {
  const commit = await resolveReviewRevision(folder, ref);
  const tree = await reviewGit(folder, ['ls-tree', '-rz', '--full-tree', commit]);
  const files = new Map<string, string>();
  for (const entry of tree.split('\0').filter(Boolean)) {
    const tab = entry.indexOf('\t'), [mode, type, sha] = entry.slice(0, tab).split(' ');
    if (type === 'blob' && mode !== '120000' && reviewPath(entry.slice(tab + 1))) files.set(entry.slice(tab + 1), sha);
  }
  return { label: commit.slice(0, 7), commit, files,
    read: file => { const sha = files.get(file); if (!sha) throw new Error('Source is unavailable.'); return reviewGit(folder, ['cat-file', 'blob', sha]); } };
}

export async function indexSnapshot(folder: string): Promise<ReviewSnapshot> {
  const entries = (await reviewGit(folder, ['ls-files', '--stage', '-z'])).split('\0').filter(Boolean);
  const files = new Map<string, string>();
  for (const entry of entries) {
    const tab = entry.indexOf('\t'), [mode, sha, stage] = entry.slice(0, tab).split(' '), file = entry.slice(tab + 1);
    if (stage !== '0') throw new Error('Resolve merge conflicts before reviewing the index.');
    if (mode !== '120000' && reviewPath(file)) files.set(file, sha);
  }
  return { label: 'Index', files, read: file => { const sha = files.get(file); if (!sha) throw new Error('Source is unavailable.'); return reviewGit(folder, ['cat-file', 'blob', sha]); } };
}

export async function workingSnapshotCore(folder: string, open: readonly { file: string; isDirty: boolean; getText(): string }[] = []): Promise<ReviewSnapshot> {
  const names = (await reviewGit(folder, ['ls-files', '-z', '--cached', '--others', '--exclude-standard'])).split('\0').filter(Boolean);
  const buffers = open.map(doc => path.relative(folder, doc.file).split(path.sep).join('/')).filter(reviewPath);
  const unlisted = buffers.filter(file => !names.includes(file));
  let ignored = new Set<string>();
  if (unlisted.length) {
    try { ignored = new Set((await reviewGit(folder, ['check-ignore', '-z', '--stdin'], `${unlisted.join('\0')}\0`)).split('\0').filter(Boolean)); }
    catch (error) { if ((error as { code?: number }).code !== 1) throw error; }
  }
  names.push(...buffers.filter(file => !ignored.has(file)));
  const files = new Map<string, string>(), contents = new Map<string, string>();
  let truncated = false, bytes = 0;
  for (const file of [...new Set(names)].filter(reviewPath)) {
    try {
      const document = open.find(doc => path.relative(folder, doc.file).split(path.sep).join('/') === file);
      const stat = await fs.lstat(path.join(folder, file)).catch(error => {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
        throw error;
      });
      // A clean buffer can outlive deletion on disk. Only an unsaved edit can
      // contribute source for a now-missing file.
      if ((!stat && !document?.isDirty) || stat?.isSymbolicLink()) continue;
      if (stat && stat.size > SOURCE_SET_LIMITS.maxSourceBytes && !document?.isDirty) { truncated = true; continue; }
      const source = document?.getText() ?? await fs.readFile(path.join(folder, file), 'utf8');
      if (Buffer.byteLength(source) > SOURCE_SET_LIMITS.maxSourceBytes || bytes + Buffer.byteLength(source) > 64_000_000 || files.size >= 8192) { truncated = true; continue; }
      bytes += Buffer.byteLength(source);
      contents.set(file, source); files.set(file, createHash('sha1').update(`blob ${Buffer.byteLength(source)}\0`).update(source).digest('hex'));
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  }
  return { label: 'Working changes', files, truncated, working: true, read: async file => { const source = contents.get(file); if (source === undefined) throw new Error('Source is unavailable.'); return source; } };
}
