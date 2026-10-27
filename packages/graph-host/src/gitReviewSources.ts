/** Read Git objects without checkout, imports, hooks, or workspace writes. */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import * as path from 'node:path';
import { realpath } from 'node:fs/promises';
import { collectSourceSet, relativeInside, SourceIO, SourceSet } from './workspaceSourceCore';

const execute = promisify(execFile);
export async function reviewGit(folder: string, args: string[]): Promise<string> {
  const env = { ...process.env, GIT_OPTIONAL_LOCKS: '0', GIT_TERMINAL_PROMPT: '0' } as NodeJS.ProcessEnv;
  // Repository-local variables (git rev-parse --local-env-vars) can redirect -C,
  // notably when VS Code or a test is launched from another repository's hook.
  for (const key of ['GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_CONFIG', 'GIT_CONFIG_PARAMETERS', 'GIT_CONFIG_COUNT',
    'GIT_OBJECT_DIRECTORY', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_IMPLICIT_WORK_TREE', 'GIT_GRAFT_FILE', 'GIT_INDEX_FILE',
    'GIT_NO_REPLACE_OBJECTS', 'GIT_REPLACE_REF_BASE', 'GIT_PREFIX', 'GIT_SHALLOW_FILE', 'GIT_COMMON_DIR']) delete env[key];
  const result = await execute('git', ['--no-pager', '-C', folder, ...args], {
    encoding: 'utf8', timeout: 15_000, maxBuffer: 10 * 1024 * 1024,
    env,
  });
  return result.stdout;
}

export async function resolveReviewRevision(folder: string, ref: string): Promise<string> {
  if (!ref || ref.length > 256 || /[\x00-\x1f]/.test(ref)) throw new Error('Enter a Git revision.');
  try { return (await reviewGit(folder, ['rev-parse', '--verify', '--end-of-options', `${ref}^{commit}`])).trim(); }
  catch { throw new Error(`Git revision “${ref}” is not available locally. Fetch it or choose another revision.`); }
}

export async function gitReviewSourceSet(folder: string, file: string, commit: string): Promise<SourceSet> {
  folder = await realpath(folder);
  const repository = await realpath((await reviewGit(folder, ['rev-parse', '--show-toplevel'])).trim());
  const prefix = path.relative(repository, folder).split(path.sep).join('/');
  if (prefix.startsWith('..') || path.isAbsolute(prefix)) throw new Error('Model is outside the Git repository.');
  const paths = (await reviewGit(repository, ['ls-tree', '-rz', '--name-only', commit, '--', prefix || '.']))
    .split('\0').filter(Boolean).map(name => relativeInside(folder, path.join(repository, name)))
    .filter((name): name is string => !!name && name.endsWith('.py')
      && !name.split('/').some(part => ['.git', '.venv', 'venv', '__pycache__', 'node_modules', '.tox'].includes(part)));
  const io: SourceIO = {
    list: async () => paths,
    openDocument: () => undefined,
    read: name => reviewGit(repository, ['show', `${commit}:${prefix ? `${prefix}/` : ''}${name}`]),
  };
  try { return await collectSourceSet(file, io); }
  catch { throw new Error(`The model file “${file}” is unavailable at ${commit.slice(0, 8)}. Choose a revision containing this file.`); }
}

export function changedReviewFiles(before: SourceSet, after: SourceSet): string[] {
  const left = new Map(before.entries.map(entry => [entry.file, entry.digest]));
  const right = new Map(after.entries.map(entry => [entry.file, entry.digest]));
  return [...new Set([...left.keys(), ...right.keys()])].filter(file => left.get(file) !== right.get(file)).sort();
}
