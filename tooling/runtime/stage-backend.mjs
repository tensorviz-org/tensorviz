import { cpSync, copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

export function backendProfiles(repository) {
  return JSON.parse(readFileSync(path.join(repository, 'runtime/backend-profiles.json'), 'utf8'));
}

export function includedBackendFile(relative, config, profile) {
  const parts = relative.split(/[\\/]/);
  const name = parts[parts.length - 1];
  if (parts.some(part => config.excludedDirectories.includes(part))
    || profile.excludedFiles?.includes(relative)
    || config.excludedFileNames.includes(name)
    || config.excludedFilePrefixes.some(prefix => name.startsWith(prefix))) return false;
  return Boolean(profile.tree || profile.files.includes(relative)
    || relative.endsWith('.py') && profile.pythonPrefixes.some(prefix => relative.startsWith(prefix)));
}

/** Preserve each host's runtime surface; neither release reads another host's staging. */
export function stageBackend(repository, destination, profileName) {
  const config = backendProfiles(repository);
  const profile = config.profiles[profileName];
  if (!profile) throw new Error(`Unknown backend profile: ${profileName}`);
  const source = path.join(repository, 'backend');
  if (!existsSync(path.join(source, profile.entry))) throw new Error(`Missing backend entry: ${profile.entry}`);
  rmSync(destination, { recursive: true, force: true });
  mkdirSync(destination, { recursive: true });
  if (profile.tree) {
    cpSync(source, destination, { recursive: true, filter: file => {
      const relative = path.relative(source, file);
      return !relative || includedBackendFile(relative, config, profile);
    } });
    return;
  }
  const files = execFileSync('git', ['ls-files', '-z', 'backend'], { cwd: repository, encoding: 'utf8' })
    .split('\0').filter(Boolean);
  for (const file of files) {
    const relative = file.slice('backend/'.length);
    if (!includedBackendFile(relative, config, profile)) continue;
    const target = path.join(destination, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    copyFileSync(path.join(repository, file), target);
  }
}

/** File the execution cache reads as the staged backend's content identity
 *  (`packages/graph-host/src/backendIdentity.ts`). */
export const BACKEND_IDENTITY_FILE = 'backend-identity.json';

/** Record a content digest of every staged file, so a cached worker result is never
 *  reused across backend builds. Same line format as `backendIdentity.ts::digestFiles`:
 *  sha256 over sorted `relpath \0 sha256(bytes) \n`. Bytecode caches are not hashed. */
export function writeBackendIdentity(destination) {
  const files = [];
  const walk = directory => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) { if (entry.name !== '__pycache__') walk(full); }
      else if (entry.isFile() && entry.name !== BACKEND_IDENTITY_FILE) files.push(path.relative(destination, full).split(path.sep).join('/'));
    }
  };
  walk(destination);
  const hash = createHash('sha256');
  for (const relative of files.sort()) {
    hash.update(`${relative}\0${createHash('sha256').update(readFileSync(path.join(destination, relative))).digest('hex')}\n`);
  }
  const digest = hash.digest('hex');
  writeFileSync(path.join(destination, BACKEND_IDENTITY_FILE), `${JSON.stringify({ format: 1, digest, files: files.length })}\n`);
  return digest;
}
