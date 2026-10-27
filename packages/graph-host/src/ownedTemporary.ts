import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { execFileSync } from 'node:child_process';

const marker = 'tensorviz-owner.json';
const sameUser = (stat: { uid: number }) => !process.getuid || stat.uid === process.getuid();
const dead = (pid: number): boolean => {
  try { process.kill(pid, 0); return false; }
  catch (error) { return (error as NodeJS.ErrnoException).code === 'ESRCH'; }
};

/** Lazy recovery, bounded to our prefix and positively identified dead owners.
 * Unknown/legacy directories, live or reused PIDs, links and other users survive. */
export async function recoverOwnedTemporary(parent: string, prefix: string): Promise<number> {
  let removed = 0, scanned = 0, candidates = 0;
  const entries = await fs.opendir(parent);
  for await (const entry of entries) {
    if (++scanned > 4096 || candidates >= 64 || removed >= 8) break;
    if (!entry.name.startsWith(prefix) || !entry.isDirectory()) continue;
    candidates++;
    const directory = path.join(parent, entry.name), record = path.join(directory, marker);
    let stat, contents;
    try {
      stat = await fs.lstat(directory);
      const info = await fs.lstat(record);
      if (!stat.isDirectory() || stat.isSymbolicLink() || !sameUser(stat) || !info.isFile()
        || info.isSymbolicLink() || info.nlink !== 1 || !sameUser(info) || info.size > 4096) continue;
      contents = await fs.readFile(record, 'utf8');
      const owner = JSON.parse(contents);
      if (owner.version !== 1 || owner.kind !== 'tensorviz-temporary' || owner.directory !== entry.name
        || owner.host !== os.hostname() || !Number.isSafeInteger(owner.pid) || owner.pid <= 1
        || !Number.isFinite(owner.createdAt) || owner.createdAt > Date.now() - 60_000 || !dead(owner.pid)) continue;
      const checked = await fs.lstat(directory);
      if (checked.ino !== stat.ino || checked.dev !== stat.dev || await fs.readFile(record, 'utf8') !== contents || !dead(owner.pid)) continue;
    } catch { continue; } // Unreadable or unrecognized ownership is never deletion authority.
    await fs.rm(directory, { recursive: true, force: true });
    removed++;
  }
  return removed;
}

export async function ownedTemporaryDirectory(parent: string, prefix: string): Promise<{ directory: string; dispose(): Promise<void>; retain(): Promise<void> }> {
  await fs.mkdir(parent, { recursive: true, mode: 0o700 });
  const parentStat = await fs.lstat(parent);
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink()) throw new Error('Temporary storage is not a regular directory.');
  await recoverOwnedTemporary(parent, prefix);
  const directory = await fs.mkdtemp(path.join(parent, prefix));
  const dispose = () => fs.rm(directory, { recursive: true, force: true });
  try {
    if (process.platform === 'win32') execFileSync('icacls', [directory, '/inheritance:r', '/grant:r', `${os.userInfo().username}:(OI)(CI)F`], { stdio: 'ignore' });
    else await fs.chmod(directory, 0o700);
    await fs.writeFile(path.join(directory, marker), JSON.stringify({ version: 1, kind: 'tensorviz-temporary',
      directory: path.basename(directory), pid: process.pid, host: os.hostname(), createdAt: Date.now() }), { flag: 'wx', mode: 0o600 });
    return { directory, dispose, retain: () => fs.unlink(path.join(directory, marker)) };
  } catch (error) { await dispose(); throw error; }
}
