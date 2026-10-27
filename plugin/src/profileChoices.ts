import { promises as fs } from 'node:fs';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { digest, HostError } from './workspace';

/** Preferences are local to the exact checkout and model, never written into its recipe. */
export class ProfileChoices {
  constructor(private readonly storage: string) {}
  private file(root: string, file: string, cls: string): string {
    return path.join(this.storage, 'workspaces', digest(root), 'profiles', digest(JSON.stringify([file, cls])) + '.json');
  }
  async read(root: string, file: string, cls: string): Promise<string | undefined> {
    try {
      const value = JSON.parse(await fs.readFile(this.file(root, file, cls), 'utf8'));
      if (value.root !== root || value.file !== file || value.cls !== cls || typeof value.profile !== 'string'
        || !/^[A-Za-z_]\w{0,63}$/.test(value.profile)) throw new Error();
      return value.profile;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw new HostError('PROFILE_SELECTION_INVALID', 'The saved profile choice cannot be read. Supply an explicit profile to replace it.');
    }
  }
  async remember(root: string, file: string, cls: string, profile: string): Promise<void> {
    const destination = this.file(root, file, cls), temporary = `${destination}.${randomUUID()}.tmp`;
    await fs.mkdir(path.dirname(destination), { recursive: true, mode: 0o700 });
    try {
      await fs.writeFile(temporary, JSON.stringify({ root, file, cls, profile }) + '\n', { flag: 'wx', mode: 0o600 });
      await fs.rename(temporary, destination);
    } finally { await fs.rm(temporary, { force: true }); }
  }
}
