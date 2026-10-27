import { constants, promises as fs } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { applicationData } from './environment';

// Increment when the capabilities described to the user expand, not on every release.
export const permissionVersion = 1;
export const permissionSummary = `Allow TensorViz to work with local models without asking again?

When you request a model workflow, TensorViz can read your selected projects, inspect Python environments, run model/setup code, save TensorViz recipes and generated setup helpers, download verified Python runtimes, and build/install required dependencies in the selected environment. Project and package code runs with your user account's access; it is not sandboxed by TensorViz. Graph evidence is returned to your coding client.

This permission applies to TensorViz across projects, chats, compatible updates and coding clients on this computer. Your explicit read-only instructions and client/organization permissions still apply. Revoke it with the plugin's scripts/launch.sh --revoke-permission command.`;

/** User-level consent lives outside project files and the replaceable plugin cache. */
export class PluginPermissions {
  readonly directory: string;
  readonly file: string;
  constructor(storage = process.env.TENSORVIZ_DATA_DIR ?? applicationData()) {
    this.directory = path.join(storage, 'permissions');
    this.file = path.join(this.directory, 'standalone.json');
  }

  private async checkDirectory(): Promise<void> {
    const stat = await fs.lstat(this.directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || process.getuid && stat.uid !== process.getuid()
      || process.platform !== 'win32' && (stat.mode & 0o077)) throw new Error('TensorViz permissions directory must be private and owned by you.');
  }

  async status(): Promise<'granted' | 'denied' | 'missing'> {
    return (await this.receipt()).status;
  }

  async receipt(): Promise<{ status: 'granted' | 'denied' | 'missing'; revision?: string }> {
    try { await this.checkDirectory(); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { status: 'missing' }; throw error; }
    let handle;
    try {
      handle = await fs.open(this.file, constants.O_RDONLY | constants.O_NOFOLLOW);
      const stat = await handle.stat();
      if (!stat.isFile() || stat.nlink !== 1 || stat.size > 4096 || process.getuid && stat.uid !== process.getuid()
        || process.platform !== 'win32' && (stat.mode & 0o077)) throw new Error('TensorViz permission record must be a private, regular file owned by you.');
      const revision = await handle.readFile('utf8');
      const value = JSON.parse(revision);
      if (value.plugin !== 'tensorviz-standalone' || value.version !== permissionVersion) return { status: 'missing', revision };
      return { status: value.decision === 'granted' ? 'granted' : value.decision === 'denied' ? 'denied' : 'missing', revision };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { status: 'missing' };
      throw error;
    } finally { await handle?.close(); }
  }

  async save(decision: 'granted' | 'denied'): Promise<void> {
    await fs.mkdir(this.directory, { recursive: true, mode: 0o700 });
    await this.checkDirectory();
    const temporary = path.join(this.directory, `${randomUUID()}.tmp`);
    try {
      await fs.writeFile(temporary, JSON.stringify({ plugin: 'tensorviz-standalone', version: permissionVersion,
        decision, decisionId: randomUUID(), updatedAt: new Date().toISOString() }) + '\n', { flag: 'wx', mode: 0o600 });
      await fs.rename(temporary, this.file);
    } finally { await fs.rm(temporary, { force: true }); }
  }
}
