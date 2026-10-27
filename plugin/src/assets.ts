import { promises as fs, rmSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { ownedTemporaryDirectory } from '@tensorviz/graph-host/ownedTemporary';

/** Client updates can delete their cache while this host still serves a graph.
 * Pin only shipped resources, never a project, environment, or user setting. */
export async function pinAssets(plugin: string, version: string): Promise<{ directory: string; dispose(): void }> {
  const owned = await ownedTemporaryDirectory(await fs.realpath(os.tmpdir()), 'tensorviz-release-session-');
  try {
    if (JSON.parse(await fs.readFile(path.join(plugin, 'plugin.json'), 'utf8')).version !== version) {
      throw new Error('The installed host and plugin versions differ.');
    }
    const entries: { file: string; hash: string }[] = [];
    let bytes = 0;
    const scan = async (file: string) => {
      const source = path.join(plugin, file), stat = await fs.lstat(source);
      if (stat.isDirectory()) {
        await fs.mkdir(path.join(owned.directory, file), { recursive: true, mode: 0o700 });
        for (const name of await fs.readdir(source)) await scan(path.join(file, name));
      } else if (stat.isFile() && entries.length < 4096 && (bytes += stat.size) <= 128 * 1024 * 1024) {
        const contents = await fs.readFile(source);
        entries.push({ file, hash: createHash('sha256').update(contents).digest('hex') });
        await fs.writeFile(path.join(owned.directory, file), contents, { flag: 'wx', mode: 0o600 });
      } else throw new Error('The installed release contains linked, invalid or oversized resources.');
    };
    for (const file of ['backend', 'browser', 'runtime-manifest.json']) await scan(file);
    for (const entry of entries) {
      const contents = await fs.readFile(path.join(plugin, entry.file));
      if (createHash('sha256').update(contents).digest('hex') !== entry.hash) throw new Error('The installed release changed during startup.');
    }
    return { directory: owned.directory, dispose: () => rmSync(owned.directory, { recursive: true, force: true }) };
  } catch {
    await owned.dispose();
    throw new Error('TensorViz could not preserve this session’s installed resources. Finish installation and start a new chat. No model was run.');
  }
}
