import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { ownedTemporaryDirectory } from '@tensorviz/graph-host/ownedTemporary';

/** Client updates can delete their cache while this host still serves a graph.
 * Pin only shipped resources, never a project, environment, or user setting. */
export async function pinAssets(plugin: string, version: string): Promise<{ directory: string; dispose(): Promise<void> }> {
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
    let cleanup: Promise<void> | undefined;
    return { directory: owned.directory, dispose: () => cleanup ??= (async () => {
      // Retry the whole operation with one budget, rather than multiplying
      // fs.rm's per-directory retry delays across a deeply nested asset tree.
      for (let attempt = 0; ; attempt++) {
        try { await fs.rm(owned.directory, { recursive: true, force: true }); return; }
        catch (error) {
          if (attempt >= 10 || !['EBUSY', 'EPERM', 'ENOTEMPTY'].includes((error as NodeJS.ErrnoException).code ?? '')) throw error;
          await delay(100);
        }
      }
    })() };
  } catch {
    await owned.dispose();
    throw new Error('TensorViz could not preserve this session’s installed resources. Finish installation and start a new chat. No model was run.');
  }
}

/** Child handles drain before beforeExit. Keep the event loop alive while Windows
 * releases file/directory locks; an exit listener cannot await asynchronous work. */
export function disposeAssetsBeforeExit(assets: { dispose(): Promise<void> }): void {
  process.once('beforeExit', () => {
    void assets.dispose().catch(() => {
      process.stderr.write('TensorViz could not remove its private session resources.\n');
      process.exitCode = 1;
    });
  });
}
