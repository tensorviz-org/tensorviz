import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { randomUUID } from 'node:crypto';
import * as lockfile from 'proper-lockfile';
import { callHost, type HostRegistry } from './hostBus';
import { DEFAULT_MCP_PORT, type McpConnection } from './mcpServer';

export interface McpStatus {
  state: 'Off' | 'Ready' | 'Connected';
  url?: string;
  sessions: number;
  error?: string;
}
interface Preference { enabled: boolean; port: number; allocatedPort?: number }
interface LifecycleOptions {
  directory: string;
  hostId: string;
  registry: HostRegistry;
  start(port: number): Promise<McpConnection>;
  stop(): Promise<void>;
  local(): { url: string; sessions: number } | undefined;
  publish(status: McpStatus): void;
}

/** Machine-local consent and endpoint takeover, independent of workspace settings. */
export class McpLifecycle {
  private timer?: ReturnType<typeof setInterval>;
  private refreshing?: Promise<void>;
  private disposed = false;
  private value: McpStatus = { state: 'Off', sessions: 0 };
  constructor(private readonly options: LifecycleOptions) {}
  get status(): McpStatus { return this.value; }
  private get file(): string { return path.join(this.options.directory, 'mcp-connection.json'); }

  start(): void {
    void this.refresh();
    this.timer = setInterval(() => { void this.refresh(); }, 2000);
    this.timer.unref();
  }

  private async preference(): Promise<Preference> {
    try {
      const value = JSON.parse(await fs.readFile(this.file, 'utf8')) as Preference;
      if (typeof value.enabled !== 'boolean' || !Number.isInteger(value.port) || value.port < 0 || value.port > 65535
        || value.allocatedPort !== undefined && (!Number.isInteger(value.allocatedPort) || value.allocatedPort < 1 || value.allocatedPort > 65535)) throw new Error();
      return value;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { enabled: false, port: DEFAULT_MCP_PORT };
      throw new Error('MCP connection settings could not be read. Reconnect from the status bar.');
    }
  }

  async setEnabled(enabled: boolean, port?: number): Promise<void> {
    await this.withLock(async check => {
      const previous: Preference = await this.preference().catch(() => ({ enabled: false, port: DEFAULT_MCP_PORT }));
      const selected = port ?? previous.port;
      if (!Number.isInteger(selected) || selected < 0 || selected > 65535) throw new Error('Choose MCP port 0 for automatic selection, or a fixed port from 1 to 65535.');
      check();
      await this.save({ enabled, port: selected,
        ...(selected === previous.port && previous.allocatedPort ? { allocatedPort: previous.allocatedPort } : {}) });
    });
    await this.refreshing;
    await this.refresh();
    // Synchronize disconnect across windows now; polling also covers crashed peers.
    await Promise.allSettled((await this.options.registry.list()).filter(peer => peer.hostId !== this.options.hostId)
      .map(peer => callHost(peer, 'sync', {}, 3000)));
  }

  refresh(): Promise<void> {
    return this.refreshing ??= this.update().finally(() => { this.refreshing = undefined; });
  }

  async synchronize(): Promise<void> {
    // A tick already in progress may have read the previous consent value.
    await this.refreshing;
    await this.refresh();
  }

  private publish(value: McpStatus): void {
    this.value = value;
    if (!this.disposed) this.options.publish(value);
  }

  /** Serialize consent and port allocation across windows, including simultaneous startup. */
  private async withLock(action: (check: () => void) => Promise<void>): Promise<void> {
    await fs.mkdir(this.options.directory, { recursive: true, mode: 0o700 });
    let compromised = false;
    const release = await lockfile.lock(this.file, { realpath: false, stale: 10_000, update: 2000,
      retries: { retries: 100, factor: 1, minTimeout: 100, maxTimeout: 100 },
      onCompromised: () => { compromised = true; } });
    const check = () => { if (compromised) throw new Error('The MCP connection lock changed. Reconnect from the status bar.'); };
    try { check(); await action(check); check(); }
    finally { await release(); }
  }

  private async save(preference: Preference): Promise<void> {
    const temporary = `${this.file}.${randomUUID()}.tmp`;
    try {
      await fs.writeFile(temporary, JSON.stringify(preference), { mode: 0o600, flag: 'wx' });
      await fs.rename(temporary, this.file);
    } finally { await fs.rm(temporary, { force: true }); }
  }

  private async update(): Promise<void> {
    if (this.disposed) return;
    try {
      await this.withLock(async check => {
        if (this.disposed) return;
        const preference = await this.preference();
        if (!preference.enabled) {
          await this.options.stop();
          this.publish({ state: 'Off', sessions: 0 });
          return;
        }
        const port = preference.port || preference.allocatedPort || 0;
        const url = `http://127.0.0.1:${port}/mcp`;
        if (this.options.local()?.url !== url) await this.options.stop();
        if (this.disposed) return;
        if (!this.options.local()) {
          try { await this.options.start(port); }
          catch (error) { if ((error as NodeJS.ErrnoException).code !== 'EADDRINUSE') throw error; }
        }
        if (this.disposed) { await this.options.stop(); return; }
        let connection = this.options.local();
        if (!connection) {
          for (const peer of await this.options.registry.list()) {
            if (peer.hostId === this.options.hostId) continue;
            try {
              const info = await callHost(peer, 'describe', {}, 1000) as { mcp?: { url: string; sessions: number } };
              if (info.mcp?.url === url) { connection = info.mcp; break; }
            } catch { /* A closing window is retried on the next tick. */ }
          }
        }
        check();
        if (!connection && preference.port === 0) {
          await this.options.start(0);
          connection = this.options.local();
        }
        if (!connection) throw new Error('The configured MCP port is occupied by another application. Set tensorviz.mcpPort to 0 for automatic selection and reconnect.');
        if (this.disposed) { await this.options.stop(); return; }
        if (preference.port === 0 && preference.allocatedPort !== Number(new URL(connection.url).port)) {
          check();
          await this.save({ ...preference, allocatedPort: Number(new URL(connection.url).port) });
        }
        this.publish({ state: connection.sessions > 0 ? 'Connected' : 'Ready', url: connection.url, sessions: connection.sessions });
      });
    } catch (error) {
      await this.options.stop();
      this.publish({ state: 'Off', sessions: 0, error: error instanceof Error ? error.message : 'MCP could not start.' });
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    if (this.timer) clearInterval(this.timer);
    await this.refreshing;
    await this.options.stop();
  }
}
