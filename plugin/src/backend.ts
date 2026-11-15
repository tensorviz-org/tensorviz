import { spawn, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { backendPythonArgs } from '@tensorviz/graph-host/backendPython';
import { createLocalBrowserHostTransport, mintLocalBrowserSession, mintLocalBrowserViewLink,
  type LocalBrowserHostTransport, type LocalBrowserSessionMaterial } from '@tensorviz/graph-host/localBrowserHostTransport';
import { preparePortableCapture, PortableExportError, type PortableCapture } from '@tensorviz/graph-host/portableCapture';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { HostError } from './workspace';
import { CAPTURE_SERVICE_MAX_BYTES } from '@tensorviz/graph-host/captureBudgets';

/** One owned service process; it receives source snapshots, never execution authority. */
export class Backend {
  private child?: ChildProcess;
  private token?: string;
  private origin?: string;
  private session?: LocalBrowserSessionMaterial;
  transport?: LocalBrowserHostTransport;
  constructor(readonly directory: string, readonly browser: string) {}
  get ready(): boolean { return !!this.child && !!this.origin; }

  async start(interpreter: string, signal: AbortSignal): Promise<void> {
    if (this.child) return;
    signal.throwIfAborted();
    this.token = randomBytes(32).toString('hex');
    const child = spawn(interpreter, backendPythonArgs(this.directory, 'standalone_server.py', ['--parent-pipe',
      ...(existsSync(path.join(this.browser, 'index.html')) ? [this.browser] : [])]), {
      cwd: this.directory, detached: process.platform !== 'win32', windowsHide: true,
      env: { ...process.env, TENSORVIZ_LOCAL_TOKEN: this.token, PYTHONDONTWRITEBYTECODE: '1' },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.child = child;
    child.stderr!.resume();
    const abort = () => this.dispose();
    signal.addEventListener('abort', abort, { once: true });
    try {
      const port = await new Promise<number>((resolve, reject) => {
        let text = '', done = false;
        const finish = (port?: number) => {
          if (done) return;
          done = true;
          clearTimeout(timer);
          if (port) resolve(port); else reject(new HostError('BACKEND_UNAVAILABLE', 'The selected Python could not start the plugin service. Inspect its missing requirements.'));
        };
        const timer = setTimeout(() => { finish(); this.dispose(); }, 30_000);
        child.once('error', () => finish());
        child.once('exit', () => {
          finish();
          if (this.child === child) { this.transport?.dispose(); this.child = undefined; this.origin = undefined; this.session = undefined; }
        });
        child.stdout!.on('data', chunk => {
          if (done) return;
          text += chunk.toString();
          if (text.length > 4096) { finish(); this.dispose(); return; }
          if (!text.includes('\n')) return;
          try {
            const value = JSON.parse(text.slice(0, text.indexOf('\n')));
            if (!Number.isInteger(value.port) || value.port < 1 || value.port > 65535) throw new Error();
            finish(value.port);
          } catch { finish(); this.dispose(); }
        });
      });
      this.origin = `http://127.0.0.1:${port}`;
      // The service reports its bound port before the event loop has started.
      for (let attempt = 0; ; attempt++) {
        try { await this.request('/api/health', undefined, signal); break; }
        catch (error) {
          if (signal.aborted || !this.child || attempt >= 40) throw error;
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
    } catch (error) { this.dispose(); throw error; }
    finally { signal.removeEventListener('abort', abort); }
  }

  async request<T = Record<string, unknown>>(route: string, body: unknown, signal: AbortSignal): Promise<T> {
    if (!this.origin || !this.token) throw new HostError('BACKEND_UNAVAILABLE', 'The local service is unavailable. Reopen the graph.');
    const response = await fetch(`${this.origin}${route}`, {
      method: body === undefined ? 'GET' : 'POST', redirect: 'error',
      headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': this.token },
      body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]),
    });
    let size = 0;
    const chunks: Uint8Array[] = [];
    const reader = response.body!.getReader();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > CAPTURE_SERVICE_MAX_BYTES) { await reader.cancel(); throw new HostError('RESULT_LIMIT', 'The local-service result exceeds its limit.'); }
      chunks.push(value);
    }
    const result = JSON.parse(Buffer.concat(chunks).toString());
    if (!response.ok) throw new HostError('PREPARATION_FAILED', typeof result.detail === 'string' ? result.detail : 'The supplied source or capture could not be projected.');
    return result as T;
  }

  /** Retained evidence detached for a graph file. Python owns detachment; the launch
   *  token never reaches the view. Nothing is executed. */
  async portable(capture: CaptureSnapshot, signal: AbortSignal): Promise<PortableCapture> {
    if (!this.origin || !this.token) throw new PortableExportError('Reopen the graph before exporting its capture.');
    return preparePortableCapture(capture, { origin: this.origin, token: this.token, signal });
  }

  async connect(signal: AbortSignal): Promise<LocalBrowserHostTransport> {
    if (this.transport && this.transport.status.phase !== 'ended') return this.transport;
    this.transport?.dispose();
    this.transport = undefined;
    if (!this.origin || !this.token) throw new HostError('BACKEND_UNAVAILABLE', 'Open a graph before opening its browser view.');
    this.session = await mintLocalBrowserSession({ backendBaseUrl: this.origin, launchOrigin: this.origin, launchToken: this.token });
    signal.throwIfAborted();
    this.transport = createLocalBrowserHostTransport({ backendBaseUrl: this.origin, session: this.session, log: () => {} });
    await this.transport.connect();
    return this.transport;
  }

  async viewLink(signal: AbortSignal) {
    if (!existsSync(path.join(this.browser, 'index.html'))) throw new HostError('BROWSER_ASSETS_MISSING', 'This installation is missing its browser assets. Rebuild or reinstall the complete plugin; capture evidence remains inspectable.');
    if (!this.session || !this.origin || !this.token) throw new HostError('BACKEND_UNAVAILABLE', 'The browser session is unavailable.');
    return mintLocalBrowserViewLink({ backendBaseUrl: this.origin, launchToken: this.token, session: this.session, signal });
  }

  sessionCurrent(id: string): boolean { return !!this.child && this.transport?.status.phase !== 'ended' && this.session?.workspaceSessionId === id; }

  dispose(): void {
    this.transport?.dispose(); this.transport = undefined;
    const child = this.child;
    this.child = undefined; this.origin = undefined; this.session = undefined; this.token = undefined;
    // Release the service's parent pipe even if platform termination is denied.
    child?.stdin?.end();
    if (child?.pid) {
      try {
        if (process.platform === 'win32') spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true }).unref();
        else process.kill(-child.pid, 'SIGKILL');
      } catch { /* The owned process may already have exited. */ }
    }
  }
}

