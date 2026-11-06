/** Publish from the plugin's Canvas to the member's tensorViz account, as the editor
 * extension does. The view sends the graph file Export graph would save; this host signs
 * the member in through tensorviz.dev and uploads it with the shared website client.
 *
 * The view, not this process, opens the website: the member opens the connect page and
 * the graph's Share page from the Canvas. The website sends the sign-in back to a
 * one-time loopback address here (its connect.js accepts
 * `http://127.0.0.1:<port>/tensorviz/auth`). Only the refresh token is kept, in a private
 * file next to the plugin's permission record. See docs/share-to-website.md.
 */
import { constants, promises as fs } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { WebsiteAccount, WebsiteError, type AccountHost, type Cancellation, type WebsiteConfig,
  type WebsiteFetch, type WebsiteWording } from '@tensorviz/graph-host/websiteAccount';
import { graphUrl, shareableGraph, shareGraph, SHARE_LIMITS } from '@tensorviz/graph-host/websiteShare';
import type { LocalBrowserPublishedGraph } from '@tensorviz/contracts/protocol';
import { applicationData } from './environment';

/** The website's own Firebase web app: the same public identifiers as
 *  extension/src/websiteAccount.ts. They identify tensorviz.dev; they are not credentials. */
export const WEBSITE: WebsiteConfig = {
  site: 'https://tensorviz.dev',
  apiKey: 'AIzaSyBB-uxMsrJ4lzA_ywP0JdZMxM0vsPNoxxU',
  projectId: 'project-fe5816f4-14e7-42f8-aa0',
  identity: 'https://identitytoolkit.googleapis.com',
  secureToken: 'https://securetoken.googleapis.com',
};

const WORDING: WebsiteWording = {
  client: 'the tensorViz plugin',
  timedOut: 'Publish again to sign in.',
  notConfirmed: 'Publish again to sign in.',
};

const CALLBACK_PATH = '/tensorviz/auth';
const SIGNED_IN_PATH = '/tensorviz/signed-in';
const EXPIRED_PATH = '/tensorviz/sign-in-expired';
/** Serve the closing page after the callback, then stop listening. */
const CALLBACK_LINGER_MS = 60_000;
const SESSION_FILE_MAX_BYTES = 64 * 1024;

type SecretStore = AccountHost['secrets'];

/** The plugin's secret store: one private JSON file, checked like the permission record. */
export class SessionFile implements SecretStore {
  readonly directory: string;
  readonly file: string;
  constructor(storage = process.env.TENSORVIZ_DATA_DIR ?? applicationData()) {
    this.directory = path.join(storage, 'website');
    this.file = path.join(this.directory, 'session.json');
  }

  private async checkDirectory(): Promise<void> {
    const stat = await fs.lstat(this.directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || process.getuid && stat.uid !== process.getuid()
      || process.platform !== 'win32' && (stat.mode & 0o077)) {
      throw new WebsiteError('setup', `The tensorViz sign-in folder ${this.directory} must be private and owned by you.`);
    }
  }

  private async read(): Promise<Record<string, string>> {
    try { await this.checkDirectory(); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {}; throw error; }
    let handle;
    try {
      handle = await fs.open(this.file, constants.O_RDONLY | constants.O_NOFOLLOW);
      const stat = await handle.stat();
      if (!stat.isFile() || stat.nlink !== 1 || stat.size > SESSION_FILE_MAX_BYTES || process.getuid && stat.uid !== process.getuid()
        || process.platform !== 'win32' && (stat.mode & 0o077)) {
        throw new WebsiteError('setup', `The tensorViz sign-in file ${this.file} must be a private, regular file owned by you.`);
      }
      const value = JSON.parse(await handle.readFile('utf8'));
      if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
      return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, string] => typeof entry[1] === 'string'));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT' || error instanceof SyntaxError) return {};
      throw error;
    } finally { await handle?.close(); }
  }

  private async write(entries: Record<string, string>): Promise<void> {
    await fs.mkdir(this.directory, { recursive: true, mode: 0o700 });
    await this.checkDirectory();
    if (!Object.keys(entries).length) { await fs.rm(this.file, { force: true }); return; }
    const temporary = path.join(this.directory, `${randomUUID()}.tmp`);
    try {
      await fs.writeFile(temporary, JSON.stringify(entries) + '\n', { flag: 'wx', mode: 0o600 });
      await fs.rename(temporary, this.file);
    } finally { await fs.rm(temporary, { force: true }); }
  }

  async get(key: string): Promise<string | undefined> { return (await this.read())[key]; }

  async store(key: string, value: string): Promise<void> { await this.write({ ...await this.read(), [key]: value }); }

  async delete(key: string): Promise<void> {
    const entries = await this.read();
    if (!(key in entries)) return;
    delete entries[key];
    await this.write(entries);
  }
}

const page = (title: string, text: string) => `<!doctype html><html lang="en"><meta charset="utf-8"><title>${title}</title>`
  + `<body style="font:16px system-ui;max-width:640px;margin:64px auto;padding:0 24px"><h1>${title}</h1><p>${text}</p></body></html>`;

/** One sign-in's loopback callback. It answers only its own path, on its own host name,
 *  and leaves the ID token out of the page the browser keeps in its history. */
class SignInCallback {
  private lingering?: ReturnType<typeof setTimeout>;
  private closed = false;

  private constructor(private readonly server: Server, private readonly onClose: () => void) {}

  static async start(accept: (query: string) => boolean, onClose: (callback: SignInCallback) => void): Promise<{ callback: SignInCallback; url: string }> {
    let origin = '';
    const server = createServer((request, response) => {
      response.setHeader('Cache-Control', 'no-store');
      response.setHeader('Referrer-Policy', 'no-referrer');
      response.setHeader('X-Content-Type-Options', 'nosniff');
      response.setHeader('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'");
      if (request.method !== 'GET' || request.headers.host !== new URL(origin).host) { response.writeHead(403).end(); return; }
      const url = new URL(request.url ?? '/', origin);
      if (url.pathname === CALLBACK_PATH) {
        const accepted = accept(url.search.slice(1));
        response.writeHead(303, { Location: accepted ? SIGNED_IN_PATH : EXPIRED_PATH }).end();
        return;
      }
      const text = url.pathname === SIGNED_IN_PATH
        ? page('Signed in to tensorViz', 'Return to your graph: it finishes publishing there. You can close this tab.')
        : url.pathname === EXPIRED_PATH
          ? page('This sign-in has ended', 'It expired, was cancelled or was already used. Choose Publish in your graph to sign in again.')
          : undefined;
      if (!text) { response.writeHead(404).end(); return; }
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }).end(text);
    });
    server.requestTimeout = 10_000;
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const address = server.address();
    if (!address || typeof address === 'string') {
      server.close();
      throw new WebsiteError('unavailable', 'The sign-in callback could not start on this computer.');
    }
    origin = `http://127.0.0.1:${address.port}`;
    const callback: SignInCallback = new SignInCallback(server, () => onClose(callback));
    return { callback, url: `${origin}${CALLBACK_PATH}` };
  }

  /** Keeps answering briefly so the browser can load the closing page, then stops. */
  finish(): void {
    if (this.closed || this.lingering) return;
    this.lingering = setTimeout(() => this.close(), CALLBACK_LINGER_MS);
    this.lingering.unref();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.lingering);
    this.server.closeAllConnections();
    this.server.close();
    this.onClose();
  }
}

const cancellation = (signal: AbortSignal): Cancellation => ({
  get isCancellationRequested() { return signal.aborted; },
  onCancellationRequested: listener => signal.addEventListener('abort', listener, { once: true }),
});

/** The view's graph file: gzipped JSON in base64, never inflated past the website's limit. */
export function publishedFile(file: string): Uint8Array {
  let bytes: Buffer;
  try { bytes = gunzipSync(Buffer.from(file, 'base64'), { maxOutputLength: SHARE_LIMITS.bytes + 1 }); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ERR_BUFFER_TOO_LARGE') throw new WebsiteError('invalid', 'tensorViz takes graph files up to 10 MB.');
    throw new WebsiteError('invalid', 'This is not a tensorViz graph file.');
  }
  return bytes;
}

export interface PublishProgress {
  /** Hands the connect page to the asking view. False when no view can show it. */
  signingIn(url: string, detail: string): boolean;
  uploading(detail: string): void;
}

export interface WebsitePublisherOptions {
  storage?: string;
  fetch?: WebsiteFetch;
  env?: Record<string, string | undefined>;
  now?: () => number;
}

/** One publish at a time per plugin host, as in the editor. */
export class WebsitePublisher {
  readonly account: WebsiteAccount;
  private readonly callbacks = new Set<SignInCallback>();
  private signIn?: (url: string) => boolean;
  private busy = false;

  constructor(private readonly options: WebsitePublisherOptions = {}) {
    this.account = new WebsiteAccount({
      secrets: new SessionFile(options.storage),
      callback: async () => {
        const { callback, url } = await SignInCallback.start(query => {
          const accepted = this.account.handleUri('/auth', query);
          if (accepted) callback.finish();
          return accepted;
        }, closed => this.callbacks.delete(closed));
        this.callbacks.add(callback);
        return url;
      },
      openExternal: async url => this.signIn?.(url) ?? false,
      fetch: options.fetch, env: options.env, now: options.now, wording: WORDING,
    }, WEBSITE);
  }

  get publishing(): boolean { return this.busy; }

  /** Signs the member in through the asking view when needed, then adds the graph to
   *  their account as private. Cancelling stops a sign-in; an upload already sent finishes. */
  async publish(file: string, progress: PublishProgress, signal: AbortSignal): Promise<LocalBrowserPublishedGraph> {
    if (this.busy) throw new WebsiteError('too-soon', 'tensorViz is still publishing your last graph.');
    this.busy = true;
    try {
      const graph = shareableGraph(publishedFile(file));
      if (!await this.account.session()) {
        this.signIn = url => progress.signingIn(url, `Sign in to tensorViz in your browser to publish ${graph.title}.`);
        try { await this.account.signIn(cancellation(signal)); }
        finally {
          this.signIn = undefined;
          for (const callback of this.callbacks) callback.finish();
        }
      }
      if (signal.aborted) throw new WebsiteError('cancelled', 'Publishing was cancelled.');
      progress.uploading(`Publishing ${graph.title} to tensorViz…`);
      const id = await shareGraph(this.account, graph, this.options.fetch);
      const config = this.account.config;
      return { id, title: graph.title, url: graphUrl(config, id), shareUrl: graphUrl(config, id, true) };
    } finally { this.busy = false; }
  }

  async signOut(): Promise<boolean> { return this.account.signOut(); }

  dispose(): void {
    for (const callback of [...this.callbacks]) callback.close();
  }
}
