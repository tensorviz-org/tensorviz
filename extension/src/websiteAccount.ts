/** The member's tensorViz website account, which Add to tensorViz sends graphs to.
 *
 * Sign-in goes through the website. The extension opens tensorviz.dev/explore/connect/
 * with a random state and a callback to its own URI handler; the member picks a Google
 * account there, and the page sends back that state with a Google ID token, valid for an
 * hour. The extension exchanges the token with Firebase Authentication for its own
 * session and keeps only the refresh token, in VS Code's secret storage, which every
 * window shares. Nothing here imports `vscode`; websiteCommands.ts wires it up. See
 * docs/share-to-website.md.
 */
import { randomBytes, timingSafeEqual } from 'node:crypto';

export interface WebsiteConfig {
  /** Origin of the website: its connect page and its graph pages. */
  site: string;
  /** The Firebase web app's public identifiers. The website's security rules, not these,
   *  protect members' data. */
  apiKey: string;
  projectId: string;
  /** REST roots of Firebase Authentication, which the emulators replace. Graphs go to
   *  the website's own upload API, at `${site}/api/`. */
  identity: string;
  secureToken: string;
}

/** The website's own Firebase web app (src/explorer/community-config.js in its repository). */
export const WEBSITE: WebsiteConfig = {
  site: 'https://tensorviz.dev',
  apiKey: 'AIzaSyBB-uxMsrJ4lzA_ywP0JdZMxM0vsPNoxxU',
  projectId: 'project-fe5816f4-14e7-42f8-aa0',
  identity: 'https://identitytoolkit.googleapis.com',
  secureToken: 'https://securetoken.googleapis.com',
};

export const EMULATOR_VARIABLE = 'TENSORVIZ_WEBSITE_EMULATOR';
/** Development: the website's dev server in front of the real project, to try a website
 *  branch from the editor with a real account before it ships. */
export const SITE_VARIABLE = 'TENSORVIZ_WEBSITE_URL';

/** `cancelled` is the member's own doing (or a newer sign-in) and needs no message. */
export type WebsiteErrorCode = 'setup' | 'cancelled' | 'expired' | 'signed-out' | 'denied' | 'too-soon' | 'invalid' | 'unavailable';

export class WebsiteError extends Error {
  constructor(readonly code: WebsiteErrorCode, message: string) {
    super(message);
    this.name = 'WebsiteError';
  }
}

/** A development variable's loopback http origin, or a setup error naming it. */
function loopbackOrigin(name: string, value: string, example: string): string {
  let url: URL | undefined;
  try { url = new URL(value); } catch { url = undefined; }
  if (!url || url.protocol !== 'http:' || !['localhost', '127.0.0.1'].includes(url.hostname) || url.pathname !== '/'
      || url.search || url.hash || url.username || url.password) {
    throw new WebsiteError('setup', `${name} must be the website's local dev server, such as ${example}.`);
  }
  return url.origin;
}

/** The production website or, for development, the website's dev server: with its Firebase
 *  emulators through TENSORVIZ_WEBSITE_EMULATOR (`http://localhost:5174`, whose /api is the
 *  Functions emulator), or in front of the real project through TENSORVIZ_WEBSITE_URL
 *  (`http://localhost:5173`). Only loopback origins are accepted, so neither variable can
 *  send a sign-in off this machine. */
export function websiteConfig(env: Record<string, string | undefined> = process.env): WebsiteConfig {
  const emulator = env[EMULATOR_VARIABLE];
  const site = env[SITE_VARIABLE];
  if (emulator && site) throw new WebsiteError('setup', `Set ${EMULATOR_VARIABLE} or ${SITE_VARIABLE}, not both.`);
  if (site) return { ...WEBSITE, site: loopbackOrigin(SITE_VARIABLE, site, 'http://localhost:5173') };
  if (!emulator) return WEBSITE;
  return {
    site: loopbackOrigin(EMULATOR_VARIABLE, emulator, 'http://localhost:5174'), apiKey: 'demo-key', projectId: 'demo-tensorviz',
    identity: 'http://127.0.0.1:9099/identitytoolkit.googleapis.com',
    secureToken: 'http://127.0.0.1:9099/securetoken.googleapis.com',
  };
}

/** Where the website sends the member back: this extension's `/auth` URI in the running
 *  editor. Deliberately not vscode.env.asExternalUri: the `?windowId=` it adds would put a
 *  `?` inside the connect link's query, which vscode.env.openExternal mangles (it escapes
 *  the `?` in Uri.toString(true), then encodeURI escapes that again), so the page refused
 *  the link. VS Code hands the reply to the last active window instead. */
export function editorCallback(uriScheme: string, extensionId: string): string {
  return `${uriScheme}://${extensionId}/auth`;
}

/** The connect page for one sign-in. The website sends the member back only to callbacks
 *  it recognises (its connect.js), with this state. */
export function connectUrl(config: WebsiteConfig, callback: string, state: string): string {
  return `${config.site}/explore/connect/?${new URLSearchParams({ state, callback })}`;
}

export type WebsiteFetch = typeof fetch;
const TIMEOUT_MS = 30_000;

/** One request to Firebase Authentication or the tensorViz API. A network failure is a
 *  WebsiteError; HTTP errors are the caller's. */
export async function websiteRequest(fetcher: WebsiteFetch, url: string, init: RequestInit, timeoutMs = TIMEOUT_MS): Promise<{ status: number; body: any }> {
  let response: Response;
  try {
    response = await fetcher(url, { ...init, redirect: 'error', signal: AbortSignal.timeout(timeoutMs) });
  } catch {
    throw new WebsiteError('unavailable', 'Could not reach tensorViz. Check your connection and try again.');
  }
  return { status: response.status, body: await response.json().catch(() => undefined) };
}

/** Firebase's error code, such as TOKEN_EXPIRED from "TOKEN_EXPIRED : details". */
export function firebaseCode(body: any): string {
  const message = body?.error?.message;
  return typeof message === 'string' ? message.split(/[\s:]/, 1)[0] : '';
}

/** The claims of a Firebase ID token, read without verifying it: Firebase verifies it on
 *  every request, and the website's rules compare a post's author with these claims. */
export function tokenClaims(idToken: string): Record<string, unknown> {
  try {
    const claims = JSON.parse(Buffer.from(idToken.split('.')[1] ?? '', 'base64url').toString('utf8'));
    if (claims && typeof claims === 'object' && !Array.isArray(claims)) return claims;
  } catch { /* reported below */ }
  throw new WebsiteError('invalid', 'tensorViz sent back a sign-in VS Code could not read. Try again.');
}

export interface WebsiteMember {
  uid: string;
  /** The Google account's name and email, for "Signed in as". */
  name: string | null;
  email: string | null;
}

export interface WebsiteSession extends WebsiteMember {
  idToken: string;
  expiresAt: number;
  /** What the rules expect as a post's author name and photo: the token's own claims. */
  authorName: string;
  authorPhoto: string | null;
}

interface Stored extends WebsiteMember { refreshToken: string }

/** The rules' fallback when a Google account has no display name. */
const NO_NAME = 'tensorViz member';
const SECRET = 'tensorviz.website.session';
const SIGN_IN_MS = 10 * 60_000;
/** Refresh an ID token this long before it expires, so an upload never carries a dying one. */
const REFRESH_EARLY_MS = 5 * 60_000;
/** A refresh token Firebase no longer honours: the member must sign in again. */
const SESSION_GONE = new Set(['TOKEN_EXPIRED', 'INVALID_REFRESH_TOKEN', 'USER_NOT_FOUND', 'USER_DISABLED', 'INVALID_ID_TOKEN']);

export interface Cancellation {
  isCancellationRequested: boolean;
  onCancellationRequested(listener: () => void): unknown;
}

export interface AccountHost {
  /** VS Code's SecretStorage. */
  secrets: {
    get(key: string): PromiseLike<string | undefined>;
    store(key: string, value: string): PromiseLike<void>;
    delete(key: string): PromiseLike<void>;
  };
  /** This extension's `/auth` URI as the browser must open it (vscode.env.asExternalUri). */
  callback(): Promise<string>;
  openExternal(url: string): Promise<boolean>;
  fetch?: WebsiteFetch;
  now?: () => number;
  env?: Record<string, string | undefined>;
}

type Outcome = { googleIdToken: string } | { error: WebsiteError };

export class WebsiteAccount {
  private pending?: { state: string; settle(outcome: Outcome): void };
  private current?: WebsiteSession;
  private refreshing?: Promise<WebsiteSession | undefined>;

  constructor(private readonly host: AccountHost) {}

  /** Read per use, so a bad emulator variable is reported where the member acts. */
  get config(): WebsiteConfig {
    return websiteConfig(this.host.env);
  }

  private get fetcher(): WebsiteFetch {
    return this.host.fetch ?? fetch;
  }

  private now(): number {
    return (this.host.now ?? Date.now)();
  }

  /** Each Firebase project keeps its own sign-in, so an emulator session never reaches tensorviz.dev. */
  private key(config = this.config): string {
    return `${SECRET}.${config.projectId}`;
  }

  /** True for the secret this class keeps; another window changed it when it fires. */
  owns(secretKey: string): boolean {
    return secretKey.startsWith(`${SECRET}.`);
  }

  /** Drops the cached ID token; the next session() reads the stored sign-in again. */
  forget(): void {
    this.current = undefined;
  }

  /** The signed-in member as stored, without a network request. */
  async member(): Promise<WebsiteMember | undefined> {
    const stored = await this.stored();
    return stored && { uid: stored.uid, name: stored.name, email: stored.email };
  }

  /** A session good for at least five more minutes, or undefined when signed out. */
  async session(): Promise<WebsiteSession | undefined> {
    if (this.current && this.current.expiresAt - this.now() > REFRESH_EARLY_MS) return this.current;
    this.refreshing ??= this.refresh().finally(() => { this.refreshing = undefined; });
    return this.refreshing;
  }

  /** One browser confirmation per sign-in. A second caller joins it instead of opening another page. */
  private signingIn?: Promise<WebsiteSession>;

  /** Opens the connect page and waits, up to ten minutes, for the website's callback. */
  signIn(cancel?: Cancellation): Promise<WebsiteSession> {
    if (this.signingIn) return this.signingIn;
    const attempt = this.beginSignIn(cancel).finally(() => {
      if (this.signingIn === attempt) this.signingIn = undefined;
    });
    this.signingIn = attempt;
    return attempt;
  }

  private async beginSignIn(cancel?: Cancellation): Promise<WebsiteSession> {
    const config = this.config;
    const state = randomBytes(24).toString('base64url');
    const page = connectUrl(config, await this.host.callback(), state);
    const googleIdToken = await new Promise<string>((resolve, reject) => {
      this.pending?.settle({ error: new WebsiteError('cancelled', 'A newer sign-in replaced this one.') });
      const settle = (outcome: Outcome): void => {
        if (this.pending?.state !== state) return;
        this.pending = undefined;
        clearTimeout(timer);
        if ('error' in outcome) reject(outcome.error); else resolve(outcome.googleIdToken);
      };
      const timer = setTimeout(() => settle({ error: new WebsiteError('expired', 'Sign-in timed out. Run "Sign in to tensorViz" to try again.') }), SIGN_IN_MS);
      this.pending = { state, settle };
      if (cancel?.isCancellationRequested) settle({ error: new WebsiteError('cancelled', 'Sign-in was cancelled.') });
      cancel?.onCancellationRequested(() => settle({ error: new WebsiteError('cancelled', 'Sign-in was cancelled.') }));
      this.host.openExternal(page).then(
        opened => { if (!opened) settle({ error: new WebsiteError('unavailable', 'The browser did not open, so sign-in stopped.') }); },
        () => settle({ error: new WebsiteError('unavailable', 'The browser did not open, so sign-in stopped.') }));
    });
    return this.exchange(config, googleIdToken);
  }

  /** The website's callback, `/auth?state=…&id_token=…`. False when it does not answer the
   *  sign-in this window started: an old link, or one VS Code routed to another window. */
  handleUri(path: string, query: string): boolean {
    if (path !== '/auth' || !this.pending) return false;
    const params = new URLSearchParams(query);
    const state = Buffer.from(params.get('state') ?? '');
    const expected = Buffer.from(this.pending.state);
    if (state.length !== expected.length || !timingSafeEqual(state, expected)) return false;
    const googleIdToken = params.get('id_token') ?? '';
    // A Google ID token; the Auth emulator's stand-in is JSON. Firebase checks either.
    this.pending.settle(googleIdToken && googleIdToken.length <= 16_384 && !/[\x00-\x1f\x7f]/.test(googleIdToken)
      ? { googleIdToken }
      : { error: new WebsiteError('invalid', 'tensorViz sent back a sign-in VS Code could not read. Try again.') });
    return true;
  }

  /** Forgets the sign-in on this computer. Returns whether there was one. */
  async signOut(): Promise<boolean> {
    const had = (await this.stored()) !== undefined;
    this.current = undefined;
    await this.host.secrets.delete(this.key());
    return had;
  }

  private async exchange(config: WebsiteConfig, googleIdToken: string): Promise<WebsiteSession> {
    const { status, body } = await websiteRequest(this.fetcher, `${config.identity}/v1/accounts:signInWithIdp?key=${encodeURIComponent(config.apiKey)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // requestUri is required and unused for an ID token; Firebase's own SDK sends this one.
      body: JSON.stringify({ postBody: `id_token=${encodeURIComponent(googleIdToken)}&providerId=google.com`,
        requestUri: 'http://localhost', returnSecureToken: true }),
    });
    if (status !== 200 || body?.needConfirmation) {
      const code = firebaseCode(body);
      if (code === 'OPERATION_NOT_ALLOWED') throw new WebsiteError('setup', 'Google sign-in is turned off for tensorViz.');
      if (code === 'USER_DISABLED') throw new WebsiteError('denied', 'This tensorViz account is disabled.');
      throw new WebsiteError(status >= 500 ? 'unavailable' : 'denied', 'Google could not confirm this sign-in. Run "Sign in to tensorViz" again.');
    }
    return this.keep(config, body.idToken, body.refreshToken, body.expiresIn);
  }

  private async refresh(): Promise<WebsiteSession | undefined> {
    const config = this.config;
    const stored = await this.stored(config);
    if (!stored) return undefined;
    const { status, body } = await websiteRequest(this.fetcher, `${config.secureToken}/v1/token?key=${encodeURIComponent(config.apiKey)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ grant_type: 'refresh_token', refresh_token: stored.refreshToken }).toString(),
    });
    if (status !== 200) {
      if (status === 400 && SESSION_GONE.has(firebaseCode(body))) {
        this.current = undefined;
        await this.host.secrets.delete(this.key(config));
        return undefined;
      }
      throw new WebsiteError(status >= 500 ? 'unavailable' : 'denied', 'tensorViz did not renew your sign-in. Try again shortly.');
    }
    return this.keep(config, body.id_token, body.refresh_token, body.expires_in, stored);
  }

  /** Checks the tokens Firebase returned, caches the ID token and stores the refresh token. */
  private async keep(config: WebsiteConfig, idToken: unknown, refreshToken: unknown, expiresIn: unknown, stored?: Stored): Promise<WebsiteSession> {
    const seconds = Number(expiresIn);
    if (typeof idToken !== 'string' || typeof refreshToken !== 'string' || !refreshToken || !(seconds > 0)) {
      throw new WebsiteError('invalid', 'tensorViz sent back a sign-in VS Code could not read. Try again.');
    }
    const claims = tokenClaims(idToken);
    const uid = claims.user_id ?? claims.sub;
    if (typeof uid !== 'string' || !uid) throw new WebsiteError('invalid', 'tensorViz sent back a sign-in VS Code could not read. Try again.');
    // The website's rules accept posts from Google accounts only.
    if ((claims.firebase as { sign_in_provider?: unknown } | undefined)?.sign_in_provider !== 'google.com') {
      throw new WebsiteError('denied', 'tensorViz takes graphs from Google accounts only.');
    }
    const name = typeof claims.name === 'string' ? claims.name : null;
    const email = typeof claims.email === 'string' ? claims.email : null;
    const session: WebsiteSession = {
      uid, name, email, idToken, expiresAt: this.now() + seconds * 1000,
      authorName: name ?? NO_NAME, authorPhoto: typeof claims.picture === 'string' ? claims.picture : null,
    };
    if (!stored || stored.uid !== uid || stored.refreshToken !== refreshToken || stored.name !== name || stored.email !== email) {
      const next: Stored = { uid, name, email, refreshToken };
      await this.host.secrets.store(this.key(config), JSON.stringify(next));
    }
    this.current = session;
    return session;
  }

  private async stored(config = this.config): Promise<Stored | undefined> {
    const text = await this.host.secrets.get(this.key(config));
    if (!text) return undefined;
    try {
      const value = JSON.parse(text);
      if (typeof value?.uid === 'string' && typeof value.refreshToken === 'string') {
        return { uid: value.uid, refreshToken: value.refreshToken,
          name: typeof value.name === 'string' ? value.name : null, email: typeof value.email === 'string' ? value.email : null };
      }
    } catch { /* an unreadable entry is a signed-out one */ }
    return undefined;
  }
}
