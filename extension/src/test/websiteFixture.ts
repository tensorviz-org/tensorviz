/** A fake Firebase and VS Code host for the tensorViz website tests: secret storage as a
 *  map, a browser that records what it opened, and Firebase Authentication answering with
 *  tokens whose claims the test chooses. */
import * as assert from 'node:assert/strict';
import { WEBSITE, WebsiteAccount, type AccountHost, type Cancellation, type WebsiteFetch, type WebsiteSession } from '../websiteAccount';

// Only the claims of a Firebase ID token matter here; Firebase verifies the signature.
export const firebaseToken = (claims: Record<string, unknown>) => `e30.${Buffer.from(JSON.stringify(claims)).toString('base64url')}.signature`;
export const google = { user_id: 'uid-1', name: 'Ada Lovelace', email: 'ada@example.com', picture: 'https://example.com/ada.png',
  firebase: { sign_in_provider: 'google.com' } };
export const CALLBACK = 'vscode://tensorviz.tensorviz-vscode/auth';
// The website's connect.js accepts only states of this shape.
export const WEBSITE_STATE = /^[A-Za-z0-9_-]{16,128}$/;
export const SECRET = `tensorviz.website.session.${WEBSITE.projectId}`;

type Responder = (url: string, init?: RequestInit) => Response | undefined;

export function setup(respond: Responder = () => undefined) {
  const secrets = new Map<string, string>();
  const opened: string[] = [];
  const seen: Array<{ url: string; init?: RequestInit }> = [];
  let clock = 1_000_000_000;
  let claims: Record<string, unknown> = google;
  const fetcher: WebsiteFetch = async (input, init) => {
    const url = String(input);
    seen.push({ url, init });
    const answer = respond(url, init);
    if (answer) return answer;
    if (url.includes('accounts:signInWithIdp')) return Response.json({ idToken: firebaseToken(claims), refreshToken: 'refresh-1', expiresIn: '3600', localId: 'uid-1' });
    if (url.includes('securetoken')) return Response.json({ id_token: firebaseToken(claims), refresh_token: 'refresh-1', expires_in: '3600', user_id: 'uid-1' });
    return new Response('{}', { status: 404 });
  };
  const host: AccountHost = {
    secrets: { get: async key => secrets.get(key), store: async (key, value) => { secrets.set(key, value); }, delete: async key => { secrets.delete(key); } },
    callback: async () => CALLBACK,
    openExternal: async url => { opened.push(url); return true; },
    fetch: fetcher, now: () => clock, env: {},
  };
  return {
    secrets, opened, seen, fetcher, host,
    account: new WebsiteAccount(host),
    tick: (ms: number) => { clock += ms; },
    now: () => clock,
    setClaims: (next: Record<string, unknown>) => { claims = next; },
  };
}

export const settled = () => new Promise(resolve => setImmediate(resolve));

/** Starts a sign-in and answers it the way the website's connect page does. */
export async function signIn(world: ReturnType<typeof setup>): Promise<WebsiteSession> {
  const pending = world.account.signIn();
  await settled();
  const state = new URL(world.opened.at(-1)!).searchParams.get('state')!;
  assert.ok(world.account.handleUri('/auth', `state=${state}&id_token=google-id-token`));
  return pending;
}

export function graphFile(nodes: number, name = 'TinyNet', extra: Record<string, unknown> = {}): Buffer {
  return Buffer.from(JSON.stringify({ format: 'tensorviz.graph', version: 3, status: 'static',
    document: { semantic: { name, nodes: Array.from({ length: nodes }, (_, index) => ({ node_id: `layer${index}` })), ...extra } } }));
}

export function cancellation(): Cancellation & { cancel(): void } {
  const listeners: Array<() => void> = [];
  return {
    isCancellationRequested: false,
    onCancellationRequested(listener) { listeners.push(listener); },
    cancel() { this.isCancellationRequested = true; listeners.forEach(listener => listener()); },
  };
}
