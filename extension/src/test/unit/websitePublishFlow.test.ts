import * as assert from 'node:assert/strict';
import { mock } from 'node:test';
import { gunzipSync } from 'node:zlib';
import * as vscode from 'vscode';
import { registerWebsiteCommands } from '../../websiteCommands';
import { SHARE_GRAPH_FILE, SIGN_IN_TO_WEBSITE } from '../../commandIds';
import { SECRET, CALLBACK, WEBSITE_STATE, graphFile, setup, settled } from '../websiteFixture';

const GRAPH_ID = 'AbCdEfGhIjKlMnOpQrSt';

suite('website Publish mock end-to-end', () => {
  for (const scenario of ['success', '401', 'expired token', 'timeout retry', 'signed-out retry', 'sign-in retry'] as const) {
    test(scenario, async () => {
      let posts = 0;
      const rejectedPosts = scenario === '401' ? 1 : scenario === 'signed-out retry' ? 2 : 0;
      const world = setup(url => {
        if (!url.endsWith('/api/graphs')) return undefined;
        return ++posts <= rejectedPosts
          ? Response.json({}, { status: 401 })
          : Response.json({ id: GRAPH_ID, thread: null }, { status: 201 });
      });
      const commands = new Map<string, (...args: any[]) => any>();
      const opened: string[] = [];
      const logs: string[] = [];
      const errors: string[] = [];
      let handler: vscode.UriHandler;
      let timeout: (() => void) | undefined;
      let connects = 0;
      const win = vscode.window as any;
      const api = vscode as any;
      const saved = new Map<string, any>();
      const install = (key: string, value: any) => { saved.set(key, api[key]); api[key] = value; };
      const oldHandler = win.registerUriHandler;
      const oldProgress = win.withProgress;
      try {
        install('env', { uriScheme: 'vscode', uiKind: 1, openExternal: async (uri: vscode.Uri) => {
          const url = new URL(uri.toString());
          opened.push(url.href);
          if (!url.pathname.endsWith('/connect/')) return true;
          connects++;
          if (connects > 1) world.setClaims({ user_id: 'uid-1', firebase: { sign_in_provider: 'google.com' } });
          const state = url.searchParams.get('state')!;
          assert.match(state, WEBSITE_STATE);
          assert.equal(url.searchParams.get('callback'), CALLBACK);
          // connectTarget permits this scheme, authority, exact path and empty query.
          const callback = new URL(url.searchParams.get('callback')!);
          assert.equal(callback.protocol, 'vscode:');
          assert.equal(callback.hostname, 'tensorviz.tensorviz-vscode');
          assert.equal(callback.pathname, '/auth');
          assert.equal(callback.search, '');
          if ((scenario === 'timeout retry' || scenario === 'sign-in retry') && connects === 1) {
            queueMicrotask(() => timeout!());
          } else {
            handler.handleUri({ path: 'auth', query: `?windowId=123&state=${state}&id_token=google-id-token` } as vscode.Uri);
          }
          return true;
        } });
        install('UIKind', { Web: 2 });
        install('ProgressLocation', { Notification: 15 });
        win.registerUriHandler = (next: vscode.UriHandler) => { handler = next; return { dispose() {} }; };
        win.withProgress = async (_options: unknown, run: Function) => run({}, { isCancellationRequested: false, onCancellationRequested() {} });
        mock.method(vscode.commands, 'registerCommand', (id: string, run: (...args: any[]) => any) => {
          commands.set(id, run); return { dispose() {} };
        });
        mock.method(win, 'showInformationMessage', async () => undefined);
        mock.method(win, 'showErrorMessage', async (message: string, options: unknown, ...actions: string[]) => {
          errors.push(message);
          assert.deepEqual(options, { modal: true });
          assert.deepEqual(actions, ['Retry']);
          return errors.length === 1 ? 'Retry' : undefined;
        });
        mock.method(globalThis, 'fetch', world.fetcher);
        const originalTimer = globalThis.setTimeout;
        mock.method(globalThis, 'setTimeout', ((fn: () => void, ms: number, ...args: any[]) => {
          if (ms === 600_000) timeout = fn;
          return originalTimer(fn, ms, ...args);
        }) as typeof setTimeout);
        const context = { secrets: { ...world.host.secrets, onDidChange: () => ({ dispose() {} }) },
          extension: { id: 'tensorviz.tensorviz-vscode' }, subscriptions: [] } as unknown as vscode.ExtensionContext;
        registerWebsiteCommands(context, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel);
        if (scenario === 'expired token') {
          world.secrets.set(SECRET, JSON.stringify({ uid: 'uid-1', refreshToken: 'refresh-1' }));
        }
        await commands.get(scenario === 'sign-in retry' ? SIGN_IN_TO_WEBSITE : SHARE_GRAPH_FILE)!({ bytes: graphFile(2).toString('base64') });
        await settled();
        assert.equal(JSON.parse(world.secrets.get(SECRET)!).refreshToken, 'refresh-1');
        assert.doesNotMatch(world.secrets.get(SECRET)!, /google-id-token|\.signature/);
        const exchanges = world.seen.filter(request => request.url.includes('accounts:signInWithIdp'));
        assert.equal(exchanges.length, scenario === 'expired token' ? 0 : scenario === 'signed-out retry' ? 2 : 1);
        for (const exchange of exchanges) {
          assert.equal(JSON.parse(String(exchange.init?.body)).postBody, 'id_token=google-id-token&providerId=google.com');
        }
        if (scenario !== 'sign-in retry') {
          const uploads = world.seen.filter(request => request.url.endsWith('/api/graphs'));
          assert.equal(uploads.length, scenario === '401' ? 2 : scenario === 'signed-out retry' ? 3 : 1);
          for (const request of uploads) {
            assert.equal(request.init?.method, 'POST');
            assert.match((request.init?.headers as Record<string, string>).Authorization, /^Bearer e30\./);
            const body = JSON.parse(String(request.init?.body));
            assert.deepEqual(Object.keys(body).sort(), ['file', 'title', 'visibility']);
            assert.equal(body.visibility, 'private');
            assert.equal(body.title, 'TinyNet');
            assert.deepEqual(gunzipSync(Buffer.from(body.file, 'base64')), graphFile(2));
          }
          assert.equal(opened.at(-1), `https://tensorviz.dev/explore/graph/?id=${GRAPH_ID}&share=1`);
        }
        assert.equal(world.seen.filter(request => request.url.includes('securetoken')).length,
          ['401', 'signed-out retry', 'expired token'].includes(scenario) ? 1 : 0);
        assert.equal(errors.length, scenario.includes('retry') ? 1 : 0);
        assert.doesNotMatch(logs.join('\n'), /google-id-token|refresh-1|\.signature/);
      } finally {
        mock.restoreAll();
        win.registerUriHandler = oldHandler;
        win.withProgress = oldProgress;
        for (const [key, value] of saved) { if (value === undefined) delete api[key]; else api[key] = value; }
      }
    });
  }
});
