import { loopbackOrigin } from '../../remoteSupport';
import * as assert from 'node:assert/strict';
import { clientOrigin, forwardShellTarget } from '../../remoteForwarding';
import { CanvasBridge, shellHtml } from '../../shell';
const vscode = require('../vscodeStub');

suite('desktop SSH forwarding', () => {
  const restore: Array<() => void> = [];
  const calls: string[] = [];
  let forwarded: string;
  function patch(object: object, key: string, value: unknown) {
    const old = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, value });
    restore.push(() => old ? Object.defineProperty(object, key, old) : Reflect.deleteProperty(object, key));
  }
  setup(() => {
    calls.length = 0; forwarded = 'http://127.0.0.1:43127/';
    patch(vscode, 'env', { remoteName: 'ssh-remote' });
    patch(vscode.Uri, 'parse', (value: string) => ({ toString: () => value }));
    patch(vscode.env, 'asExternalUri', async (uri: { toString(): string }) => {
      calls.push(uri.toString()); return { toString: () => forwarded };
    });
  });
  teardown(() => { for (const undo of restore.splice(0).reverse()) undo(); });

  test('iframe URL, CSP and postMessage agree with the actual forwarded port', async () => {
    const target = await forwardShellTarget({ appUrl: 'http://127.0.0.1:8631/?token=fixture',
      appOrigin: 'http://127.0.0.1:8631', port: 8631, workspaceDir: '/remote/project' });
    assert.deepEqual(calls, ['http://127.0.0.1:8631'], 'forwarding never receives launch credentials');
    assert.equal(target.appUrl, 'http://127.0.0.1:43127/?token=fixture');
    const html = shellHtml(target);
    assert.match(html, /frame-src http:\/\/127\.0\.0\.1:43127;/);
    assert.match(html, /APP_ORIGIN = "http:\/\/127\.0\.0\.1:43127"/);
    assert.deepEqual(CanvasBridge.options(target.port, target.forwarded).portMapping, [], 'a resolved URL must not be forwarded a second time');
    assert.deepEqual(CanvasBridge.options(8631).portMapping, [{ webviewPort: 8631, extensionHostPort: 8631 }]);
  });

  test('explicit reopen re-resolves a closed or reassigned tunnel', async () => {
    assert.equal(await clientOrigin('http://127.0.0.1:8631'), 'http://127.0.0.1:43127');
    forwarded = 'http://localhost:53128';
    assert.equal(await clientOrigin('http://127.0.0.1:8631'), forwarded);
    assert.equal(calls.length, 2);
  });

  test('public/path-based forwarding and unsupported remote kinds fail closed', async () => {
    for (const value of ['https://public.example', 'http://0.0.0.0:8631', 'http://127.0.0.1:8631/proxy',
      'http://user:secret@localhost:8631', 'http://localhost:8631/?auth=x', 'file:///tmp/page']) {
      assert.throws(() => loopbackOrigin(value), /private loopback/);
    }
    forwarded = 'https://public.example';
    await assert.rejects(clientOrigin('http://127.0.0.1:8631'), /private loopback/);
    patch(vscode.env, 'remoteName', 'wsl');
    await assert.rejects(clientOrigin('http://127.0.0.1:8631'), /not supported/);
  });

  test('local desktop keeps its existing origin without requesting forwarding', async () => {
    patch(vscode.env, 'remoteName', undefined);
    assert.equal(await clientOrigin('http://localhost:3000'), 'http://localhost:3000');
    assert.deepEqual(calls, []);
  });
});
