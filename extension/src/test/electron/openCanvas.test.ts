import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';

import * as vscode from 'vscode';

const EXTENSION_ID = 'tensorviz.tensorviz';

/** The checkout this test drives the extension against: three directories up from
 *  out/test/electron. */
const REPO = path.resolve(__dirname, '..', '..', '..', '..');

interface AppMessage {
  type: string;
  t?: number;
  mounted?: boolean;
  [key: string]: unknown;
}

interface TestApi {
  sidebarState(): { messages: AppMessage[]; ready: boolean; sessionId: string } | undefined;
  backendOrigin(): string | undefined;
  launchToken(): string | undefined;
  graphOpeningState(): { id: string; phase: string; detail: string; target: { cls: string } } | undefined;
  post(message: unknown): Promise<boolean>;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Hold 8765 for the duration, so "the backend picked a port" is a claim about
 *  behaviour and not about an empty machine. Already-held is the same precondition. */
function squat(port: number): Promise<net.Server | undefined> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(undefined));
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

suite('openCanvas', () => {
  let api: TestApi;
  let squatter: net.Server | undefined;
  let modelDir: string;
  const exportsDir = path.join(REPO, 'tensorviz-exports');
  const exportsDirExisted = fs.existsSync(exportsDir);
  const stateDir = path.join(REPO, '.tensorviz');
  // With no folder open the checkout IS the workspace, so the backend writes its
  // lock and state database here. Remove them afterwards only if this run is what
  // created the directory — a developer's own .tensorviz/views is not ours to delete.
  const stateDirExisted = fs.existsSync(stateDir);

  suiteSetup(async function () {
    this.timeout(240_000);

    // Never skipped. An Electron run without an interpreter proves nothing about
    // the backend, and a skipped test that reads as green is the failure mode
    // this rule exists to prevent (task.md M2, "Test infrastructure rule").
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'not run: TENSORVIZ_TEST_PYTHON unset');
    assert.ok(
      fs.existsSync(path.join(REPO, 'extension', 'media', 'app', 'index.html')),
      'not run: extension/media/app/index.html is missing — run `npm run build:webview` first',
    );

    // interpreterCandidates reads BACKEND_PYTHON from the extension host's own
    // environment, and this test runs in that process — so this is how the host
    // is told which interpreter to use with ms-python absent (--disable-extensions).
    process.env.BACKEND_PYTHON = python;

    squatter = await squat(8765);

    const extension = vscode.extensions.getExtension(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} was not loaded by the extension host`);
    api = ((await extension.activate()) as { __test: TestApi }).__test;

    // No folder is open in this run, so the checkout is both the repo and the
    // workspace; the setting is what stops resolveRepoPath falling to the picker.
    await vscode.workspace
      .getConfiguration('tensorviz')
      .update('repoPath', REPO, vscode.ConfigurationTarget.Global);

    await vscode.commands.executeCommand('tensorviz.openCanvas');
    assert.equal(api.backendOrigin(), undefined, 'opening the starting screen must not start the backend');
    // Empty Canvas is intentionally an instruction, not a mounted builder.
    // Select a disposable model through the public sidebar route for liveness.
    modelDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-canvas-model-'));
    const file = path.join(modelDir, 'model.py');
    fs.writeFileSync(file, 'import torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 4)\n    def forward(self, x):\n        return self.fc(x)\n');
    await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: vscode.Uri.file(file).toString(), cls: 'Net' });
    assert.ok(api.backendOrigin(), 'explicit graph opening must start the backend');
  });

  suiteTeardown(async () => {
    squatter?.close();
    if (modelDir) fs.rmSync(modelDir, { recursive: true, force: true });
    fs.rmSync(path.join(exportsDir, 'mlp'), { recursive: true, force: true });
    if (!exportsDirExisted) {
      fs.rmSync(exportsDir, { recursive: true, force: true });
    }
    if (!stateDirExisted) {
      fs.rmSync(stateDir, { recursive: true, force: true });
    } else {
      fs.rmSync(path.join(stateDir, 'lock'), { force: true });
    }
    await vscode.workspace
      .getConfiguration('tensorviz')
      .update('repoPath', undefined, vscode.ConfigurationTarget.Global);
    await vscode.commands.executeCommand('tensorviz.hideCanvas');
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');
  });

  test('starts the backend on a port 8765 is not holding, serving the app', async () => {
    const origin = api.backendOrigin();
    assert.ok(origin);
    assert.ok(!origin.endsWith(':8765'), `expected a port other than 8765, got ${origin}`);

    const health = await fetch(`${origin}/health`);
    assert.equal(health.status, 200);
    const body = (await health.json()) as { capabilities: { ide: boolean; static: boolean } };
    assert.equal(body.capabilities.static, true);
    assert.equal(body.capabilities.ide, true);
  });

  test('refuses /api/* without the per-launch token and answers with it', async () => {
    const origin = api.backendOrigin()!;
    const token = api.launchToken()!;

    const anonymous = await fetch(`${origin}/api/layer-types`);
    assert.equal(anonymous.status, 401);

    const authenticated = await fetch(`${origin}/api/layer-types`, {
      headers: { 'X-tensorViz-Token': token },
    });
    assert.equal(authenticated.status, 200);
  });

  test('refuses the in-process importer even to an authenticated caller', async () => {
    const origin = api.backendOrigin()!;
    const token = api.launchToken()!;
    const response = await fetch(`${origin}/api/ingest/repo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': token },
      body: JSON.stringify({ repo: REPO }),
    });
    assert.equal(response.status, 403);
    const body = (await response.json()) as { detail: { code: string } };
    assert.equal(body.detail.code, 'ide_mode_uses_worker');
  });

  test('the canvas is still mounted eight seconds after the app said it was ready', async function () {
    // The tldraw licence gate unmounts an unlicensed "production" editor five
    // seconds in (task.md §2.3). Stage A exists to avoid exactly that, and this is
    // the measurement inside VS Code — the app reports it, because neither the
    // extension host nor the shell can read the DOM of a cross-origin iframe.
    this.timeout(120_000);

    const deadline = Date.now() + 90_000;
    let ready: AppMessage | undefined;
    let mountedLate: AppMessage | undefined;
    while (Date.now() < deadline && !mountedLate) {
      const messages = api.sidebarState()?.messages ?? [];
      ready = ready ?? messages.find((m) => m.type === 'ready');
      if (ready?.t !== undefined) {
        mountedLate = messages.find(
          (m) => m.type === 'canvasState' && m.mounted === true && (m.t ?? 0) >= (ready!.t as number) + 8000,
        );
      }
      if (!mountedLate) {
        await sleep(500);
      }
    }

    assert.ok(ready, 'the app never posted {type:"ready"}');
    assert.ok(
      mountedLate,
      'the app never reported a mounted canvas 8s after ready — the tldraw licence gate unmounted it',
    );
  });

  test('a multi-file export lands in the workspace as files, and opens', async () => {
    const files = [
      { name: 'model.py', bytes: Buffer.from('import torch\n').toString('base64') },
      { name: 'sample_run.py', bytes: Buffer.from('print("hi")\n').toString('base64') },
    ];
    assert.equal(await api.post({ type: 'saveFiles', dir: 'mlp', files }), true);

    const dir = path.join(exportsDir, 'mlp');
    assert.equal(fs.readFileSync(path.join(dir, 'model.py'), 'utf8'), 'import torch\n');
    assert.equal(fs.readFileSync(path.join(dir, 'sample_run.py'), 'utf8'), 'print("hi")\n');
    // The point of saveFiles: the app hands over the files it has, so the user gets
    // a directory they can run, not an archive they have to unpack.
    assert.deepEqual(
      fs.readdirSync(dir).filter((name) => name.endsWith('.zip')),
      [],
    );

    assert.equal(
      vscode.window.activeTextEditor?.document.uri.fsPath,
      path.join(dir, 'model.py'),
      'the exported model.py was not opened beside the canvas',
    );
  });

  test('refuses a path that would escape the export directory', async () => {
    await api.post({
      type: 'saveFiles',
      dir: '../escaped',
      files: [{ name: 'model.py', bytes: Buffer.from('x').toString('base64') }],
    });
    assert.equal(fs.existsSync(path.join(REPO, '..', 'escaped')), false);
  });

  test('choosing the separate environment reconnects Canvas without another Open command', async function () {
    this.timeout(120_000);
    const previousToken = api.launchToken();
    const previousSession = api.sidebarState()?.sessionId;
    const previousOpening = api.graphOpeningState()?.id;
    const surface = vscode.window as unknown as Record<string, unknown>;
    const originalPicker = surface.showQuickPick, originalInfo = surface.showInformationMessage;
    surface.showQuickPick = async (items: ReadonlyArray<{ choice?: { kind: string } }>) => {
      const managed = items.find(item => item.choice?.kind === 'managed');
      assert.ok(managed, 'the separate environment must be available');
      return managed;
    };
    surface.showInformationMessage = async () => undefined;
    try {
      assert.equal(await vscode.commands.executeCommand('tensorviz.selectEnvironment'), true);
      assert.notEqual(api.launchToken(), previousToken, 'the environment change must restart the backend');
      assert.equal((await fetch(`${api.backendOrigin()}/health`)).status, 200);
      const deadline = Date.now() + 30_000;
      while (!api.sidebarState()?.ready && Date.now() < deadline) await sleep(100);
      assert.equal(api.sidebarState()?.ready, true, 'a healthy backend must not leave Canvas on the connection error page');
      assert.notEqual(api.sidebarState()?.sessionId, previousSession, 'Canvas must reconnect with the new launch token');
      const captureDeadline = Date.now() + 60_000;
      while (api.graphOpeningState()?.phase !== 'ready' && Date.now() < captureDeadline) await sleep(100);
      assert.notEqual(api.graphOpeningState()?.id, previousOpening, 'environment selection must recheck dependencies and reopen the retained model');
      assert.equal(api.graphOpeningState()?.target.cls, 'Net');
      assert.equal(api.graphOpeningState()?.phase, 'ready', api.graphOpeningState()?.detail);
    } finally {
      surface.showQuickPick = originalPicker;
      surface.showInformationMessage = originalInfo;
    }
  });
});
