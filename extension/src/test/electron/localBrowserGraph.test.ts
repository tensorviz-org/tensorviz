/** Explicit-browser Electron legs: Open as Graph selects the sidebar, then
 *  Open Canvas in Browser opens the EXTERNAL browser,
 *  and the session it mints carries real navigation in both directions.
 *
 * What these prove end-to-end that the unit tests cannot: the command in the
 * real extension host takes the {uri, cls} its callers hand it, posts the live
 * buffer to the real backend, mints a real local-browser session against the
 * real broker-facing backend, and hands the page URL to the OS — where "the
 * OS" is the TENSORVIZ_LOCAL_BROWSER_LINK_FILE seam, so the run never opens a
 * tab on the developer's machine.
 *
 * Models live in disposable workspace folders and name their class the way the
 * CodeLens and Models view do. Execution cases select the folder's Python through
 * the real picker; the static-opening case deliberately leaves it unselected.
 *
 * The third test is step 6's leg: a scripted view client bootstraps with the
 * fragment credential exactly as the page does and drives the live relay —
 * its revealNode lands the cursor on the node's own cached span, a stale
 * identity is refused with STALE_TARGET without moving anything, the reveal
 * echoes no focusNode back, and a cursor entering a node's span answers with
 * focusNode. The scripted view synchronizes exactly as the page does —
 * snapshotRequest on connect, then the ordered target stream — and navigates
 * only once the current projection is settled. The unconfigured navigation fixture
 * stays source-only; separate execution cases select their owning folder's Python.
 */

import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';

import * as vscode from 'vscode';
import { selectManagedEnvironment } from './workspaceFixture';

import {
  connectLocalWebSocket,
  LocalWebSocketCloseEvent,
} from '@tensorviz/graph-host/localBrowserSocket';
import {
  LOCAL_BROWSER_BOOTSTRAP_PATH,
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_WEBSOCKET_PATH,
} from '@tensorviz/contracts/protocol';

const EXTENSION_ID = 'tensorviz.tensorviz';

/** The checkout, three directories up from out/test/electron. */
const REPO = path.resolve(__dirname, '..', '..', '..', '..');
const FIXTURE = path.join(REPO, 'backend', 'tests', 'fixtures', 'ide_repo');
const LINK_FILE_ENV = 'TENSORVIZ_LOCAL_BROWSER_LINK_FILE';

interface LocalBrowserTarget {
  workspaceFolder: string;
  file: string;
  cls: string;
  documentVersion: number;
  sourceDigest: string;
  backendRevision: number;
}

interface TestApi {
  backendOrigin(): string | undefined;
  launchToken(): string | undefined;
  localBrowserSession(): { workspaceSessionId: string; pageUrl: string } | undefined;
  localBrowserTarget(): LocalBrowserTarget | undefined;
  localBrowserTracePending(): boolean;
  localBrowserViewPresence(): string | undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Poll until `read` returns something truthy, or fail with `what`. */
async function until<T>(
  what: string,
  timeoutMs: number,
  read: () => T | undefined | Promise<T | undefined>,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await read();
    if (value) {
      return value;
    }
    assert.ok(Date.now() < deadline, `timed out waiting for ${what}`);
    await sleep(100);
  }
}

/** The hints VS Code's own renderer would draw over `uri`, as {line, column, label} — the
 *  1-based line a person reads, so a failure names the line they would look at. */
async function inlayHints(
  uri: vscode.Uri,
  lineCount: number,
): Promise<Array<{ line: number; column: number; label: string }>> {
  const hints = await vscode.commands.executeCommand<vscode.InlayHint[]>(
    'vscode.executeInlayHintProvider',
    uri,
    new vscode.Range(0, 0, lineCount, 0),
  );
  return (hints ?? []).map((hint) => ({
    line: hint.position.line + 1,
    column: hint.position.character,
    label:
      typeof hint.label === 'string'
        ? hint.label
        : hint.label.map((part) => part.value).join(''),
  }));
}

/** The URLs the seam recorded, one per line, in order. */
function linkFileLines(linkFile: string): string[] {
  if (!fs.existsSync(linkFile)) {
    return [];
  }
  return fs.readFileSync(linkFile, 'utf8').split('\n').filter((line) => line !== '');
}

/** Attach to the session as the PAGE does: bootstrap with the fragment credential
 *  for the view cookie, then the broker socket with the page's own handshake. No
 *  test backdoors.
 *
 *  Needed wherever a test asserts something about a session that HAS a tab open.
 *  The link-file seam means this run never launches a browser, so without this the
 *  view roster is genuinely empty — and on an empty roster the product is right to
 *  reopen the page rather than publish a retarget into nothing. Asserting "no second
 *  tab" there asserts something that cannot be true.
 */
async function attachView(
  origin: string,
  session: { workspaceSessionId: string; pageUrl: string },
): Promise<{ close: () => void; send: (message: Record<string, unknown>) => void; messages: Array<Record<string, unknown>> }> {
  const fragment = new URLSearchParams(session.pageUrl.slice(session.pageUrl.indexOf('#') + 1));
  const credential = fragment.get('credential');
  assert.ok(credential, 'the session page URL carries no credential fragment');

  const bootstrap = await fetch(`${origin}${LOCAL_BROWSER_BOOTSTRAP_PATH}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Origin: origin },
    body: JSON.stringify({ workspaceSessionId: session.workspaceSessionId, credential }),
  });
  assert.ok(bootstrap.ok, `the broker refused the view credential: ${bootstrap.status}`);
  const setCookie = bootstrap.headers.get('set-cookie');
  assert.ok(setCookie, 'bootstrap answered without the view cookie');
  const cookie = setCookie.split(';')[0];

  const messages: Array<Record<string, unknown>> = [];
  let connected = false;
  let resolveConnected!: () => void;
  let rejectConnected!: (error: Error) => void;
  const connectedMessage = new Promise<void>((resolve, reject) => {
    resolveConnected = resolve;
    rejectConnected = reject;
  });
  const socket = connectLocalWebSocket(
    `${origin.replace(/^http/, 'ws')}${LOCAL_BROWSER_WEBSOCKET_PATH}`,
    {
      onOpen: () => {
        socket.sendText(
          JSON.stringify({
            type: 'viewHandshake',
            protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
            workspaceSessionId: session.workspaceSessionId,
            messageId: 'attach-handshake',
            role: 'view',
          }),
        );
      },
      onText: (text) => {
        const message = JSON.parse(text) as Record<string, unknown>;
        messages.push(message);
        if (!connected && message.type === 'connected') {
          connected = true;
          resolveConnected();
        }
      },
      onClose: (event: LocalWebSocketCloseEvent) => {
        if (!connected) {
          rejectConnected(
            new Error(`the broker refused the view socket: ${JSON.stringify(event)}`),
          );
        }
      },
    },
    { headers: { Origin: origin, Cookie: cookie } },
  );
  await connectedMessage;
  return { close: () => socket.close(), messages, send: message => {
    assert.ok(socket.sendText(JSON.stringify({ protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: session.workspaceSessionId, messageId: `view-${messages.length}`, ...message })));
  } };
}

/** The line index (0-based) of `needle`'s first occurrence. */
function lineOf(text: string, needle: string): number {
  const index = text.split('\n').findIndex((line) => line.includes(needle));
  assert.notEqual(index, -1, `${needle} is not in the file`);
  return index;
}

suite('localBrowserGraph', () => {
  let api: TestApi;
  let workspaceDir: string;
  let modelsDir: string;
  let residUri: vscode.Uri;
  /** The seam's sink: every URL the extension would have handed the OS. */
  let linkFile: string;

  suiteSetup(async function () {
    this.timeout(240_000);
    // Never skipped (task.md M2, "Test infrastructure rule").
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'not run: TENSORVIZ_TEST_PYTHON unset');
    process.env.BACKEND_PYTHON = python;

    // A copy of the fixture, because a later test types into resid.py without
    // saving; the checkout's copy is a fixture other suites read.
    workspaceDir = path.join(process.env.TENSORVIZ_TEST_WORKSPACE!, 'browser');
    fs.cpSync(FIXTURE, workspaceDir, { recursive: true });
    modelsDir = path.join(workspaceDir, 'models');
    assert.equal(vscode.workspace.getWorkspaceFolder(vscode.Uri.file(modelsDir))?.uri.fsPath, modelsDir);
    residUri = vscode.Uri.file(path.join(modelsDir, 'resid.py'));
    linkFile = path.join(workspaceDir, 'links.txt');
    process.env[LINK_FILE_ENV] = linkFile;

    const extension = vscode.extensions.getExtension(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} was not loaded by the extension host`);
    api = ((await extension.activate()) as { __test: TestApi }).__test;
    // The backend source and the model's owning workspace are separate.
    await vscode.workspace
      .getConfiguration('tensorviz')
      .update('repoPath', REPO, vscode.ConfigurationTarget.Global);
  });

  suiteTeardown(async () => {
    delete process.env[LINK_FILE_ENV];
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');
    fs.rmSync(workspaceDir, { recursive: true, force: true });
  });

  test('opens the named class in a browser page on the backend origin', async function () {
    this.timeout(180_000);
    const document = await vscode.workspace.openTextDocument(residUri);
    const editor = await vscode.window.showTextDocument(document);
    const cursor = lineOf(document.getText(), 'def forward'); // inside Net
    editor.selection = new vscode.Selection(cursor, 0, cursor, 0);

    await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: residUri.toString(), cls: 'Net' });
    await vscode.commands.executeCommand('tensorviz.openCanvasInBrowser');

    const session = await until('the local-browser session', 180_000, () =>
      api.localBrowserSession(),
    );
    const target = api.localBrowserTarget();
    assert.ok(target, 'the session exists but no target was anchored');

    // The class the caller named, with the containing-folder identity.
    assert.equal(target.cls, 'Net');
    assert.equal(target.file, 'resid.py');
    assert.equal(target.workspaceFolder, modelsDir);
    assert.equal(target.documentVersion, document.version);
    assert.equal(
      target.sourceDigest,
      createHash('sha256').update(document.getText(), 'utf8').digest('hex'),
    );

    // Step-5 sync: the immediate AST projection must land against the real
    // backend, so the anchored identity becomes the backend's own revision of
    // this buffer, not the prologue's "0 = not projected yet".
    await until('the AST projection to land', 60_000, () =>
      (api.localBrowserTarget()?.backendRevision ?? 0) >= 1
        ? api.localBrowserTarget()
        : undefined,
    );

    // The OS was handed exactly the session's page URL: backend origin plus a
    // fragment carrying session+credential and NOTHING ELSE — in particular not
    // the launch token.
    const links = linkFileLines(linkFile);
    assert.deepEqual(links, [session.pageUrl]);
    const origin = api.backendOrigin() as string;
    assert.ok(
      session.pageUrl.startsWith(`${origin}/#`),
      `the page URL is the backend origin plus a fragment: ${session.pageUrl}`,
    );
    const params = new URLSearchParams(session.pageUrl.slice(session.pageUrl.indexOf('#') + 1));
    assert.equal(params.get('session'), session.workspaceSessionId);
    assert.ok(params.get('credential'), 'the view credential rides the fragment');
    assert.ok(
      !session.pageUrl.includes(api.launchToken() as string),
      'the launch token must not travel in a URL',
    );
  });

  test('a class that exists only as unsaved text resolves — in place, in the same session', async function () {
    this.timeout(180_000);
    const document = await vscode.workspace.openTextDocument(residUri);
    const editor = await vscode.window.showTextDocument(document);
    const before = document.getText();
    const added = `${before}\n\nclass Unsaved(nn.Module):\n    pass\n`;
    const edit = new vscode.WorkspaceEdit();
    edit.replace(residUri, new vscode.Range(0, 0, document.lineCount, 0), added);
    assert.ok(await vscode.workspace.applyEdit(edit), 'the WorkspaceEdit was refused');
    const session = api.localBrowserSession();
    assert.ok(session, 'the first test left no session to retarget');
    const sessionId = session.workspaceSessionId;

    // A tab has to be open for "retarget in place" to be the behaviour under test.
    // The link-file seam means no browser was ever launched, so the broker's roster
    // is empty and the product correctly REOPENS the page instead — the branch that
    // exists so a user who closed every tab is not left with a command that appears
    // to do nothing. Attaching as the page does puts the session in the state this
    // test is about; without it the assertion below could never hold.
    const origin = api.backendOrigin() as string;
    const view = await attachView(origin, session);
    try {
      await until('the broker to tell the host a page is attached', 30_000, () =>
        api.localBrowserViewPresence() === 'attached' ? 'attached' : undefined,
      );

      const cursor = lineOf(added, 'class Unsaved') + 1; // on the body line
      editor.selection = new vscode.Selection(cursor, 0, cursor, 0);
      const versionBefore = document.version;

      await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: residUri.toString(), cls: 'Unsaved' });

      await until('the target to retarget to Unsaved', 60_000, () =>
        api.localBrowserTarget()?.cls === 'Unsaved' ? api.localBrowserTarget() : undefined,
      );
      const target = api.localBrowserTarget() as LocalBrowserTarget;

      // The unsaved class, resolved from the buffer the backend never saw on disk.
      assert.equal(
        target.sourceDigest,
        createHash('sha256').update(added, 'utf8').digest('hex'),
      );
      assert.equal(
        target.documentVersion,
        versionBefore,
        `the identity must name the edited document's version, not a saved one's`,
      );

      // Retargeted in place: same session, still exactly one browser tab.
      assert.equal(api.localBrowserSession()?.workspaceSessionId, sessionId);
      assert.equal(
        linkFileLines(linkFile).length,
        1,
        'a retarget must not open a second tab',
      );
    } finally {
      view.close();
    }
  });

  test('a view navigates the editor to a node; the editor answers with focus', async function () {
    this.timeout(180_000);
    // Re-anchor Net first: the previous test left this session aimed at Unsaved
    // (resid.py's buffer still carries that class unsaved, which changes nothing
    // about where Net's own statements stand).
    const document = await vscode.workspace.openTextDocument(residUri);
    const editor = await vscode.window.showTextDocument(document);
    const cursor = lineOf(document.getText(), 'def forward'); // inside Net
    editor.selection = new vscode.Selection(cursor, 0, cursor, 0);

    await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: residUri.toString(), cls: 'Net' });

    // Ordering guard only: the identity the navigation legs name comes from
    // the view's own stream below, not from this host-side read — the trace
    // scheduled here is exactly what re-projects Net and rolls its revision.
    await until('the session to re-anchor Net, projected', 120_000, () => {
      const target = api.localBrowserTarget();
      return target?.cls === 'Net' &&
        target.documentVersion === document.version &&
        target.backendRevision >= 1
        ? target
        : undefined;
    });

    // Attach as a real view: the page's own fragment bootstrap, then the broker
    // socket with the page's own handshake — no test backdoors.
    const session = api.localBrowserSession();
    assert.ok(session, 'no local-browser session to attach a view to');
    const origin = api.backendOrigin() as string;
    const fragment = new URLSearchParams(session.pageUrl.slice(session.pageUrl.indexOf('#') + 1));
    const credential = fragment.get('credential');
    assert.ok(credential, 'the session page URL carries no credential fragment');

    const bootstrap = await fetch(`${origin}${LOCAL_BROWSER_BOOTSTRAP_PATH}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Origin: origin },
      body: JSON.stringify({ workspaceSessionId: session.workspaceSessionId, credential }),
    });
    assert.ok(bootstrap.ok, `the broker refused the view credential: ${bootstrap.status}`);
    const setCookie = bootstrap.headers.get('set-cookie');
    assert.ok(setCookie, 'bootstrap answered without the view cookie');
    const cookie = setCookie.split(';')[0];

    const received: Array<Record<string, unknown>> = [];
    let closed: LocalWebSocketCloseEvent | null = null;
    let connected = false;
    let resolveConnected!: (value: Record<string, unknown>) => void;
    let rejectConnected!: (error: Error) => void;
    const connectedMessage = new Promise<Record<string, unknown>>((resolve, reject) => {
      resolveConnected = resolve;
      rejectConnected = reject;
    });
    const socket = connectLocalWebSocket(
      `${origin.replace(/^http/, 'ws')}${LOCAL_BROWSER_WEBSOCKET_PATH}`,
      {
        onOpen: () => {
          socket.sendText(
            JSON.stringify({
              type: 'viewHandshake',
              protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
              workspaceSessionId: session.workspaceSessionId,
              messageId: 'nav-handshake',
              role: 'view',
            }),
          );
        },
        onText: (text) => {
          const message = JSON.parse(text) as Record<string, unknown>;
          received.push(message);
          if (!connected && message.type === 'connected') {
            connected = true;
            resolveConnected(message);
          }
        },
        onClose: (event) => {
          closed = event;
          if (!connected) {
            rejectConnected(
              new Error(`the broker refused the view socket: ${JSON.stringify(event)}`),
            );
          }
        },
      },
      { headers: { Origin: origin, Cookie: cookie } },
    );

    try {
      await connectedMessage;

      const sendReveal = (messageId: string, target: LocalBrowserTarget, nodeId: string): void => {
        assert.ok(
          socket.sendText(
            JSON.stringify({
              type: 'revealNode',
              protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
              workspaceSessionId: session.workspaceSessionId,
              messageId,
              target,
              nodeId,
            }),
          ),
          'the view socket refused the reveal frame',
        );
      };
      const stillOpen = (): void => {
        assert.ok(!closed, `the broker dropped the view mid-test: ${JSON.stringify(closed)}`);
      };

      // Synchronize as the page itself does (localBrowserViewTransport sends
      // snapshotRequest on (re)connect): ask for one snapshot, so an identity
      // that settled before this socket attached still reaches this view.
      assert.ok(
        socket.sendText(
          JSON.stringify({
            type: 'snapshotRequest',
            protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
            workspaceSessionId: session.workspaceSessionId,
            messageId: 'nav-snapshot',
            role: 'view',
            lastHostSequence: 0,
          }),
        ),
        'the view socket refused the snapshot request',
      );

      /** The newest identity the page displays, read from its own ordered
       *  stream: the target the newest syncState names, or the snapshot's
       *  activeTarget when this view's snapshotRequest is the latest thing
       *  answered. Every label — runtime-traced, source-only, stale — names
       *  the identity the page shows for it. */
      const displayedTarget = (): LocalBrowserTarget | undefined => {
        for (let i = received.length - 1; i >= 0; i -= 1) {
          const message = received[i];
          if (message.type === 'syncState') {
            return message.target as LocalBrowserTarget;
          }
          if (message.type === 'snapshot') {
            const state = message.state as {
              activeTarget?: LocalBrowserTarget | null;
            };
            return state.activeTarget ?? undefined;
          }
        }
        return undefined;
      };

      // Settle before navigating: the save-coalesced trace scheduled at
      // re-anchor re-projects Net and advances backendRevision when it lands,
      // so an identity named while that trace is armed or running races the
      // later revision. Once the host reports no trace pending, the identity
      // is final — and the page displays exactly it once the ordered relay
      // catches up: runtime-traced when the model ran, honestly source-only
      // when this environment cannot trace. Both are terminal states here.
      // The host's own fields only gate CONVERGENCE; the identity the
      // navigation legs name is read from the view's ordered stream.
      const live = await until('the page-visible Net to settle trace-decided', 150_000, () => {
        stillOpen();
        const host = api.localBrowserTarget();
        const displayed = displayedTarget();
        return !api.localBrowserTracePending() &&
          host?.cls === 'Net' &&
          host.documentVersion === document.version &&
          displayed !== undefined &&
          displayed.workspaceFolder === host.workspaceFolder &&
          displayed.file === host.file &&
          displayed.cls === host.cls &&
          displayed.documentVersion === host.documentVersion &&
          displayed.sourceDigest === host.sourceDigest &&
          displayed.backendRevision === host.backendRevision
          ? displayed
          : undefined;
      });

      // node → line: the page names a node plus the exact identity it shows;
      // the host resolves its own cached span and moves the cursor there.
      sendReveal('nav-reveal-l2', live, 'l2');
      const ack = await until('the ack for the reveal', 30_000, () => {
        stillOpen();
        return received.find((m) => m.type === 'ack' && m.ackId === 'nav-reveal-l2');
      });
      assert.equal(ack.ok, true, `the reveal was refused: ${JSON.stringify(ack)}`);
      assert.deepEqual(ack.target, live);

      const l2Line = lineOf(document.getText(), 'self.l2');
      await until('the cursor on self.l2', 30_000, () =>
        vscode.window.activeTextEditor?.document.uri.toString() === residUri.toString() &&
        vscode.window.activeTextEditor.selection.active.line === l2Line
          ? true
          : undefined,
      );

      // A page behind the live state is refused, and nothing moves: revision 0
      // is the prologue's "not projected yet", never this graph's identity.
      sendReveal('nav-reveal-stale', { ...live, backendRevision: 0 }, 'l1');
      const refusal = await until('the stale reveal to be refused', 30_000, () => {
        stillOpen();
        return received.find((m) => m.type === 'ack' && m.ackId === 'nav-reveal-stale');
      });
      assert.equal(refusal.ok, false, `a stale identity was honored: ${JSON.stringify(refusal)}`);
      assert.equal(refusal.errorCode, 'STALE_TARGET');
      const afterRefusal = vscode.window.activeTextEditor;
      assert.ok(afterRefusal, 'no active editor after the refused reveal');
      assert.equal(
        afterRefusal.selection.active.line,
        l2Line,
        'a refused reveal moved the cursor',
      );

      // No echo: the cursor move the reveal itself caused must not bounce a
      // focusNode back at the page that asked.
      assert.ok(
        !received.some((m) => m.type === 'focusNode'),
        `the reveal echoed a focusNode back: ${JSON.stringify(
          received.filter((m) => m.type === 'focusNode'),
        )}`,
      );

      // line → node: entering a node's span focuses it on the page.
      const l1Line = lineOf(document.getText(), 'self.l1');
      afterRefusal.selection = new vscode.Selection(l1Line, 0, l1Line, 0);
      const focus = await until("the editor's focusNode for l1", 30_000, () => {
        stillOpen();
        return received.find((m) => m.type === 'focusNode' && (m.nodeIds as string[] | undefined)?.includes('l1'));
      });
      assert.deepEqual(focus.nodeIds, ['l1'],
        'only the layer owns this line; review-only declaration keys must not reach live focus');
      assert.deepEqual(focus.target, live);
    } finally {
      socket.close();
    }
  });


  test('opening before a Python environment is chosen never imports executable source or arms a trace, even when trace-on-save is enabled', async function () {
    this.timeout(60_000);
    const marker = path.join(workspaceDir, 'unexpected-model-execution');
    const uri = vscode.Uri.file(path.join(modelsDir, 'static_open.py'));
    fs.writeFileSync(uri.fsPath, [
      'from pathlib import Path',
      `Path(${JSON.stringify(marker)}).write_text("executed")`,
      'from torch import nn',
      'class StaticOpen(nn.Module):',
      '    def __init__(self):',
      '        super().__init__()',
      '        self.fc = nn.Linear(3, 2)',
      '    def forward(self, x):',
      '        return self.fc(x)',
      '',
    ].join('\n'));
    const config = vscode.workspace.getConfiguration('tensorviz');
    const previous = config.inspect<boolean>('traceOnSave')?.globalValue;
    try {
      await config.update('traceOnSave', true, vscode.ConfigurationTarget.Global);
      const doc = await vscode.workspace.openTextDocument(uri);
      const editor = await vscode.window.showTextDocument(doc);
      editor.selection = new vscode.Selection(6, 0, 6, 0);
      await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: uri.toString(), cls: 'StaticOpen' });
      await until('static executable model projection', 30_000, () => {
        const target = api.localBrowserTarget();
        return target?.cls === 'StaticOpen' && target.backendRevision > 0 ? target : undefined;
      });
      assert.equal(api.localBrowserTracePending(), false, 'opening must not even schedule model execution');
      await sleep(3000); // longer than the real 1.5-second save-coalescing window
      assert.equal(api.localBrowserTracePending(), false);
      assert.equal(fs.existsSync(marker), false, 'static graph opening imported workspace code');
    } finally {
      await config.update('traceOnSave', previous, vscode.ConfigurationTarget.Global);
    }
  });

  test('tensor slices, addition and methods show final expression shapes and refresh after saving', async function () {
    this.timeout(300_000);
    const project = path.join(workspaceDir, 'tensor-expression-project');
    assert.equal(vscode.workspace.getWorkspaceFolder(vscode.Uri.file(project))?.uri.fsPath, project);
    const uri = vscode.Uri.file(path.join(project, 'expressions.py'));
    fs.writeFileSync(uri.fsPath, [
      'import torch',
      'import torch.nn as nn',
      '',
      'class TensorExpressions(nn.Module):',
      '    def __init__(self):',
      '        super().__init__()',
      '        self.embed = nn.Embedding(16, 32)',
      '        self.pos = nn.Parameter(torch.zeros(1, 12, 32))',
      '        self.head = nn.Linear(32, 16)',
      '',
      '    def forward(self, tokens):',
      '        tokens = tokens[:, :8]',
      '        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]',
      '        return self.head(x.transpose(1, 2).transpose(1, 2))',
      '',
    ].join('\n'));
    fs.writeFileSync(path.join(project, 'tensorviz.toml'), `version = 1
[profiles.expressions]
kind = "model"
model = "expressions:TensorExpressions"
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 0
init = { args = [], kwargs = {} }
inputs = { args = [{ kind = "tensor", shape = [1, 8], dtype = "int64", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }
`);
    const config = vscode.workspace.getConfiguration('tensorviz');
    const previous = config.inspect<boolean>('traceOnSave')?.globalValue;
    try {
      await config.update('traceOnSave', true, vscode.ConfigurationTarget.Global);
      const document = await vscode.workspace.openTextDocument(uri);
      const editor = await vscode.window.showTextDocument(document);
      await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: uri.toString(), cls: 'TensorExpressions' });
      await selectManagedEnvironment();
      const expected = (length: number) => [
        { line: 12, column: 30, label: ` → 1x${length}` },
        { line: 13, column: 30, label: ` → 1x${length}x32` },
        { line: 13, column: 63, label: ` → 1x${length}x32` },
        { line: 14, column: 42, label: ` → 1x32x${length}` },
        { line: 14, column: 58, label: ` → 1x${length}x32` },
        { line: 14, column: 59, label: ` → 1x${length}x16` },
      ];
      const initial = await until('the tensor-expression capture', 60_000, async () => {
        const hints = await inlayHints(uri, document.lineCount);
        return hints.length ? hints : undefined;
      });
      // Exact equality excludes the full positional tensor, intermediate slices,
      // constructor declarations and the Linear/Embedding ATen internals.
      assert.deepEqual(initial, expected(8));
      await until('the expression capture to finish', 30_000, () => !api.localBrowserTracePending());
      await editor.edit(builder => builder.replace(new vscode.Range(0, 0, document.lineCount, 0),
        document.getText().replace('tokens[:, :8]', 'tokens[:, :4]')));
      assert.ok(await document.save());
      const saved = await until('updated slice and method shapes', 60_000, async () => {
        const hints = await inlayHints(uri, document.lineCount);
        return hints.some(hint => hint.line === 13 && hint.column === 63 && hint.label === ' → 1x4x32') ? hints : undefined;
      });
      assert.deepEqual(saved, expected(4));
    } finally {
      await config.update('traceOnSave', previous, vscode.ConfigurationTarget.Global);
    }
  });

  /** The measured shape, in the editor, reached the way a user reaches it.
   *
   * Only this level can prove it. The provider is registered by `activate`, fed by
   * whichever surface holds the projection, and asked by VS Code's own renderer — so a
   * unit test of any one of those three can be green while the editor shows nothing,
   * which is exactly what shipped: `ShapeInlayHints` was built inside `openCanvas`, and
   * "Open as Graph" draws the browser page and never goes near it. Hints were dead for
   * every user who used the documented entry point. */
  test('a measured save shows each observed shape beside its nested forward call', async function () {
    this.timeout(300_000);
    // Other cases deliberately leave dirty buffers; saved-profile validation
    // checks the whole project, so this measurement owns a separate project.
    const measuredDir = path.join(workspaceDir, 'measured-project');
    assert.equal(vscode.workspace.getWorkspaceFolder(vscode.Uri.file(measuredDir))?.uri.fsPath, measuredDir);
    const uri = vscode.Uri.file(path.join(measuredDir, 'measured.py'));
    fs.writeFileSync(uri.fsPath, [
      'import torch.nn as nn',
      '',
      '',
      'class Measured(nn.Module):',
      '    def __init__(self, width=8):',
      '        super().__init__()',
      '        self.up = nn.Linear(width, width * 2)',
      '        self.down = nn.Linear(width * 2, width)',
      '',
      '    def forward(self, x):',
      '        return self.down(self.up(x))',
      '',
      '',
      '',
    ].join('\n'));
    fs.writeFileSync(path.join(measuredDir, 'tensorviz.toml'), `version = 1
[profiles.measured]
kind = "model"
model = "measured:Measured"
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 0
init = { args = [], kwargs = {} }
inputs = { args = [{ kind = "tensor", shape = [2, 8], dtype = "float32", requires_grad = false, generator = { kind = "normal", mean = 0, std = 1, seed = 0 } }], kwargs = {} }
`);
    const config = vscode.workspace.getConfiguration('tensorviz');
    const previous = config.inspect<boolean>('traceOnSave')?.globalValue;
    try {
      await config.update('traceOnSave', true, vscode.ConfigurationTarget.Global);
      const document = await vscode.workspace.openTextDocument(uri);
      const editor = await vscode.window.showTextDocument(document);
      const cursor = lineOf(document.getText(), 'def forward');
      editor.selection = new vscode.Selection(cursor, 0, cursor, 0);
      await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: uri.toString(), cls: 'Measured' });
      await selectManagedEnvironment();
      await until('the source projection of Measured', 60_000, () => {
        const target = api.localBrowserTarget();
        return target?.cls === 'Measured' && target.backendRevision > 0 ? target : undefined;
      });

      // Open as Graph captures the saved profile. Establish its measured baseline
      // before editing, so stale hints cannot satisfy the save assertion below.
      const initial = await until('the initial capture shapes', 60_000, async () => {
        const hints = await inlayHints(uri, document.lineCount);
        return hints.length > 0 ? hints : undefined;
      });
      assert.deepEqual(initial, [
        { line: 11, column: 35, label: ' \u2192 2x16' },
        { line: 11, column: 36, label: ' \u2192 2x8' },
      ]);
      await until('the initial capture to finish', 30_000, () => !api.localBrowserTracePending());
      const view = await attachView(api.backendOrigin()!, api.localBrowserSession()!);
      try {
        view.send({ type: 'validateRequest', requestId: 'run-model', target: api.localBrowserTarget() });
        const review = await until('the manual execution review', 30_000, () => {
          const state = view.messages.find(message => message.type === 'graphOpening' &&
            (message.state as { execution?: { phase: string } })?.execution?.phase === 'review')?.state as
            { id: string; target: LocalBrowserTarget; execution: { id: string; device: string } } | undefined;
          return state;
        });
        assert.ok(!view.messages.some(message => message.type === 'validateResult' && message.requestId === 'run-model'),
          'manual execution waits for Run in the review');
        view.send({ type: 'graphSetupRequest', requestId: 'approve-run-model', openingId: review.id, target: review.target,
          action: { kind: 'execution', action: 'run', reviewId: review.execution.id, device: review.execution.device } });
        const receipt = await until('the Run model result', 120_000, () =>
          view.messages.find(message => message.type === 'validateResult' && message.requestId === 'run-model'));
        assert.equal(receipt.status, 'passed', JSON.stringify(receipt));
        assert.equal('validationPassed' in receipt, false);
        const hints = await inlayHints(uri, document.lineCount);
        assert.deepEqual(hints.map(hint => hint.label), [' \u2192 2x16', ' \u2192 2x8'], 'a run shows shapes, never timing');
      } finally { view.close(); }
      await editor.edit((builder) => builder.replace(
        new vscode.Range(0, 0, document.lineCount, 0),
        document.getText().replaceAll('width * 2', 'width * 3'),
      ));
      assert.ok(await document.save(), 'the buffer did not save');

      const measured = await until('the saved model shapes to reach the editor', 60_000, async () => {
        const hints = await inlayHints(uri, document.lineCount);
        return hints.some(hint => hint.line === 11 && hint.column === 35 && hint.label === ' \u2192 2x24') ? hints : undefined;
      });

      // The inner call widens 8 to 24; the outer call still brings it back.
      // Exact results also exclude constructor hints and internal torch operations.
      assert.deepEqual(measured, [
        { line: 11, column: 35, label: ' \u2192 2x24' },
        { line: 11, column: 36, label: ' \u2192 2x8' },
      ]);
    } finally {
      await config.update('traceOnSave', previous, vscode.ConfigurationTarget.Global);
    }
  });
});
