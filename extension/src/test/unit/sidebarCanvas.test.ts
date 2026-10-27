import * as assert from 'node:assert/strict';
import type * as vscode from 'vscode';
import { SidebarCanvas, SIDEBAR_CANVAS_VIEW, CANVAS_REQUESTED, CANVAS_VISIBLE } from '../../sidebarCanvas';
import { LOCAL_BROWSER_PROTOCOL_VERSION, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { resetWebviews, StubWebviewView, webviewState } from '../vscodeStub';
import { ShellTarget } from '../../shell';
import { LocalBrowserValidation } from '../../localBrowserValidation';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';

const target: ShellTarget = { appUrl: 'http://127.0.0.1:9000/?token=secret', appOrigin: 'http://127.0.0.1:9000', port: 9000, workspaceDir: '/ws' };
const model: LocalBrowserTargetIdentity = { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 2 };
const ready = (view: StubWebviewView) => {
  view.receive({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION });
  const post = view.webview.postMessage;
  view.webview.postMessage = async (message: any) => {
    const delivered = await post(message);
    if (message.type === 'canvasCheckpoint') view.receive({ type: 'canvasPresentation', requestId: message.requestId });
    return delivered;
  };
};
const publications = (view: StubWebviewView): any[] => view.messages.filter((m: any) => m.type === 'canvasMessage').map((m: any) => m.message);
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const homeHtml = (view: StubWebviewView): string => (view.messages.filter((m: any) => m.type === 'canvasHomeState').at(-1) as any)?.html ?? view.webview.html;

suite('SidebarCanvas lifecycle', () => {
  let canvas: SidebarCanvas;
  let registration: vscode.Disposable;
  let prepares: number;
  let replays: number;
  let received: unknown[];
  let connections: Array<{ viewId: string; connected: boolean }>;
  const create = (prepare = async () => target, refreshTarget = false) => {
    canvas = new SidebarCanvas({ prepare: async () => { prepares++; return prepare(); },
      refreshTarget,
      receive: message => received.push(message),
      viewConnection: (viewId, connected) => connections.push({ viewId, connected }),
      replay: () => { replays++; canvas.post({ type: 'snapshot', state: { activeTarget: model,
        projection: { evidence: 'measured' }, traced: true, greyed: false, syncState: 'runtime-traced', connectionState: 'connected' } }); },
      channel: { appendLine() {} } as unknown as vscode.OutputChannel });
    registration = canvas.register();
  };
  setup(() => { resetWebviews(); prepares = 0; replays = 0; received = []; connections = []; });
  teardown(() => { canvas?.dispose(); registration?.dispose(); resetWebviews(); });

  test('native first resolution and ordinary reveals never provision a runtime', async () => {
    create();
    const view = new StubWebviewView();
    webviewState.views.push(view);
    canvas.resolveWebviewView(view as unknown as vscode.WebviewView);
    assert.equal(prepares, 0);
    assert.match(view.webview.html, /Explore your model/);
    assert.match(view.webview.html, /Open Folder/);
    assert.match(view.webview.html, /empty folder to try a starter model/);
    assert.doesNotMatch(view.webview.html, /data-action="starter"/);
    assert.doesNotMatch(view.webview.html, /iframe|http:/);
    assert.equal(canvas.connectedViewId, undefined);
    await canvas.reveal();
    await canvas.hide();
    await canvas.reveal();
    assert.equal(prepares, 0, 'reopening the starting screen is not install consent');
    assert.equal(canvas.visible, true);
    assert.equal(received.length, 0);
  });

  test('a model opening can finish preparation before VS Code resolves the first sidebar view', async () => {
    create();
    // Real view resolution is asynchronous; command completion does not guarantee a mounted view.
    webviewState.handlers.set(`${SIDEBAR_CANVAS_VIEW}.focus`, () => undefined);
    assert.equal(await canvas.show(), true);
    assert.equal(webviewState.views.length, 0);
    const view = new StubWebviewView(); webviewState.views.push(view);
    canvas.resolveWebviewView(view as unknown as vscode.WebviewView);
    assert.match(view.webview.html, /iframe/);
    assert.match(view.webview.html, /Explore your model/);
    assert.equal(prepares, 1);
  });

  test('starting screen uses cached discovery, routes host-owned models and cannot replace a graph', async () => {
    let home = { hasWorkspace: true, indexed: false, models: [] as Array<{ name: string; file: string; uri: string }> };
    canvas = new SidebarCanvas({
      prepare: async () => { prepares++; return target; }, home: () => home,
      receive() {}, replay() {}, channel: { appendLine() {} } as unknown as vscode.OutputChannel,
    });
    registration = canvas.register();
    const opens: unknown[] = [];
    webviewState.handlers.set('tensorviz.openAsGraph', (...args) => opens.push(args));
    await canvas.reveal();
    const view = webviewState.views[0];
    assert.match(view.webview.html, /Discovered models will appear/);
    assert.doesNotMatch(view.webview.html, /No PyTorch models were found/);
    home = { hasWorkspace: true, indexed: true, models: [{ name: 'Net<script>', file: 'models/<net>.py', uri: 'file:///ws/net.py' }] };
    canvas.refreshHome();
    assert.match(homeHtml(view), /Net&lt;script&gt;/);
    assert.match(homeHtml(view), /models\/&lt;net&gt;.py/);
    assert.doesNotMatch(homeHtml(view), /data-action="starter"/);
    assert.equal(prepares, 0);
    view.receive({ type: 'homeModel', index: -1 });
    view.receive({ type: 'homeModel', index: 99, uri: 'file:///foreign.py' });
    view.receive({ type: 'homeAction', action: 'arbitrary.command' });
    assert.deepEqual(opens, []);
    view.receive({ type: 'homeModel', index: 0, uri: 'file:///foreign.py' });
    view.receive({ type: 'homeAction', action: 'open' });
    assert.deepEqual(opens, [[{ uri: 'file:///ws/net.py', cls: 'Net<script>' }]], 'pending requests cannot open a second model');
    await tick();
    view.receive({ type: 'homeAction', action: 'open' });
    await tick();
    await canvas.show();
    ready(view);
    canvas.post({ type: 'activeTarget', target: model });
    const html = view.webview.html;
    canvas.refreshHome();
    assert.equal(view.webview.html, html, 'discovery updates cannot retarget or unmount a graph');
    view.receive({ type: 'homeModel', index: 0 });
    assert.equal(opens.length, 2, 'retired home messages cannot open models');
  });

  test('an empty workspace offers the existing starter picker without preparing or opening a graph', async () => {
    let pickers = 0, opens = 0, finish!: () => void;
    canvas = new SidebarCanvas({ prepare: async () => { prepares++; return target; },
      home: () => ({ hasWorkspace: true, indexed: false, models: [] }),
      receive() {}, replay() {}, channel: { appendLine() {} } as never });
    registration = canvas.register();
    webviewState.handlers.set('tensorviz.newModelFromTemplate', () => {
      pickers++;
      return new Promise<void>(resolve => { finish = resolve; });
    });
    webviewState.handlers.set('tensorviz.openAsGraph', () => { opens++; });
    await canvas.reveal(); const view = webviewState.views[0];
    assert.match(homeHtml(view), /data-action="starter"/);
    assert.equal(pickers, 0, 'showing the home must not create a file or select a template');
    view.receive({ type: 'homeAction', action: 'starter' });
    view.receive({ type: 'homeAction', action: 'starter' });
    assert.equal(pickers, 1, 'pending selection cannot create duplicate pickers');
    assert.match(homeHtml(view), /data-action="starter" disabled/);
    assert.equal(prepares, 0, 'starter creation uses the local template command, without acquiring a runtime');
    finish(); await tick();
    assert.doesNotMatch(homeHtml(view), /Choosing a starter model|Continue opening|Retry opening/);
    assert.equal(opens, 0, 'creating or cancelling a starter must not execute a model');
    assert.equal(prepares, 0);
  });

  test('home checks are automatic, replace obsolete results and never prepare the service', async () => {
    const checks: Array<{ signal: AbortSignal; finish: (value: any) => void }> = [];
    canvas = new SidebarCanvas({ prepare: async () => { prepares++; return target; },
      home: () => ({ hasWorkspace: true, indexed: false, models: [] }),
      checkHome: signal => new Promise(finish => checks.push({ signal, finish })),
      receive() {}, replay() {}, channel: { appendLine() {} } as never });
    registration = canvas.register(); await canvas.reveal();
    const view = webviewState.views[0], writes = view.htmlWrites;
    assert.equal(checks.length, 1); assert.match(homeHtml(view), /Checking tensorViz runtime/);
    canvas.refreshHome(true);
    assert.equal(checks[0].signal.aborted, true);
    checks[1].finish({ phase: 'ready', detail: 'Current folder ready', runtimeReady: true }); await tick();
    checks[0].finish({ phase: 'unavailable', detail: 'Obsolete folder' }); await tick();
    assert.match(homeHtml(view), /Current folder ready/); assert.doesNotMatch(homeHtml(view), /Obsolete folder/);
    const messages = view.messages.length;
    view.receive({ type: 'homeReady' });
    assert.match(homeHtml(view), /Current folder ready/);
    assert.ok(view.messages.length > messages, 'late-loading webviews receive the latest completed check');
    assert.equal(checks.length, 2, 'ready handshake replays metadata rather than rerunning checks');
    assert.equal(prepares, 0); assert.equal(view.htmlWrites, writes, 'readiness updates keep one starting document');
    canvas.refreshHome(true); canvas.dispose();
    assert.equal(checks[2].signal.aborted, true);
    checks[2].finish({ phase: 'ready', detail: 'Disposed' }); await tick();
    assert.doesNotMatch(homeHtml(view), /Disposed/);
  });

  test('the connected empty renderer stays hidden until a selected model is published', async () => {
    canvas = new SidebarCanvas({ prepare: async () => target,
      home: () => ({ hasWorkspace: true, indexed: false, models: [] }), receive() {},
      replay: () => canvas.post({ type: 'snapshot', state: { activeTarget: null, projection: null,
        traced: false, greyed: false, syncState: null, connectionState: 'connected' } }), channel: { appendLine() {} } as never });
    registration = canvas.register(); await canvas.show(); const view = webviewState.views[0]; ready(view);
    assert.equal(canvas.inspectForTest().homeVisible, true);
    assert.match(view.webview.html, /hidden aria-hidden="true"/);
    assert.equal((view.messages.filter((m: any) => m.type === 'canvasContent').at(-1) as any).visible, false);
    canvas.post({ type: 'activeTarget', target: model });
    assert.equal(canvas.inspectForTest().homeVisible, false);
    assert.equal((view.messages.filter((m: any) => m.type === 'canvasContent').at(-1) as any).visible, true);
    canvas.post({ type: 'snapshot', state: { activeTarget: null, projection: null, traced: false, greyed: false,
      syncState: null, connectionState: 'connected' } });
    assert.equal(canvas.inspectForTest().homeVisible, true, 'clearing the selection returns to the same home');
  });

  test('runtime setup does not replace the retained model request; retry uses its host-owned identity', async () => {
    let selected = { name: 'Net', file: 'net.py', uri: 'file:///ws/net.py' };
    let attempts = 0;
    const opens: unknown[] = [];
    canvas = new SidebarCanvas({ prepare: async () => target,
      home: () => ({ hasWorkspace: true, indexed: true, models: [selected] }),
      receive() {}, replay() {}, channel: { appendLine() {}, show() {} } as never });
    registration = canvas.register(); await canvas.reveal(); const view = webviewState.views[0];
    webviewState.handlers.set('tensorviz.openAsGraph', (...args) => {
      opens.push(args); if (++attempts === 1) throw new Error('private diagnostic');
    });
    view.receive({ type: 'homeModel', index: 0 }); await tick();
    assert.match(homeHtml(view), /Retry opening/); assert.doesNotMatch(homeHtml(view), /private diagnostic/);
    selected = { name: 'Other', file: 'other.py', uri: 'file:///ws/other.py' }; canvas.refreshHome();
    view.receive({ type: 'homeAction', action: 'python' }); await tick();
    assert.equal(opens.length, 1, 'retired home Python actions cannot select or open another environment');
    view.receive({ type: 'homeAction', action: 'setup' }); await tick();
    assert.match(homeHtml(view), /Continue opening/, 'successful runtime setup keeps a visible way to resume the original opening');
    view.receive({ type: 'homeAction', action: 'retry' }); await tick();
    assert.deepEqual(opens, [[{ uri: 'file:///ws/net.py', cls: 'Net' }], [{ uri: 'file:///ws/net.py', cls: 'Net' }]]);
    assert.doesNotMatch(homeHtml(view), /Continue opening/, 'a completed request is no longer a retained failure');
  });

  test('Open PR is available while tensorViz runtime checks are still running', async () => {
    let prs = 0, finish!: (value: any) => void;
    canvas = new SidebarCanvas({ prepare: async () => target,
      home: () => ({ hasWorkspace: true, indexed: false, models: [] }),
      checkHome: () => new Promise(resolve => { finish = resolve; }), receive() {}, replay() {}, channel: { appendLine() {} } as never });
    registration = canvas.register(); webviewState.handlers.set('tensorviz.openPullRequest', () => { prs++; });
    await canvas.reveal(); const view = webviewState.views[0];
    view.receive({ type: 'homeAction', action: 'pr' }); await tick();
    assert.equal(prs, 1); assert.equal(prepares, 0);
    finish({ phase: 'needs_setup', detail: 'tensorViz runtime needs setup' }); await tick();
  });

  test('SSH show re-resolves forwarding and remounts only when its origin changes', async () => {
    let current = { ...target, forwarded: true };
    create(async () => current, true);
    await canvas.show(); const view = webviewState.views[0]; ready(view);
    const writes = view.htmlWrites;
    await canvas.show();
    assert.equal(view.htmlWrites, writes, 'unchanged forwarding preserves the live iframe');
    current = { ...current, appOrigin: 'http://127.0.0.1:53128', appUrl: 'http://127.0.0.1:53128/?token=fixture' };
    const showing = canvas.show(); await tick();
    const request = view.messages.find((m: any) => m.type === 'canvasCheckpoint') as any;
    const presentation = { camera: { x: 99, y: 10, z: 0.6 } };
    view.receive({ type: 'canvasPresentation', requestId: request.requestId, state: presentation });
    await showing;
    assert.match(view.webview.html, /frame-src http:\/\/127\.0\.0\.1:53128;/);
    assert.deepEqual(view.webview.options.portMapping, []);
    assert.deepEqual(canvas.inspectForTest().presentation, presentation);
    ready(view);
    assert.equal(canvas.inspectForTest().ready, true);
  });

  test('registration is lazy; concurrent reveal starts once and never selects or executes a model', async () => {
    let release!: (target: ShellTarget) => void;
    create(() => new Promise(resolve => { release = resolve; }));
    assert.equal(prepares, 0);
    assert.deepEqual(webviewState.registrations, [{ id: SIDEBAR_CANVAS_VIEW, options: { webviewOptions: { retainContextWhenHidden: true } } }]);
    const shows = Promise.all([canvas.show(), canvas.show(), canvas.show()]);
    await tick();
    assert.equal(prepares, 1);
    assert.equal(webviewState.views[0].visible, true, 'the first click reveals Canvas before service preparation completes');
    assert.match(homeHtml(webviewState.views[0]), /Preparing tensorViz/);
    assert.match(webviewState.views[0].webview.html, /Explore your model/);
    release(target); await shows;
    assert.equal(webviewState.views.length, 1);
    assert.equal(received.length, 0, 'visibility is not a graph/validation request');
    const view = webviewState.views[0];
    assert.match(view.webview.html, /view=canvas/);
    const writes = view.htmlWrites;
    ready(view);
    await canvas.show();
    assert.equal(view.htmlWrites, writes, 'reveal must not reload the renderer');
    assert.equal(replays, 1, 'unchanged reveal does not republish/reset presentation');
  });

  test('a view attached after preparation starts still displays first-click progress', async () => {
    let release!: (target: ShellTarget) => void;
    create(() => new Promise(resolve => { release = resolve; }));
    webviewState.handlers.set(`${SIDEBAR_CANVAS_VIEW}.focus`, () => undefined);
    const showing = canvas.show();
    await tick();
    assert.equal(prepares, 1);
    assert.equal(webviewState.views.length, 0, 'VS Code can resolve the view after its focus command returns');
    const view = new StubWebviewView();
    webviewState.views.push(view);
    canvas.resolveWebviewView(view as unknown as vscode.WebviewView);
    assert.match(homeHtml(view), /Preparing tensorViz/);
    assert.ok(!view.webview.html.includes('again to restart'), 'in-flight preparation does not ask for a retry');
    assert.equal(prepares, 1, 'late view attachment shares the running preparation');
    release(target);
    assert.equal(await showing, true);
    assert.match(view.webview.html, /view=canvas/);
  });

  test('hide preserves the host and replays validation completed while hidden', async () => {
    create(); await canvas.show(); const view = webviewState.views[0]; ready(view);
    const writes = view.htmlWrites;
    await canvas.hide();
    assert.equal(canvas.requestedVisible, false);
    assert.equal(canvas.visible, false, 'native view toggle reports hidden without removing the contribution');
    view.setVisible(false);
    canvas.post({ type: 'validateResult', target: model, requestId: 'run', status: 'passed', detail: 'Measured tensors passed.' });
    await tick();
    assert.equal(webviewState.contexts.get(CANVAS_VISIBLE), false);
    await canvas.show();
    assert.equal(prepares, 1);
    assert.equal(view.htmlWrites, writes);
    assert.equal(publications(view).at(-1).status, 'passed');
    assert.equal(publications(view).filter(m => m.type === 'snapshot').at(-1).state.projection.evidence, 'measured');
    assert.ok(webviewState.commands.every(c => ['setContext', `${SIDEBAR_CANVAS_VIEW}.focus`, `${SIDEBAR_CANVAS_VIEW}.removeView`].includes(c.id)),
      'uses only the native view actions; never resets or moves another extension');
  });

  test('actual view disposal keeps model and results for a fresh bridge with no second startup', async () => {
    create(); await canvas.show(); const first = webviewState.views[0]; ready(first);
    await canvas.hide(); first.dispose();
    canvas.post({ type: 'validateResult', target: model, requestId: 'run', status: 'failed', detail: 'Shape mismatch.' });
    await canvas.show(); const next = webviewState.views[1]; ready(next);
    assert.equal(prepares, 1);
    assert.equal(publications(next)[0].state.activeTarget.cls, 'Net');
    assert.equal(publications(next).at(-1).status, 'failed');
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), true);
    assert.equal(webviewState.contexts.get(CANVAS_VISIBLE), true);
    first.receive({ type: 'canvasMessage', message: { type: 'validateCancel' } });
    assert.equal(received.length, 0, 'old view subscriptions were removed');
  });

  test('the same validation controller finishes after view disposal; state and presentation are replayed', async () => {
    create(); await canvas.show(); const first = webviewState.views[0]; ready(first);
    const presentation = { graphId: 'Net', camera: { x: 23, y: 42, z: 0.6 }, selected: ['shape:n'], nodes: [{ id: 'n', expanded: true }] };
    first.receive({ type: 'canvasPresentation', state: presentation });
    let finish!: (result: { status: 'passed'; detail: string }) => void;
    let signal!: AbortSignal;
    let runs = 0;
    const profile = { kind: 'model', model: 'net:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} }, limits: { timeout_ms: 1000 } };
    const validation = new LocalBrowserValidation({ validate: async (_target: unknown, running: AbortSignal) => {
      runs++; signal = running;
      return new Promise(resolve => { finish = resolve; });
    } } as never, message => canvas.post(message), {
      snapshot: async (_root, selected) => (selected
        ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected } } } }
        : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
          { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
        ] } }) as never,
      choose: async () => undefined,
    },
    new ValidationSummaryStore(summary => canvas.post({ type: 'validationState', ...summary })));
    validation.handle({ type: 'validateRequest', target: model, requestId: 'run', messageId: 'request',
      workspaceSessionId: 'native', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION });
    await tick();
    await canvas.hide(); first.dispose();
    assert.equal(signal.aborted, false);
    finish({ status: 'passed', detail: 'Saved model passed.' }); await tick();
    await canvas.show(); const next = webviewState.views[1]; ready(next);
    assert.equal(runs, 1);
    assert.equal(signal.aborted, false);
    assert.equal(publications(next).find(message => message.type === 'validateResult').status, 'passed');
    assert.equal(publications(next).find(message => message.type === 'validationState').status, 'passed');
    assert.deepEqual((next.messages.find((m: any) => m.type === 'canvasConnected') as any).presentation, presentation);
  });

  test('receipt replay preserves the completed revision instead of certifying newer measurements', async () => {
    create(); await canvas.show(); const view = webviewState.views[0]; ready(view);
    canvas.post({ type: 'projection', target: { ...model, backendRevision: 3 }, projection: {}, traced: true, greyed: false });
    canvas.post({ type: 'validationState', target: { ...model, backendRevision: 3 },
      runId: 'run', status: 'passed', detail: 'Passed.', projectionRevision: 3 });
    canvas.post({ type: 'validateResult', target: model, requestId: 'run', status: 'passed', detail: 'Passed.' });
    assert.equal(publications(view).filter(message => message.type === 'validationState').length, 1,
      'a terminal receipt does not invent a second host summary');
    view.setVisible(false);
    canvas.post({ type: 'projection', target: { ...model, backendRevision: 4 }, projection: {}, traced: true, greyed: false });
    view.setVisible(true);
    const receipt = publications(view).find(message => message.type === 'validationState');
    assert.equal(receipt.target.backendRevision, 3);
    assert.equal(receipt.projectionRevision, 3);
  });

  test('source refresh keeps old outcomes stale, while explicit retargeting drops them', async () => {
    create(); await canvas.show(); const first = webviewState.views[0]; ready(first);
    canvas.post({ type: 'validationState', target: model, runId: 'run', status: 'passed', detail: 'Previous saved source.' });
    canvas.post({ type: 'validateResult', target: model, requestId: 'run', status: 'passed', detail: 'Previous saved source.' });
    canvas.post({ type: 'projection', target: { ...model, documentVersion: 2, sourceDigest: 'b'.repeat(64), backendRevision: 3 },
      projection: {}, traced: true, greyed: true });
    first.dispose(); await canvas.show(); const next = webviewState.views[1]; ready(next);
    assert.equal(publications(next).find(message => message.type === 'validationState').target.sourceDigest, model.sourceDigest);
    canvas.post({ type: 'activeTarget', target: { ...model, cls: 'Other' } });
    next.dispose(); await canvas.show(); const third = webviewState.views[2]; ready(third);
    assert.equal(publications(third).filter(message => message.type === 'validationState').length, 0);
  });

  test('hide waits for the final checkpoint and an intervening Show wins', async () => {
    create(); await canvas.show(); const view = webviewState.views[0];
    view.receive({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION });
    const hidden = canvas.hide();
    await tick();
    const request = view.messages.find((m: any) => m.type === 'canvasCheckpoint') as any;
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), true, 'view must remain alive until it answers');
    view.receive({ type: 'canvasPresentation', requestId: 'old', state: {} });
    await tick();
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), true);
    await canvas.show();
    const state = { camera: { x: 99, y: 10, z: 0.6 } };
    view.receive({ type: 'canvasPresentation', requestId: request.requestId, state });
    await hidden;
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), true, 'late hide cannot override a newer show');
    assert.deepEqual(canvas.inspectForTest().presentation, state);
  });

  test('overlapping hides cannot remove the view before the newest checkpoint answers', async () => {
    create(); await canvas.show(); const view = webviewState.views[0];
    view.receive({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION });
    const first = canvas.hide();
    const second = canvas.hide();
    await first;
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), true);
    const requests = view.messages.filter((m: any) => m.type === 'canvasCheckpoint') as any[];
    view.receive({ type: 'canvasPresentation', requestId: requests[0].requestId });
    await tick();
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), true);
    view.receive({ type: 'canvasPresentation', requestId: requests[1].requestId });
    await second;
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), false);
  });

  test('an unresponsive renderer cannot block hide; disposal releases a pending checkpoint', async () => {
    create(); await canvas.show(); const view = webviewState.views[0];
    view.receive({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION });
    await canvas.hide();
    assert.equal(webviewState.contexts.get(CANVAS_REQUESTED), false);
    await canvas.show();
    const pending = canvas.hide();
    view.dispose();
    await pending;
    assert.equal(canvas.visible, false);
  });

  test('failure is visible, redacted and retryable; concurrent retry starts once', async () => {
    let fail = true;
    create(async () => { if (fail) throw new Error('secret-token and source body'); return target; });
    await canvas.show();
    const view = webviewState.views[0];
    assert.match(homeHtml(view), /could not start/);
    assert.match(view.webview.html, /Explore your model/);
    assert.doesNotMatch(homeHtml(view), /secret-token/);
    fail = false;
    await Promise.all([canvas.show(), canvas.show()]);
    assert.equal(prepares, 2);
    ready(view); assert.equal(replays, 1);
    canvas.setError(); await canvas.show(); ready(view);
    assert.equal(prepares, 3, 'backend death retries with the same owner');
  });

  test('an environment restart reconnects the retained view and presentation without revealing a hidden Canvas', async () => {
    let current = target;
    create(async () => current);
    await canvas.show(); const view = webviewState.views[0]; ready(view);
    const presentation = { camera: { x: 20, y: 30, z: 0.5 } };
    view.receive({ type: 'canvasPresentation', state: presentation });
    await canvas.hide(); view.setVisible(false);
    const commands = webviewState.commands.length;
    await canvas.prepareBackendRestart();
    assert.match(view.webview.html, /Switching Python environment/);
    assert.equal(canvas.inspectForTest().ready, false);
    current = { ...target, appUrl: 'http://127.0.0.1:9001/?token=new', appOrigin: 'http://127.0.0.1:9001', port: 9001 };
    await canvas.reconnect(); ready(view);
    assert.match(view.webview.html, /frame-src http:\/\/127\.0\.0\.1:9001;/);
    assert.equal(canvas.requestedVisible, false);
    assert.equal(canvas.visible, false);
    assert.deepEqual(canvas.inspectForTest().presentation, presentation);
    assert.equal(publications(view).filter(message => message.type === 'snapshot').at(-1).state.activeTarget.cls, 'Net');
    assert.ok(webviewState.commands.slice(commands).every(command => command.id === 'setContext'));
    assert.equal(received.length, 0, 'reconnecting must not execute the retained model');
  });

  test('disposal during startup cannot resurrect the view', async () => {
    let release!: (target: ShellTarget) => void;
    create(() => new Promise(resolve => { release = resolve; }));
    const showing = canvas.show(); await tick();
    const view = webviewState.views[0], writes = view.htmlWrites;
    canvas.dispose(); release(target); await showing;
    assert.equal(view.htmlWrites, writes);
    assert.equal(replays, 0);
  });

  test('the bridge accepts only current-session typed view requests', async () => {
    create(); await canvas.show(); const view = webviewState.views[0]; ready(view);
    const session = (view.messages.find((m: any) => m.type === 'canvasConnected') as any).sessionId;
    const request = { type: 'validateRequest', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: session, messageId: 'request', requestId: 'run', target: model };
    view.receive({ type: 'canvasMessage', message: { ...request, workspaceSessionId: 'other-window' } });
    view.receive({ type: 'canvasMessage', message: { ...request, type: 'projection' } });
    assert.equal(received.length, 0);
    view.receive({ type: 'canvasMessage', message: request });
    assert.deepEqual(received, [request]);
  });

  test('semantic inspection, portable export and candidates reach the shared controller', async () => {
    create(); await canvas.show(); const view = webviewState.views[0]; ready(view);
    const identity = view.messages.find((m: any) => m.type === 'canvasConnected') as any;
    const common = { protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION, workspaceSessionId: identity.sessionId,
      messageId: 'semantic', requestId: 'request', target: model };
    const requests = [
      { ...common, type: 'captureInspectRequest', query: { captureId: 'capture:1', sourceId: 'source:1', path: [], offset: 0, limit: 20 } },
      { ...common, type: 'captureExportRequest', reference: { captureId: 'capture:1', sourceId: 'source:1' } },
      { ...common, type: 'captureExportCancel' },
      { ...common, type: 'candidateRequest', viewId: identity.viewId, proposalId: 'proposal', action: { kind: 'list' } },
      { ...common, type: 'graphSetupRequest', openingId: 'opening', action: { kind: 'submit', draft: { profile: { kind: 'model', model: 'net:Net' } } } },
    ];
    for (const message of requests) view.receive({ type: 'canvasMessage', message });
    assert.deepEqual(received, requests, 'supported semantic requests must not silently time out in the sidebar');
    view.receive({ type: 'canvasMessage', message: { ...requests[3], viewId: 'foreign' } });
    assert.equal(received.length, requests.length, 'candidate ownership still uses the host-assigned view');
  });

  test('terminal receipts and replay retain the authoritative attempt evidence', async () => {
    create(); await canvas.show(); const view = webviewState.views[0]; ready(view);
    const summaries = new ValidationSummaryStore(summary => canvas.post({ type: 'validationState', ...summary }));
    const attempt = summaries.begin('failed-run', model);
    attempt('failed', 'Forward failed before capture.');
    const summary = summaries.snapshot(model)!;
    canvas.post({ type: 'validateResult', target: model, requestId: 'failed-run', status: 'failed', detail: summary.detail });
    view.setVisible(false); view.setVisible(true);
    assert.deepEqual(publications(view).filter(message => message.type === 'validationState').at(-1), {
      ...publications(view).filter(message => message.type === 'validationState').at(-1), ...summary,
    }, 'replay must not replace ledger evidence with a guessed projection revision');
    assert.deepEqual(canvas.inspectForTest().receipts.find(message => message.type === 'validationState'),
      { type: 'validationState', ...summary });
  });

  test('MCP identity belongs to the host; visibility and recreation require fresh selection and receipt', async () => {
    create(); await canvas.show(); const view = webviewState.views[0];
    assert.deepEqual(connections, [], 'an unready renderer cannot answer MCP');
    ready(view);
    const identity = view.messages.find((m: any) => m.type === 'canvasConnected') as any;
    assert.equal(typeof identity.viewId, 'string');
    assert.deepEqual(connections, [{ viewId: identity.viewId, connected: true }]);
    const selection = { type: 'viewSelection', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: identity.sessionId, messageId: 'selection', target: model,
      viewId: identity.viewId, nodeIds: ['n'], edgeIds: [] };
    view.receive({ type: 'canvasMessage', message: { ...selection, viewId: 'another-view' } });
    assert.equal(received.length, 0);
    view.receive({ type: 'canvasMessage', message: selection });
    assert.deepEqual(received, [selection]);
    const focus = { type: 'focusView' as const, viewId: identity.viewId, target: model, requestId: 'focus', nodeIds: ['n'], edgeIds: [] };
    assert.equal(canvas.post(focus), true);
    await canvas.hide(); view.setVisible(false);
    assert.deepEqual(connections.at(-1), { viewId: identity.viewId, connected: false });
    assert.equal(canvas.post(focus), false, 'hidden focus is not replayed or reported as delivered');
    view.receive({ type: 'canvasMessage', message: selection });
    assert.equal(received.length, 1);
    await canvas.show();
    assert.deepEqual(connections.at(-1), { viewId: identity.viewId, connected: true });
    assert.ok(view.messages.some((m: any) => m.type === 'canvasVisible'));
    view.dispose(); await canvas.show(); const next = webviewState.views[1]; ready(next);
    const recreated = next.messages.find((m: any) => m.type === 'canvasConnected') as any;
    assert.equal(recreated.viewId, identity.viewId, 'one logical native view per window');
    assert.notEqual(recreated.sessionId, identity.sessionId, 'old renderer traffic cannot own the recreated connection');
    next.receive({ type: 'canvasMessage', message: selection });
    assert.equal(received.length, 1);
    canvas.setError();
    assert.deepEqual(connections.at(-1), { viewId: identity.viewId, connected: false });
  });
});
