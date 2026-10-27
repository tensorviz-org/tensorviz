import * as assert from 'node:assert/strict';
import { createLocalBrowserHostTransport, LocalBrowserHostTransport } from '@tensorviz/graph-host/localBrowserHostTransport';
import { LocalWebSocketHandlers } from '@tensorviz/graph-host/localBrowserSocket';
import { LocalBrowserMessage } from '@tensorviz/contracts/protocol';
import { LocalBrowserGraph } from '../../localBrowserGraph';
import { LocalBrowserProjectDeclareMessage, LocalBrowserProjectIdentity } from '@tensorviz/contracts/protocol';
import { sha256 } from '../../sourceChanges';
import { editorState, reset, resetEditors, state, stubSourceIO } from '../vscodeStub';

const vscode = require('../vscodeStub');
const BEFORE = 'from model import Net\nmodel = Net()\n';
const AFTER = 'import tensorviz\ntensorviz.edge("model", "trainer", "feeds")\n' + BEFORE;
const FILE = '/work/train.py';
const DEP = '/work/model.py';
const TOKEN = 'private-launch-token';

async function until(check: () => boolean): Promise<void> {
  for (let i = 0; i < 100; i++) {
    if (check()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.ok(check(), 'expected host operation to settle');
}

/** Drive the actual graph message handler, project index, reviewSource toolbar
 * and SourceWriter. Only VS Code primitives and the authenticated backend are fake.
 * F4 tests inject host calls; F5 tests enable the actual validating host transport. */
suite('localBrowserGraph — reviewed project declarations', () => {
  let graph: LocalBrowserGraph;
  let restore: Array<() => void>;
  let receive: (message: unknown) => void;
  let published: any[], requests: any[], diffs: any[], notices: string[], reads: string[];
  let commands: Map<string, () => void>, providers: Map<string, any>;
  let tabs: any[], contexts: boolean[];
  let declarationReply: (body: any, signal: AbortSignal) => Promise<Response>;
  let beforeRead: (file: string) => void;
  let entrypoint: string, origin: string | undefined;
  let messageId: number;
  let useWire: boolean, failIndex: boolean;
  let wire: LocalBrowserHostTransport, socket: LocalWebSocketHandlers;
  let manifests: Array<Array<{ file: string; digest: string }>>;

  function patch(object: any, key: string, value: any) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, writable: true, value });
    restore.push(() => descriptor ? Object.defineProperty(object, key, descriptor) : delete object[key]);
  }
  function reply(overrides = {}) {
    return new Response(JSON.stringify({ status: 'ok', file: 'train.py', expected_digest: sha256(BEFORE), contents: AFTER, ...overrides }));
  }
  function project(): LocalBrowserProjectIdentity {
    return published.filter(item => item.type === 'projectIndex' || item.type === 'projectDocument').at(-1).project;
  }
  function message(overrides = {}): LocalBrowserProjectDeclareMessage {
    return { type: 'projectDeclare', protocolVersion: 1, workspaceSessionId: 'session', messageId: `declare-${++messageId}`,
      project: project(), declaration: 'edge', args: ['model', 'trainer', 'feeds'], ...overrides };
  }
  function apply() { commands.get('tensorviz.applySourceProposal')!(); }
  function discard() { commands.get('tensorviz.discardSourceProposal')!(); }
  async function finished() { await until(() => notices.length + state.warningMessages.length > 0 && commands.size === 0); }
  async function open() {
    await graph.openProject({ folder: '/work' });
    await until(() => published.some(item => item.type === 'projectDocument'));
  }

  setup(() => {
    useWire = false; failIndex = false; manifests = [];
    reset(); resetEditors(); restore = []; published = []; requests = []; diffs = []; notices = []; reads = [];
    providers = new Map(); commands = new Map(); tabs = []; contexts = [];
    beforeRead = () => {}; entrypoint = 'train.py'; origin = 'http://127.0.0.1:9901'; messageId = 0;
    state.workspaceFolders = [{ uri: { fsPath: '/work' } }];
    editorState.documents.set(FILE, BEFORE); editorState.documents.set(DEP, 'class Net: pass\n');
    declarationReply = async () => reply();
    const openDocument = vscode.workspace.openTextDocument;
    patch(vscode.workspace, 'openTextDocument', async (uri: any) => {
      reads.push(uri.fsPath); beforeRead(uri.fsPath);
      const document = await openDocument(uri);
      return Object.assign(document, { positionAt(offset: number) {
        const lines = document.getText().slice(0, offset).split('\n');
        return new vscode.Position(lines.length - 1, lines.at(-1)!.replace(/\r$/, '').length);
      } });
    });
    patch(vscode.Uri, 'from', (value: any) => ({ ...value, toString: () => `${value.scheme}:${value.path}` }));
    patch(vscode, 'TabInputTextDiff', class { constructor(public original: any, public modified: any) {} });
    patch(vscode.workspace, 'registerTextDocumentContentProvider', (scheme: string, provider: any) => {
      providers.set(scheme, provider); return { dispose: () => providers.delete(scheme) };
    });
    patch(vscode, 'commands', {
      registerCommand(name: string, action: () => void) {
        assert.ok(!commands.has(name), 'native source reviews must never overlap');
        commands.set(name, action); return { dispose: () => commands.delete(name) };
      },
      async executeCommand(name: string, ...args: any[]) {
        if (name === 'setContext') { contexts.push(args[1]); return; }
        assert.equal(name, 'vscode.diff');
        const [before, after, title] = args, provider = providers.get(after.scheme);
        diffs.push({ before: provider.provideTextDocumentContent(before), after: provider.provideTextDocumentContent(after), title });
        (editorState.visibleTextEditors as any[]).push({ document: { uri: after } });
        tabs.push({ input: new vscode.TabInputTextDiff(before, after) });
      },
    });
    patch(vscode.window, 'tabGroups', {
      get all() { return [{ tabs }]; }, onDidChangeTabs: () => ({ dispose() {} }),
      close: async () => { tabs = []; },
    });
    patch(vscode.window, 'showInformationMessage', async (text: string) => {
      if (!text.includes('editor toolbar')) notices.push(text);
    });
    graph = new LocalBrowserGraph({
      channel: { appendLine() {} }, backend: { origin: () => origin, launchToken: () => TOKEN },
      launchOrigin: () => undefined, remoteName: () => undefined, ensureBackend: async () => true, openExternal: async () => true,
      sourceIO: stubSourceIO,
      projectSourceIO: folder => ({ ...stubSourceIO(folder),
        openDocument: file => vscode.workspace.textDocuments.find((doc: { uri: { fsPath: string } }) => doc.uri.fsPath === `${folder}/${file}`),
        readIgnore: async () => undefined }),
      mintSession: async () => ({ workspaceSessionId: 'session', launchOrigin: origin!, expiresAt: '', hostCookieKey: 'key',
        viewCredential: 'view-key', pageUrl: `${origin}/` }),
      createTransport: options => {
        if (useWire) {
          wire = createLocalBrowserHostTransport({ ...options, reconnectDelaysMs: [0], connectSocket: (_url, handlers) => {
            socket = handlers;
            receive = value => handlers.onText(JSON.stringify(value));
            queueMicrotask(() => {
              handlers.onOpen();
              receive({ type: 'connected', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'connected', role: 'host',
                session: { workspaceSessionId: 'session', protocolVersion: 1, role: 'host', expiresAt: 'later',
                  capabilities: { readOnly: true, canNavigateSource: true, canWriteSource: false, canRunAgent: false, canEditGraph: false } } });
            });
            return { sendText: text => { published.push(JSON.parse(text)); return true; }, close() {} };
          } });
          return wire;
        }
        return { status: { phase: 'connected', epoch: 1, error: null }, protocolViolations: 0, viewPresence: 'attached',
          connect: async () => {}, dispose() {}, publish: value => { published.push(value); return null; },
          // F4 deliberately injects malformed host calls; F5 tests below use the real transport.
          onMessage: handler => { receive = value => handler(value as LocalBrowserMessage); return () => {}; },
          onStatusChange: () => () => {},
        };
      },
      fetchImpl: (async (url, init) => {
        const body = JSON.parse(String(init?.body));
        if (String(url).endsWith('/project/index')) {
          if (failIndex) throw new Error('private backend exception with source and token');
          manifests.push(body.manifest);
          return new Response(JSON.stringify({ status: 'ok', digest: sha256(JSON.stringify(body.manifest)),
          index: { experiments: ['exp:train', 'exp:other'].map(experiment_id => ({ experiment_id, label: experiment_id,
            entrypoint, root: '', closure: ['train.py', 'model.py'], graph: { nodes: [{ label: 'model' }, { label: 'trainer' }] } })),
          roots: [''], unreached: [], truncated: false } }));
        }
        if (String(url).endsWith('/project/document')) return new Response(JSON.stringify({ status: 'ok',
          document: { experiment_id: body.experiment_id, name: body.experiment_id, nodes: [{ id: 'net', data: {} }], edges: [] }, spanmap: {}, expanded: body.expanded }));
        assert.ok(String(url).endsWith('/project/declare'));
        requests.push({ body, signal: init!.signal, headers: init!.headers });
        return declarationReply(body, init!.signal as AbortSignal);
      }) as typeof fetch,
    });
  });
  teardown(() => { graph?.dispose(); for (const undo of restore.reverse()) undo(); });

  test('real transport publishes project state and Apply re-indexes the changed buffer without reopening', async () => {
    useWire = true;
    const originalApply = vscode.workspace.applyEdit;
    patch(vscode.workspace, 'applyEdit', async (edit: typeof editorState.appliedEdits[number]) => {
      const accepted = await originalApply(edit);
      for (const change of edit.replacements) editorState.documents.set(change.uri.fsPath, change.newText);
      return accepted;
    });
    await open();
    const before = project();
    receive(message()); await until(() => diffs.length === 1);
    assert.equal(editorState.documents.get(FILE), BEFORE, 'wire declaration cannot bypass native review');
    apply(); await finished();
    assert.equal(editorState.documents.get(FILE), AFTER);
    assert.equal(manifests.length, 2, 'normal Apply automatically refreshes once');
    assert.equal(manifests[1].find(item => item.file === 'train.py')?.digest, sha256(AFTER));
    assert.notEqual(project().indexDigest, before.indexDigest);
    assert.equal(project().indexRevision, before.indexRevision + 1);
    assert.equal(published.at(-1).type, 'projectDocument');
    assert.equal(wire.protocolViolations, 0);
    assert.ok(!notices.some(text => text.includes('Reopen')));
  });

  test('real transport drops malformed declarations and serializes valid declarations through the shared review guard', async () => {
    useWire = true; await open();
    receive(message({ args: ['one'] }));
    receive(message({ target: {} }));
    assert.equal(wire.protocolViolations, 2);
    assert.equal(requests.length, 0);
    receive(message()); await until(() => diffs.length === 1);
    receive(message()); await until(() => state.warningMessages.length === 1);
    assert.equal(requests.length, 1); assert.equal(diffs.length, 1);
    discard(); await finished();
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('real transport replays focus and expansion after reload and host reconnect without indexing again', async () => {
    useWire = true; await open();
    receive({ ...message(), type: 'projectFocus', declaration: undefined, args: undefined, experimentId: 'exp:other' });
    await until(() => published.at(-1).project?.experimentId === 'exp:other');
    receive({ type: 'projectExpand', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'expand', project: project(), nodeId: 'net', expanded: true });
    await until(() => published.at(-1).expanded?.includes('net'));
    const ask = () => receive({ type: 'snapshotRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'reload', role: 'view', lastHostSequence: 0 });
    const before = published.length; ask();
    assert.deepEqual(published.slice(before).map(item => item.type), ['snapshot', 'projectIndex', 'projectDocument']);
    assert.deepEqual(published.at(-1).expanded, ['net']);
    const oldSocket = socket;
    oldSocket.onClose({ kind: 'closed', code: 1006, reason: '' });
    await new Promise(resolve => setTimeout(resolve, 5));
    await until(() => socket !== oldSocket && wire.status.phase === 'connected');
    ask();
    assert.equal(published.at(-3).hostSequence, 1);
    assert.equal(published.at(-1).project.experimentId, 'exp:other');
    assert.equal(manifests.length, 1);
  });

  test('index errors are safe, actionable and replayed after reload; retry recovers', async () => {
    useWire = true; failIndex = true;
    await graph.openProject({ folder: '/work' });
    assert.match(published.at(-1).message, /Open Project as Graph/);
    assert.ok(!JSON.stringify(published).includes('private backend exception'));
    const before = published.length;
    receive({ type: 'snapshotRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'reload', role: 'view', lastHostSequence: 0 });
    assert.deepEqual(published.slice(before).map(item => item.type), ['snapshot', 'error']);
    failIndex = false; await open();
    assert.equal(published.at(-1).type, 'projectDocument');
    assert.equal(wire.protocolViolations, 0);
  });

  test('class/project switches revoke source authority and replay only the selected surface', async () => {
    useWire = true; await open();
    const old = message();
    await graph.open({ uri: DEP, cls: 'Net' });
    await until(() => published.some(item => item.type === 'activeTarget'));
    const before = published.length;
    receive(old);
    receive({ type: 'snapshotRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'class-reload', role: 'view', lastHostSequence: 0 });
    assert.equal(published.slice(before).some(item => item.type.startsWith('project')), false);
    assert.equal(requests.length, 0);
    await open();
    assert.equal(graph.target, undefined, 'the previous class has no execution or source authority on the project surface');
    const replay = published.length;
    receive({ type: 'snapshotRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'project-reload', role: 'view', lastHostSequence: 0 });
    assert.deepEqual(published.slice(replay).map(item => item.type), ['snapshot', 'projectIndex', 'projectDocument']);
  });

  test('shows the full native diff; only toolbar Apply submits one unsaved, undoable WorkspaceEdit', async () => {
    await open(); receive(message()); await until(() => diffs.length === 1);
    assert.equal(editorState.appliedEdits.length, 0);
    assert.deepEqual(diffs[0], { before: BEFORE, after: AFTER, title: 'tensorViz project declaration — train.py' });
    assert.equal(requests[0].body.file, 'train.py'); assert.equal(requests[0].body.source, BEFORE);
    assert.equal(requests[0].headers['X-tensorViz-Token'], TOKEN);
    assert.deepEqual(requests[0].body.known_labels, ['model', 'trainer']);
    apply(); await finished();
    assert.equal(editorState.appliedEdits.length, 1);
    const changes = editorState.appliedEdits[0].replacements;
    assert.equal(changes.length, 1); assert.equal(changes[0].uri.fsPath, FILE); assert.equal(changes[0].newText, AFTER);
    assert.equal(editorState.documents.get(FILE), BEFORE, 'unit stub records edits without saving or mutating disk');
    assert.ok(notices.some(text => text.includes('Undo')));
    assert.deepEqual(contexts, [true, false]); assert.equal(providers.size, 0); assert.equal(tabs.length, 0);
    assert.ok(!JSON.stringify(published).includes(TOKEN));
  });

  for (const changed of [false, true]) test(`native declaration review reopens a closed dependency (changed=${changed})`, async () => {
    await open();
    const original = vscode.workspace.openTextDocument;
    let frozen: any, closed = false, reopened = false;
    patch(vscode.workspace, 'openTextDocument', async (uri: any) => {
      const document = await original(uri);
      if (uri.fsPath === DEP) {
        if (frozen) reopened = true;
        else {
          frozen = document;
          const text = document.getText(), version = document.version;
          document.getText = () => text;
          Object.defineProperty(document, 'isClosed', { get: () => closed });
          Object.defineProperty(document, 'version', { get: () => version });
        }
      }
      return document;
    });
    receive(message()); await until(() => diffs.length === 1);
    assert.ok(frozen);
    closed = true;
    if (changed) editorState.documents.set(DEP, '# changed after closure\n');
    apply(); await finished();
    assert.equal(reopened, true);
    assert.equal(editorState.appliedEdits.length, changed ? 0 : 1);
  });

  test('native Discard writes nothing and releases the shared review controls', async () => {
    await open(); receive(message()); await until(() => diffs.length === 1); discard(); await finished();
    assert.equal(editorState.appliedEdits.length, 0); assert.equal(providers.size, 0);
    assert.match(notices.at(-1)!, /discarded/);
  });

  for (const field of ['workspaceFolder', 'projectRoot', 'experimentId', 'indexDigest', 'indexRevision'] as const) {
    test(`foreign ${field} cannot grant filesystem authority`, async () => {
      await open(); reads = [];
      receive(message({ project: { ...project(), [field]: field === 'indexRevision' ? 999 : 'foreign' } })); await finished();
      assert.deepEqual(reads, []); assert.deepEqual(requests, []); assert.deepEqual(diffs, []);
      assert.equal(editorState.appliedEdits.length, 0); assert.equal(state.warningMessages.length, 1);
    });
  }
  for (const mode of ['untrusted', 'dirty-entrypoint', 'dirty-dependency', 'unknown-label', 'bad-arity', 'path-escape'] as const) {
    test(`refuses ${mode} before proposing Python`, async () => {
      if (mode === 'path-escape') entrypoint = '../outside.py';
      await open();
      if (mode === 'untrusted') state.isTrusted = false;
      if (mode === 'dirty-entrypoint') editorState.documents.set(FILE, '# new work\n');
      if (mode === 'dirty-dependency') editorState.documents.set(DEP, '# new work\n');
      receive(message(mode === 'unknown-label' ? { args: ['model', 'foreign'] } : mode === 'bad-arity' ? { args: ['model'] } : {}));
      await finished(); assert.deepEqual(requests, []); assert.deepEqual(diffs, []); assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  for (const moment of ['preparation', 'review', 'final-read'] as const) {
    for (const change of ['text', 'version', 'dependency', 'trust'] as const) {
      test(`${change} changed during ${moment} cannot overwrite newer work`, async () => {
        await open();
        const mutate = () => {
          if (change === 'text') editorState.documents.set(FILE, '# user edit\n');
          if (change === 'version') editorState.versions.set(FILE, 2);
          if (change === 'dependency') editorState.documents.set(DEP, '# dependency edit\n');
          if (change === 'trust') state.isTrusted = false;
        };
        if (moment === 'preparation') declarationReply = async () => { mutate(); return reply(); };
        receive(message());
        if (moment !== 'preparation') {
          await until(() => diffs.length === 1);
          if (moment === 'review') mutate();
          else beforeRead = mutate;
          apply();
        }
        await finished();
        assert.equal(diffs.length, moment === 'preparation' ? 0 : 1);
        assert.equal(editorState.appliedEdits.length, 0); assert.equal(state.warningMessages.length, 1);
      });
    }
  }

  for (const moment of ['request', 'review'] as const) {
    for (const change of ['focus', 'reindex', 'class', 'dispose'] as const) {
      test(`${change} cancels a pending ${moment}, including an uncooperative backend`, async () => {
        await open(); let release!: (response: Response) => void;
        if (moment === 'request') declarationReply = () => new Promise(resolve => { release = resolve; });
        receive(message()); await until(() => moment === 'request' ? requests.length === 1 : diffs.length === 1);
        if (change === 'focus') receive({ type: 'projectFocus', project: project(), experimentId: 'exp:other' });
        if (change === 'reindex') await graph.openProject({ folder: '/work' });
        if (change === 'class') await graph.open({ uri: FILE, cls: 'Net' });
        if (change === 'dispose') graph.dispose();
        await finished();
        if (moment === 'request') {
          assert.equal(requests[0].signal.aborted, true); release(reply());
          await new Promise(resolve => setImmediate(resolve)); assert.equal(diffs.length, 0);
        }
        assert.equal(editorState.appliedEdits.length, 0); assert.equal(providers.size, 0);
      });
    }
  }

  test('project, browser and MCP proposals cannot open concurrent reviews', async () => {
    await open(); receive(message()); await until(() => diffs.length === 1);
    receive(message()); await until(() => state.warningMessages.length === 1);
    receive({ type: 'sourceEditRequest', requestId: 'browser-edit', target: {}, edit: {} });
    await until(() => published.some(item => item.type === 'sourceEditResult'));
    assert.match(published.find(item => item.type === 'sourceEditResult').detail, /Another source review/);
    // This is the same host entry point McpProposals calls, with no browser reply.
    const result = await (graph as any).sourceEdits.propose({ requestId: 'mcp-edit', target: {}, edit: {} });
    assert.match(result.detail, /Another source review/);
    assert.equal(requests.length, 1); assert.equal(diffs.length, 1);
    discard(); await finished();
  });

  for (const failure of ['network', 'http', 'json', 'wrong-file', 'wrong-digest', 'missing-contents', 'no-change', 'backend-restart'] as const) {
    test(`surfaces ${failure} without source/credential leaks or a review`, async () => {
      await open(); declarationReply = async () => {
        if (failure === 'network') throw new Error(`${TOKEN}: ${BEFORE}`);
        if (failure === 'http') return new Response(`${TOKEN}: ${BEFORE}`, { status: 500 });
        if (failure === 'json') return new Response('not JSON');
        if (failure === 'backend-restart') { origin = undefined; return reply(); }
        return reply(failure === 'wrong-file' ? { file: '../outside.py' } : failure === 'wrong-digest' ? { expected_digest: 'b'.repeat(64) }
          : failure === 'missing-contents' ? { contents: null } : { contents: BEFORE });
      };
      receive(message()); await finished();
      assert.equal(state.warningMessages.length, 1); assert.equal(diffs.length, 0); assert.equal(editorState.appliedEdits.length, 0);
      assert.ok(!state.warningMessages[0].includes(TOKEN)); assert.ok(!state.warningMessages[0].includes(BEFORE));
    });
  }

  test('ten-second deadline bounds response decoding and ignores a late success', async () => {
    await open(); const schedule = global.setTimeout; let expire!: () => void;
    patch(global, 'setTimeout', ((callback: () => void, ms: number, ...args: any[]) => {
      if (ms === 10_000) { expire = callback; return schedule(() => {}, 60_000); }
      return schedule(callback, ms, ...args);
    }) as typeof setTimeout);
    let release!: (value: unknown) => void;
    declarationReply = async () => ({ ok: true, json: () => new Promise(resolve => { release = resolve; }) }) as Response;
    receive(message()); await until(() => !!release); expire(); await finished();
    assert.equal(requests[0].signal.aborted, true);
    release(await reply().json()); await new Promise(resolve => setImmediate(resolve));
    assert.equal(diffs.length, 0); assert.equal(editorState.appliedEdits.length, 0); assert.equal(state.warningMessages.length, 1);
  });

  for (const failure of ['reject', 'throw'] as const) {
    test(`editor ${failure} is surfaced and does not claim success`, async () => {
      await open(); patch(vscode.workspace, 'applyEdit', async () => {
        if (failure === 'throw') throw new Error(TOKEN);
        return false;
      });
      receive(message()); await until(() => diffs.length === 1); apply(); await finished();
      assert.equal(state.warningMessages.length, 1); assert.ok(!state.warningMessages[0].includes(TOKEN));
      if (failure === 'throw') assert.match(state.warningMessages[0], /Inspect the buffer/);
      assert.ok(!notices.some(text => text.includes('Applied')));
    });
  }
});
