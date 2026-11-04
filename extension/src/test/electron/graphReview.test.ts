import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import * as net from 'node:net';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { LOCAL_BROWSER_PROTOCOL_VERSION, type GraphReviewState } from '@tensorviz/contracts/protocol';

const REPO = path.resolve(__dirname, '../../../..');
suite('graphReview', () => {
  let root: string, api: any, provider: vscode.Disposable, before: string, after: string;
  const source = (width: number) => `from torch import nn\nraise RuntimeError('Review must not execute this source')\nclass Encoder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, ${width})\n    def forward(self, x):\n        return self.fc(x)\nclass Decoder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(${width}, 4)\n    def forward(self, x):\n        return self.fc(x)\n`;
  const current = (): GraphReviewState | undefined => api.graphReviewState();
  const until = async (test: () => boolean, label = 'review state did not arrive') => {
    const deadline = Date.now() + 30_000;
    while (!test()) { assert.ok(Date.now() < deadline, label); await new Promise(resolve => setTimeout(resolve, 100)); }
  };
  suiteSetup(async function () {
    this.timeout(240_000);
    assert.ok(process.env.TENSORVIZ_TEST_PYTHON);
    process.env.BACKEND_PYTHON = process.env.TENSORVIZ_TEST_PYTHON;
    api = (await vscode.extensions.getExtension('tensorviz.tensorviz-vscode')!.activate()).__test;
    await vscode.workspace.getConfiguration('tensorviz').update('repoPath', REPO, vscode.ConfigurationTarget.Global);
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-direct-review-')));
    await reviewGit(root, ['init', '-q']);
    await fs.writeFile(path.join(root, 'model.py'), source(8)); await reviewGit(root, ['add', '.']);
    const commit = async (label: string) => {
      await reviewGit(root, ['-c', 'user.name=Local test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qam', label]);
      return (await reviewGit(root, ['rev-parse', 'HEAD'])).trim();
    };
    before = await commit('base');
    await fs.writeFile(path.join(root, 'model.py'), source(16)); after = await commit('change');
    provider = vscode.workspace.registerTextDocumentContentProvider('review', {
      provideTextDocumentContent: uri => reviewGit(root, ['show', `${JSON.parse(uri.query).commit}:model.py`]),
    });
  });
  suiteTeardown(async () => {
    provider?.dispose();
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');
    await vscode.commands.executeCommand('tensorviz.hideCanvas');
    await vscode.workspace.getConfiguration('tensorviz').update('repoPath', undefined, vscode.ConfigurationTarget.Global);
    if (root) await fs.rm(root, { recursive: true, force: true });
  });
  test('public Graph diff command discovers models without selecting or executing a live model', async function () {
    this.timeout(180_000);
    const priorTarget = api.localBrowserTarget();
    const file = path.join(root, 'model.py');
    const uri = (commit: string) => vscode.Uri.from({ scheme: 'review', path: file, query: JSON.stringify({ path: file, commit }) });
    await vscode.commands.executeCommand('vscode.diff', uri(before), uri(after), 'Code review');
    await vscode.commands.executeCommand('tensorviz.graphDiffFromEditor');
    await until(() => !!current());
    assert.equal(current()!.models!.length, 2);
    assert.equal(current()!.before.traced, false); assert.equal(current()!.after.traced, false);
    assert.equal(current()!.modelStatus, 'modified');
    assert.deepEqual(api.localBrowserTarget(), priorTarget, 'review must not require, select or retarget a live executable model');
    assert.equal(await reviewGit(root, ['status', '--porcelain']), '');
  });
  test('model selection reaches the backend and opens the same immutable native comparison', async () => {
    await until(() => !!api.sidebarState()?.ready);
    const second = current()!.models![1];
    await api.post({ type: 'canvasMessage', message: { type: 'graphReviewRequest', action: 'selectModel', reviewId: current()!.id, modelId: second.id,
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION, workspaceSessionId: api.sidebarState().sessionId, messageId: 'select-second-model' } });
    await until(() => current()!.selectedModel === second.id);
    assert.equal(current()!.after.cls, second.label);
    const diffs = vscode.window.tabGroups.all.flatMap(group => group.tabs).filter(tab => tab.input instanceof vscode.TabInputTextDiff);
    assert.equal(diffs.length, 1, 'reuse the source diff instead of multiplying code panels');
  });
  test('moving into another class in the native diff follows that model', async () => {
    const editor = vscode.window.visibleTextEditors.find(editor => editor.document.uri.scheme === 'review' && JSON.parse(editor.document.uri.query).commit === after)!;
    assert.ok(editor);
    await until(() => editor.selection.active.line === 8, 'selected class was not revealed');
    const focused = await vscode.window.showTextDocument(editor.document, { viewColumn: editor.viewColumn, preserveFocus: false });
    focused.selection = new vscode.Selection(5, 0, 5, 0);
    await until(() => current()!.after.cls === 'Encoder', 'cursor did not select Encoder');
  });
  test('current Git added and deleted files keep the repository context', async () => {
    const empty = createHash('sha1').update('tree 0\0').digest('hex');
    const file = path.join(root, 'new.py');
    await fs.writeFile(file, source(32)); await reviewGit(root, ['add', 'new.py']);
    const git = await vscode.extensions.getExtension('vscode.git')!.activate();
    const repository = await git.getAPI(1).openRepository(vscode.Uri.file(root));
    assert.ok(repository, 'the built-in Git extension must own this repository');
    await repository.status();
    const uri = (ref: string) => vscode.Uri.from({ scheme: 'git', path: file, query: JSON.stringify({ path: file, ref }) });
    const previous = current()!.id;
    await vscode.commands.executeCommand('vscode.diff', uri(empty), uri(''), 'Added model file');
    await vscode.commands.executeCommand('tensorviz.graphDiffFromEditor');
    await until(() => !!current() && current()!.id !== previous);
    assert.equal(current()!.models!.length, 2, 'unchanged models from other files must not become changes');
    assert.ok(current()!.models!.every(model => model.file === 'new.py' && model.status === 'added'));
    const added = current()!.id;
    await fs.rm(file);
    await repository.status();
    await vscode.commands.executeCommand('vscode.diff', uri(''), uri(empty), 'Deleted model file');
    await vscode.commands.executeCommand('tensorviz.graphDiffFromEditor');
    await until(() => !!current() && current()!.id !== added);
    assert.ok(current()!.models!.every(model => model.file === 'new.py' && model.status === 'removed'));
    // Restore the disposable fixture's clean state for the network acceptance leg.
    await reviewGit(root, ['reset', '-q', '--', 'new.py']);
  });
  test('SCM and compatibility commands open the current sidebar review without creating legacy view files', async function () {
    this.timeout(180_000);
    const file = path.join(root, 'model.py');
    await fs.writeFile(file, source(24)); await reviewGit(root, ['add', 'model.py']);
    await fs.writeFile(file, source(32));
    const document = await vscode.workspace.openTextDocument(file);
    await vscode.window.showTextDocument(document);
    const status = await reviewGit(root, ['status', '--porcelain', '--untracked-files=all']);
    for (const [command, title, argument] of [
      ['tensorviz.graphDiffStaged', 'Staged changes', { resourceUri: document.uri }],
      ['tensorviz.graphDiffWorking', 'Unstaged changes', { resourceUri: document.uri }],
      ['tensorviz.compareWithHead', 'Changes since HEAD', document.uri],
    ] as const) {
      const previous = current()!.id;
      await vscode.commands.executeCommand(command, argument);
      await until(() => current()?.id !== previous);
      assert.equal(current()!.title, title);
      assert.equal(current()!.after.traced, false);
      assert.ok(api.sidebarState());
      assert.equal(vscode.window.tabGroups.all.flatMap(group => group.tabs).some(tab =>
        tab.input instanceof vscode.TabInputWebview && tab.input.viewType.includes('tensorviz.canvas')), false);
    }
    assert.equal(await reviewGit(root, ['status', '--porcelain', '--untracked-files=all']), status);
    assert.equal((await fs.readdir(root)).includes('.tensorviz'), false);
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');
    await reviewGit(root, ['reset', '-q', 'HEAD', '--', 'model.py']); await fs.writeFile(file, source(16));
  });
  test('MCP opens a local comparison without rendering confirmation or model execution', async function () {
    this.timeout(180_000);
    const file = path.join(root, 'model.py');
    await fs.writeFile(file, source(32));
    const document = await vscode.workspace.openTextDocument(file);
    await vscode.window.showTextDocument(document);
    const previousTarget = api.localBrowserTarget();
    const beforeStatus = await reviewGit(root, ['status', '--porcelain']);
    const socket = net.createServer();
    await new Promise<void>(resolve => socket.listen(0, '127.0.0.1', resolve));
    const port = (socket.address() as net.AddressInfo).port;
    await new Promise<void>(resolve => socket.close(() => resolve()));
    // Enable MCP only in this test's disposable profile, without touching agent settings.
    const config = path.join(api.globalStorageDir(), 'mcp-connection.json');
    await fs.writeFile(config, JSON.stringify({ enabled: true, port }));
    await until(() => !!api.mcpConnection(), 'MCP did not start in the disposable host');
    const client = new Client({ name: 'graph-diff-acceptance', version: '1' });
    try {
      await client.connect(new StreamableHTTPClientTransport(new URL(api.mcpConnection().url)));
      assert.ok((await client.listTools()).tools.some(tool => tool.name === 'open_graph_diff'));
      const response = await client.callTool({ name: 'open_graph_diff', arguments: { file, cls: 'Decoder' } });
      assert.ok(!response.isError, JSON.stringify(response.structuredContent));
      const result = response.structuredContent as any;
      assert.equal(result.status, 'opened'); assert.equal(result.selectedModel.label, 'Decoder');
      assert.equal(result.reviewId, current()!.id);
      assert.equal(current()!.after.cls, 'Decoder'); assert.equal(current()!.after.traced, false);
      assert.equal(result.comparison.executed, false);
      assert.equal('counts' in result, false); assert.equal('viewId' in result, false);
      assert.deepEqual(result.changedFiles, ['model.py']);
      assert.deepEqual(api.localBrowserTarget(), previousTarget, 'diff must not retarget or recapture the live model');
      assert.equal(await fs.readFile(file, 'utf8'), source(32));
      assert.equal(await reviewGit(root, ['status', '--porcelain']), beforeStatus);
    } finally {
      await client.close();
      await fs.writeFile(config, JSON.stringify({ enabled: false, port }));
      await vscode.commands.executeCommand('workbench.action.closeAllEditors');
      await fs.writeFile(file, source(16));
    }
  });
  if (process.env.TENSORVIZ_TEST_REVIEW_PR) test('opens an existing GitHub PR through the public command without cloning or selecting a model', async function () {
    this.timeout(300_000);
    const previous = current()!.id;
    await vscode.commands.executeCommand('tensorviz.openPullRequest', process.env.TENSORVIZ_TEST_REVIEW_PR);
    await until(() => !!current() && current()!.id !== previous);
    assert.ok(current()!.title?.includes('#1410'), 'the acceptance PR is pytorch/examples#1410');
    assert.ok(current()!.models!.some(model => model.file === 'mixture_of_experts/mixtral_moe.py'));
    assert.equal(current()!.after.traced, false);
    const pinned = current()!.id;
    await vscode.commands.executeCommand('tensorviz.graphDiffFromEditor');
    assert.equal(current()!.id, pinned, 'Graph diff on our own documents must keep the complete PR session');
    assert.equal(await reviewGit(root, ['status', '--porcelain']), '');
  });

});
