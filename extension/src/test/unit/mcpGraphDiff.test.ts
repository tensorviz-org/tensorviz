import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { openGraphDiff, registerMcpGraphDiffTool } from '../../mcpGraphDiff';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { emptyReviewProjection } from '../../graphReview';
import type { ReviewSession } from '../../reviewSnapshots';
import type { GraphReviewState, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { editorState, openStubEditor, resetEditors, state, Uri } from '../vscodeStub';

suite('MCP visual graph diff', () => {
  let root: string, client: Client, server: McpServer, current: GraphReviewState | undefined;
  let sessions: ReviewSession[], live: LocalBrowserTargetIdentity | undefined, busy: boolean, shows: number;
  let host: Parameters<typeof openGraphDiff>[1];
  const source = (activation: string) => `raise RuntimeError('must not execute')\nfrom torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        self.act = nn.${activation}()\n`;
  setup(async () => {
    resetEditors(); sessions = []; current = undefined; live = undefined; busy = false; shows = 0;
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-mcp-diff-')));
    await reviewGit(root, ['init', '-q']);
    await fs.writeFile(path.join(root, 'model.py'), source('ReLU'));
    await reviewGit(root, ['add', '.']);
    await reviewGit(root, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'baseline']);
    await fs.writeFile(path.join(root, 'model.py'), source('GELU'));
    state.workspaceFolders = [{ uri: Uri.file(root) }];
    host = { target: () => live, busy: () => busy, show: async () => { shows++; return true; },
      current: () => current,
      review: async session => {
        sessions.push(session);
        current = { id: `review-${sessions.length}`, beforeLabel: session.before.label, afterLabel: session.after.label,
          before: emptyReviewProjection('Net'), after: emptyReviewProjection('Net'), changedFiles: ['model.py'], stale: false,
          models: [{ id: 'net', label: 'Net', file: 'model.py', status: 'modified' }], selectedModel: 'net' };
        return current;
      } };
    server = new McpServer({ name: 'tensorviz', version: 'test' });
    registerMcpGraphDiffTool(server, host);
    client = new Client({ name: 'recording-agent', version: 'test' });
    const [left, right] = InMemoryTransport.createLinkedPair();
    await server.connect(left); await client.connect(right);
  });
  teardown(async () => {
    await client?.close(); await server?.close(); resetEditors(); state.workspaceFolders = undefined;
    await fs.rm(root, { recursive: true, force: true });
  });
  const call = async (args: Record<string, unknown> = {}) => {
    const response = await client.callTool({ name: 'open_graph_diff', arguments: args });
    return { error: response.isError, data: response.structuredContent as any };
  };
  test('standard tool opens HEAD versus local changes without a capture, execution, or workspace writes', async () => {
    const before = await reviewGit(root, ['status', '--porcelain']);
    const result = await call({ file: path.join(root, 'model.py'), cls: 'Net' });
    assert.equal(result.error, undefined); assert.equal(result.data.status, 'opened');
    assert.equal(result.data.comparison.executed, false); assert.equal(result.data.selectedModel.label, 'Net');
    for (const field of ['viewId', 'counts', 'edgeCounts', 'capturedContext', 'unmappedChanges']) {
      assert.equal(field in result.data, false, 'opening a comparison must not claim Canvas rendering evidence');
    }
    assert.deepEqual(result.data.changedFiles, ['model.py']);
    assert.equal(result.data.coverage.incomplete, false);
    assert.equal(await sessions[0].before.read('model.py'), source('ReLU'));
    assert.equal(await sessions[0].after.read('model.py'), source('GELU'));
    assert.equal(sessions[0].preferredClass, 'Net'); assert.equal(shows, 1);
    assert.equal(await reviewGit(root, ['status', '--porcelain']), before);
    assert.deepEqual((await fs.readdir(root)).sort(), ['.git', 'model.py']);
    const tool = (await client.listTools()).tools[0];
    assert.match(tool.description!, /do not call open_graph to recapture/);
  });
  test('unsaved buffers take precedence without saving or changing the index', async () => {
    await reviewGit(root, ['add', 'model.py']);
    openStubEditor(path.join(root, 'model.py'), source('SiLU'));
    editorState.dirtyDocuments.add(path.join(root, 'model.py'));
    const index = await reviewGit(root, ['show', ':model.py']);
    await call();
    assert.equal(await sessions[0].before.read('model.py'), source('ReLU'));
    assert.equal(await sessions[0].after.read('model.py'), source('SiLU'));
    assert.equal(await fs.readFile(path.join(root, 'model.py'), 'utf8'), source('GELU'));
    assert.equal(await reviewGit(root, ['show', ':model.py']), index);
    assert.equal(editorState.appliedEdits.length, 0);
  });
  test('a repository below a non-Git workspace opens from the model directory', async () => {
    state.workspaceFolders = [{ uri: Uri.file(path.dirname(root)) }];
    const result = await call({ file: path.join(root, 'model.py') });
    assert.equal(result.error, undefined);
    assert.equal(sessions[0].folder, root);
    assert.equal(await sessions[0].before.read('model.py'), source('ReLU'));
  });
  test('an ignored nested clone uses its own HEAD rather than the workspace repository', async () => {
    const nested = path.join(root, 'nanoGPT');
    await fs.mkdir(nested);
    await fs.writeFile(path.join(root, '.gitignore'), 'nanoGPT/\n');
    await reviewGit(nested, ['init', '-q']);
    await fs.writeFile(path.join(nested, 'model.py'), source('SiLU'));
    await reviewGit(nested, ['add', '.']);
    await reviewGit(nested, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'nested']);
    await fs.writeFile(path.join(nested, 'model.py'), source('GELU'));
    const result = await call({ file: path.join(nested, 'model.py') });
    assert.equal(result.error, undefined);
    assert.equal(sessions[0].folder, nested);
    assert.equal(await sessions[0].before.read('model.py'), source('SiLU'));
  });
  test('Git errors preserve the actual cause for an agent', async () => {
    await reviewGit(root, ['checkout', '--orphan', 'no-history']);
    assert.match((await call({ file: path.join(root, 'model.py') })).data.error.message, /no readable HEAD commit/);
    await fs.rm(path.join(root, '.git'), { recursive: true });
    assert.match((await call({ file: path.join(root, 'model.py') })).data.error.message, /not a git repository/i);
    const previous = process.env.PATH;
    try {
      process.env.PATH = '';
      assert.match((await call({ file: path.join(root, 'model.py') })).data.error.message, /git.*ENOENT/);
    } finally { process.env.PATH = previous; }
    assert.equal(shows, 0);
  });
  test('the retained model wins over an unrelated editor without retargeting it', async () => {
    live = { workspaceFolder: root, file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
    openStubEditor(path.join(root, 'unrelated.py'), '');
    const result = await call();
    assert.equal(sessions[0].preferredFile, 'model.py'); assert.equal(sessions[0].preferredClass, 'Net');
    assert.equal(result.data.selectedModel.label, 'Net'); assert.equal(live.cls, 'Net');
  });
  test('completion waits for source comparison preparation without a Canvas receipt', async () => {
    const original = host.review;
    let prepare!: () => void;
    host.review = async (session, signal) => { await new Promise<void>(resolve => { prepare = resolve; }); return original(session, signal); };
    let finished = false;
    const pending = call({ file: path.join(root, 'model.py') }).then(value => { finished = true; return value; });
    while (!prepare) await new Promise(resolve => setImmediate(resolve));
    assert.equal(finished, false);
    prepare(); assert.equal((await pending).data.status, 'opened');
  });
  test('source coverage limits remain explicit without renderer-derived counts', async () => {
    const review = host.review;
    host.review = async (session, signal) => {
      const result = await review(session, signal);
      return current = { ...result!, notes: ['An imported source file is unavailable.'] };
    };
    const result = await call({ file: path.join(root, 'model.py') });
    assert.equal(result.data.coverage.incomplete, true);
    assert.deepEqual(result.data.coverage.notes, ['An imported source file is unavailable.']);
  });
  test('stale source or a replaced review cannot be reported as opened', async () => {
    const review = host.review;
    for (const change of [{ stale: true }, { id: 'replaced' }]) {
      host.review = async (session, signal) => {
        const result = await review(session, signal);
        current = { ...result!, ...change }; return result;
      };
      assert.equal((await call({ file: path.join(root, 'model.py') })).data.error.code, 'STALE_REVIEW');
    }
  });
  test('an empty inventory explains that no affected model was discovered', async () => {
    const review = host.review;
    host.review = async (session, signal) => {
      const result = await review(session, signal);
      return current = { ...result!, models: [], selectedModel: undefined };
    };
    const result = await call({ file: path.join(root, 'model.py') });
    assert.equal(result.data.selectedModel, null);
    assert.match(result.data.detail, /no affected model was discovered/);
  });
  test('missing targets, workspace access, pending proposals and missing Git history fail before Canvas opens', async () => {
    assert.equal((await call()).data.error.code, 'MODEL_REQUIRED');
    assert.equal((await call({ file: 'model.py' })).data.error.code, 'ABSOLUTE_PATH_REQUIRED');
    assert.equal((await call({ file: '/outside/model.py' })).data.error.code, 'WORKSPACE_REQUIRED');
    busy = true;
    assert.equal((await call({ file: path.join(root, 'model.py') })).data.error.code, 'BUSY');
    busy = false;
    await reviewGit(root, ['checkout', '--orphan', 'empty']);
    assert.equal((await call({ file: path.join(root, 'model.py') })).data.error.code, 'GIT_COMPARISON_UNAVAILABLE');
    assert.equal(shows, 0);
  });
  test('cancellation never starts a comparison', async () => {
    const controller = new AbortController(); controller.abort();
    await assert.rejects(openGraphDiff({ file: path.join(root, 'model.py') }, host, controller.signal), { code: 'CANCELLED' });
    assert.equal(shows, 0); assert.equal(sessions.length, 0);
  });
});
