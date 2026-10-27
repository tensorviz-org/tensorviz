import * as assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { McpValidationRuns, registerMcpValidationTools } from '@tensorviz/graph-host/mcpValidation';
import type { WorkerEnvelope } from '../../worker';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { TraceWorker } from '../../worker';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import type * as vscode from 'vscode';

suite('MCP retained evidence with a real Python worker', function () {
  this.timeout(90_000);
  test('planned capture, production projection and retained MCP inspection use the same failed execution', async () => {
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'Set TENSORVIZ_TEST_PYTHON; canonical worker integration must not silently skip.');
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-mcp-canonical-'));
    const source = `import torch\nfrom torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.down = nn.AvgPool2d(2)\n        self.later = nn.Conv2d(1, 2, 1)\n    def forward(self, x):\n        a = self.down(x)\n        return self.later(a + x)\n`;
    const file = path.join(root, 'net.py'), backend = path.resolve(__dirname, '../../../../backend');
    let model: McpModel = { current: true, target: { workspaceFolder: root, file: 'net.py', cls: 'Net',
      documentVersion: 1, backendRevision: 1, sourceDigest: createHash('sha256').update(source).digest('hex') },
      projection: { revision: 1, traced: false, spanmap: {}, document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    const graph = new McpGraphService('canonical-worker', () => model);
    const worker = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel, { interpreter: async () => python });
    const summaries = new ValidationSummaryStore(() => {});
    let payload: any;
    const runs = new McpValidationRuns(() => model, async (_target, signal, _inputs, _progress, measured, captured, execution, _job, attempt) => {
      const cancel = () => worker.cancel();
      signal.addEventListener('abort', cancel, { once: true });
      const outcome = await worker.run({ file, cls: 'Net', workspace: root, execution, timeoutMs: 20000,
        attemptId: attempt!.snapshot().id, onResolvedPlan: snapshot => attempt!.resolved(snapshot) })
        .finally(() => signal.removeEventListener('abort', cancel));
      attempt!.captured(outcome); captured(outcome);
      assert.ok(outcome.envelope?.result, JSON.stringify(outcome));
      payload = (outcome.envelope.result as any).diagnostics.semantic_capture;
      const document = await new Promise<any>((resolve, reject) => {
        const child = execFile(python, ['-B', '-c', `import json,sys\nfrom services.ide.projection import project\np=json.load(sys.stdin)\nr=project(dict(nodes=[],edges=[]),p['envelope'],cls='Net',rel_file='net.py',source=p['source'])\nprint(r.document.model_dump_json(by_alias=True))`],
          { cwd: backend, timeout: 20000, maxBuffer: 8 * 1024 * 1024, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } },
          (error, stdout) => { try { if (error) reject(error); else resolve(JSON.parse(stdout)); } catch (e) { reject(e); } });
        child.stdin!.end(JSON.stringify({ envelope: outcome.envelope, source }));
      });
      model = { ...model, target: { ...model.target, backendRevision: 2 },
        projection: { ...model.projection, traced: true, revision: 2, document } };
      measured(model.target, model.projection);
      return { status: 'failed', detail: 'Captured failing residual.' };
    }, () => {}, summaries);
    const server = new McpServer({ name: 'canonical-worker-test', version: '1' });
    registerMcpValidationTools(server, graph, runs, 'owner');
    const client = new Client({ name: 'reader', version: '1' });
    const call = async (name: string, args: object) => (await client.callTool({ name, arguments: { ...args } })).structuredContent as any;
    try {
      await fs.writeFile(file, source);
      const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
      const started = await call('validate_model', { ...graph.reference(model), inputs: [{ shape: [1, 1, 56, 56], dtype: 'float32', fill: 'ones' }] });
      const runId = started.runId;
      let status: any;
      const deadline = Date.now() + 60000;
      do {
        await new Promise(resolve => setTimeout(resolve, 100));
        status = await call('get_run', { runId });
      } while (!status.finishedAt && Date.now() < deadline);
      assert.equal(status.status, 'failed', JSON.stringify(status));
      assert.equal(status.capture.id, payload.id);
      assert.equal(status.capture.status, 'partial');
      const failing = payload.operations.find((op: any) => op.status === 'failed');
      assert.equal(failing.target, 'aten.add.Tensor');
      model.current = false; model.target.documentVersion++;
      await fs.writeFile(file, source.replace('a + x', 'a'));
      const failure = await call('inspect_run', { runId, section: 'failure' });
      assert.equal(failure.format, 'semantic-v1');
      assert.deepEqual(failure.matchingFailureOperations.items.map((op: any) => op.id), [failing.id]);
      assert.ok(failure.upstream.items.some((op: any) => payload.operations.find((item: any) => item.id === op.id).target.includes('avg_pool2d')));
      const graphPage = await call('inspect_run', { runId, section: 'graph' });
      assert.equal(graphPage.snapshot.capture.id, payload.id);
      assert.ok(graphPage.nodes.items.some((node: any) => node.modulePaths?.includes('down')));
      const args = await call('inspect_run', { runId, section: 'capture', entity: { kind: 'operation', id: failing.id }, path: ['arguments', 'args'] });
      assert.equal(args.page.items.length, failing.arguments.args.length);
      for (const arg of failing.arguments.args) if (arg.kind === 'value') {
        const value = payload.values.find((v: any) => v.id === arg.id);
        const page = await call('inspect_run', { runId, section: 'capture', entity: { kind: 'value', id: arg.id }, path: ['shape'] });
        assert.deepEqual(page.page.items.map((item: any) => item.value), value.shape);
      }
    } finally {
      runs.clear(); worker.dispose(); await client.close(); await server.close(); await fs.rm(root, { recursive: true, force: true });
    }
  });

  test('a failing residual remains inspectable after the live model changes', async () => {
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'Set TENSORVIZ_TEST_PYTHON to a Torch/Pydantic environment; this integration test must not silently skip.');
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-mcp-evidence-'));
    const source = `import torch\nfrom torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.down = nn.AvgPool2d(2)\n        self.later = nn.Conv2d(1, 2, 1)\n    def forward(self, x):\n        return self.later(self.down(x) + x)\n`;
    const file = path.join(root, 'net.py');
    const backend = path.resolve(__dirname, '../../../../backend');
    let model: McpModel = { current: true, target: { workspaceFolder: root, file: 'net.py', cls: 'Net',
      documentVersion: 1, backendRevision: 1, sourceDigest: createHash('sha256').update(source).digest('hex') },
      projection: { revision: 1, traced: false, spanmap: {}, document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    const graph = new McpGraphService('real-worker', () => model);
    const runs = new McpValidationRuns(() => model, async (_target, signal, _inputs, progress, measured, captured, execution) => {
      progress('tracing');
      const envelope = await new Promise<WorkerEnvelope>((resolve, reject) => {
        execFile(python, ['-m', 'services.ingest.worker', '--file', file, '--cls', 'Net', '--workspace', root,
          '--execution', JSON.stringify(execution)], { cwd: backend, signal, timeout: 60_000, maxBuffer: 8 * 1024 * 1024,
          env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', PYTHON_DOTENV_DISABLED: '1', KMP_DUPLICATE_LIB_OK: 'TRUE' } },
        (error, stdout) => { try { resolve(JSON.parse(stdout)); } catch { reject(error ?? new Error('Missing worker envelope')); } });
      });
      captured({ ok: false, reason: 'Forward failed.', detail: '', exitCode: 4, envelope });
      const raw = (envelope.result as any).graph;
      // This test exercises the SDK/store/worker seam. Native projection merging
      // has its own LocalBrowserSync and installed integration checks.
      model = { ...model, target: { ...model.target, backendRevision: 2 }, projection: { ...model.projection,
        revision: 2, traced: true, spanmap: envelope.spans, document: { semantic: {
          nodes: raw.nodes.map((n: any) => ({ node_id: n.id, op: n.data.layerType.name, params: n.data.parameters })),
          edges: raw.edges.map((e: any) => ({ edge_id: e.id, source_node_id: e.source, target_node_id: e.target })),
        } } } };
      measured(model.target, model.projection);
      return { status: 'failed', detail: 'Measured residual shape mismatch.' };
    }, () => {});
    const server = new McpServer({ name: 'real-worker-test', version: '1' });
    registerMcpValidationTools(server, graph, runs, 'owner');
    const client = new Client({ name: 'standard-sdk-client', version: '1' });
    try {
      await fs.writeFile(file, source);
      const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
      const tools = await client.listTools(); assert.ok(tools.tools.some(tool => tool.name === 'inspect_run'));
      const started = await client.callTool({ name: 'validate_model', arguments: { ...graph.reference(model),
        execution: { inputs: { args: [{ shape: [1, 1, 56, 56], dtype: 'float32', fill: 'ones' }] } } } });
      assert.ok(!started.isError, JSON.stringify(started));
      const runId = (started.structuredContent as any).runId;
      let status: any;
      const deadline = Date.now() + 65_000;
      do {
        await new Promise(resolve => setTimeout(resolve, 100));
        status = (await client.callTool({ name: 'get_run', arguments: { runId } })).structuredContent;
      } while (!status.finishedAt && Date.now() < deadline);
      assert.equal(status.status, 'failed', JSON.stringify(status));
      model = { ...model, target: { ...model.target, documentVersion: 9, sourceDigest: 'b'.repeat(64) },
        projection: { ...model.projection, document: { semantic: { nodes: [], edges: [] } } } };
      await fs.writeFile(file, source.replace('self.down(x) + x', 'self.down(x)'));
      const failure = (await client.callTool({ name: 'inspect_run', arguments: { runId, section: 'failure' } })).structuredContent as any;
      assert.equal(failure.immutable, true);
      assert.equal(failure.failure.operation, 'aten.add.Tensor');
      assert.deepEqual(failure.failure.inputs.map((t: any) => t.shape), [[1, 1, 28, 28], [1, 1, 56, 56]]);
      assert.equal(await fs.realpath(failure.failure.source.file), await fs.realpath(file));
      assert.ok(failure.upstream.items.some((operation: any) => operation.modulePath === 'down'));
      const frozen = (await client.callTool({ name: 'inspect_run', arguments: { runId, section: 'graph' } })).structuredContent as any;
      assert.ok(frozen.nodes.items.length > 0); assert.equal(frozen.target.documentVersion, 1);
      const setup = (await client.callTool({ name: 'inspect_run', arguments: { runId, section: 'setup' } })).structuredContent as any;
      assert.equal(setup.setup.environment.device, 'cpu'); assert.ok(setup.setup.state.digest);
    } finally {
      runs.clear(); await client.close(); await server.close(); await fs.rm(root, { recursive: true, force: true });
    }
  });
});
