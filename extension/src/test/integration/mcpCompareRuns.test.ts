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

suite('MCP comparison of real controlled worker runs', function () {
  this.timeout(150_000);
  test('compares controlled validate_model runs after live edits', async () => {
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'Set TENSORVIZ_TEST_PYTHON to the existing Torch/Pydantic environment.');
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-mcp-comparison-'));
    const file = path.join(root, 'net.py'), backend = path.resolve(__dirname, '../../../../backend');
    const source = `import torch\nfrom torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4, bias=False)\n        self.act = nn.ReLU()\n        self.head = nn.Linear(4, 2, bias=False)\n    def forward(self, *, x):\n        return self.head(self.act(self.fc(x)))\n`;
    let version = 1;
    let model: McpModel = { current: true, target: { workspaceFolder: root, file: 'net.py', cls: 'Net', documentVersion: version,
      backendRevision: version, sourceDigest: createHash('sha256').update(source).digest('hex') },
      projection: { revision: version, traced: false, spanmap: {}, document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    const env = { ...process.env, PYTHONDONTWRITEBYTECODE: '1', PYTHON_DOTENV_DISABLED: '1', KMP_DUPLICATE_LIB_OK: 'TRUE' };
    const graph = new McpGraphService('comparison-worker', () => model);
    let workerCount = 0;
    const runs = new McpValidationRuns(() => model, async (_target, signal, _inputs, progress, measured, captured, execution, job) => {
      progress('tracing'); workerCount++;
      const envelope = await new Promise<WorkerEnvelope>((resolve, reject) => {
        execFile(python, ['-m', 'services.ingest.worker', '--file', file, '--cls', 'Net', '--workspace', root,
          '--execution', JSON.stringify(execution), ...(job ? ['--diagnostic-job', JSON.stringify(job)] : [])], { cwd: backend, env, signal, timeout: 60_000, maxBuffer: 20 * 1024 * 1024 },
          (error, stdout) => { try { resolve(JSON.parse(stdout)); } catch { reject(error ?? new Error('Missing worker envelope')); } });
      });
      const ok = !envelope.error;
      captured(ok ? { ok: true, exitCode: 0, envelope } : { ok: false, reason: 'Forward failed', detail: '', exitCode: 4, envelope });
      const raw = (envelope.result as any).graph;
      version++;
      // SDK/store/real-worker integration; installed native projection is a separate gate.
      model = { ...model, target: { ...model.target, backendRevision: version }, projection: { ...model.projection,
        revision: version, traced: true, spanmap: envelope.spans, document: { semantic: {
          nodes: raw.nodes.map((n: any) => ({ node_id: n.id, op: n.data.layerType.name, params: n.data.parameters })),
          edges: raw.edges.map((e: any) => ({ edge_id: e.id, source_node_id: e.source, target_node_id: e.target })),
        } } } };
      measured(model.target, model.projection);
      return { status: ok ? 'passed' : 'failed', detail: 'Real controlled execution.' };
    }, () => {});
    const server = new McpServer({ name: 'real-comparison', version: '1' }); registerMcpValidationTools(server, graph, runs, 'owner');
    const client = new Client({ name: 'standard-sdk-client', version: '1' });
    async function call(name: string, args: Record<string, unknown>): Promise<any> {
      const result = await client.callTool({ name, arguments: args }); assert.ok(!result.isError, JSON.stringify(result));
      return result.structuredContent;
    }
    async function run(checkpoint: string, fixture = 'batch.pt'): Promise<string> {
      const { runId } = await call('validate_model', { ...graph.reference(model), execution: {
        inputs: { fixture }, checkpoint: { file: checkpoint }, device: 'cpu', mode: 'eval', seed: 4, deterministic: true } });
      const deadline = Date.now() + 65_000;
      let result: any;
      do { await new Promise(resolve => setTimeout(resolve, 100)); result = await call('get_run', { runId }); }
      while (!result.finishedAt && Date.now() < deadline);
      assert.equal(result.status, fixture === 'batch.pt' ? 'passed' : 'failed', JSON.stringify(result)); return runId;
    }
    try {
      await fs.writeFile(file, source);
      await new Promise<void>((resolve, reject) => execFile(python, ['-c', `import torch\nfrom net import Net\nm=Net()\nwith torch.no_grad():\n m.fc.weight.fill_(0.25)\n m.head.weight.fill_(0.5)\ntorch.save(m.state_dict(), 'base.pt')\nwith torch.no_grad():\n m.head.weight.add_(0.5)\ntorch.save(m.state_dict(), 'changed.pt')\ntorch.save({'kwargs': {'x': torch.tensor([[1., 2., 3.]])}, 'batch': {'labels': torch.tensor([1])}}, 'batch.pt')\ntorch.save({'kwargs': {'x': torch.ones(1, 4)}}, 'bad.pt')`],
        { cwd: root, env, timeout: 30_000 }, error => error ? reject(error) : resolve()));
      const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
      assert.ok((await client.listTools()).tools.some(tool => tool.name === 'compare_runs'));
      const baselineRunId = await run('base.pt');
      const refactored = source.replace('return self.head(self.act(self.fc(x)))', 'hidden = self.act(self.fc(x))\n        return self.head(hidden)');
      await fs.writeFile(file, refactored);
      model.target = { ...model.target, documentVersion: ++version, sourceDigest: createHash('sha256').update(refactored).digest('hex') };
      const refactorRunId = await run('base.pt'), changedRunId = await run('changed.pt'), failedRunId = await run('base.pt', 'bad.pt');
      model = { ...model, current: false, target: { ...model.target, documentVersion: ++version, sourceDigest: 'f'.repeat(64) } };
      await fs.writeFile(file, '# current live source is now unrelated\n');
      const refactor = await call('compare_runs', { baselineRunId, candidateRunId: refactorRunId });
      assert.equal(refactor.outputs.allClose, true); assert.equal(refactor.setup.recordedControlsMatch, true);
      assert.equal(refactor.firstObservedDivergence, null); assert.equal(refactor.nodeChangeCount, 0); assert.equal(refactor.connectionChangeCount, 0);
      const changed = await call('compare_runs', { baselineRunId, candidateRunId: changedRunId });
      assert.equal(changed.outputs.allClose, false); assert.equal(changed.setup.weights.status, 'different');
      assert.equal(changed.setup.inputs.status, 'same'); assert.equal(changed.firstObservedDivergence.modulePath, 'head');
      const outputs = await call('compare_runs', { baselineRunId, candidateRunId: changedRunId, section: 'outputs', limit: 1 });
      assert.equal(outputs.items[0].mismatchedElements, 2); assert.equal(outputs.items[0].maxAbsDifference, 3);
      const failure = await call('compare_runs', { baselineRunId, candidateRunId: failedRunId });
      assert.deepEqual(failure.outcomes, { baseline: 'passed', candidate: 'failed' }); assert.equal(failure.outputs.allClose, null);
      assert.equal(failure.setup.inputs.status, 'different'); assert.ok(failure.coverageChangeCount > 0);
      assert.equal(workerCount, 4, 'comparisons must read immutable records without executing again');
    } finally {
      runs.clear(); await client.close(); await server.close(); await fs.rm(root, { recursive: true, force: true });
    }
  });
});
