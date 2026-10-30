import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { createHash } from 'node:crypto';
import type * as vscode from 'vscode';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { TraceWorker } from '../../worker';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { registerMcpCaptureTools } from '@tensorviz/graph-host/mcpCaptures';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';

const repo = path.resolve(__dirname, '../../../..');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const source = (tied: boolean) => `import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(3, 2, bias=False)
        self.right = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.left.weight.fill_(1)
            self.right.weight.fill_(1)
        if ${tied ? 'True' : 'False'}:
            self.right.weight = self.left.weight
        self.unused = torch.nn.Linear(3, 2, bias=False)
        self.register_buffer('bias', torch.ones(2))
    def forward(self, x):
        return self.left(x) + self.right(x) + self.bias
`;

suite('named state queries with real CPU capture and MCP', function () {
  this.timeout(40000);
  let root: string, worker: TraceWorker;
  suiteSetup(async () => {
    assert.ok(existsSync(python), 'Set TENSORVIZ_TEST_PYTHON to a Torch-capable interpreter.');
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-state-query-')));
    worker = new TraceWorker(path.join(repo, 'backend'), { appendLine() {} } as unknown as vscode.OutputChannel,
      { interpreter: async () => python });
  });
  suiteTeardown(async () => {
    worker?.dispose(); if (root) await fs.rm(root, { recursive: true, force: true });
  });
  for (const tied of [false, true]) test(`${tied ? 'tied' : 'distinct equal-valued'} weights preserve their actual uses`, async () => {
    const folder = path.join(root, tied ? 'tied' : 'separate'); await fs.mkdir(folder);
    const text = source(tied), file = path.join(folder, 'model.py'); await fs.writeFile(file, text);
    const target = { workspaceFolder: folder, file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: createHash('sha256').update(text).digest('hex'), backendRevision: 1 };
    const ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger);
    const attemptId = `state-${tied ? 'tied' : 'separate'}`, attempt = ledger.begin(attemptId, target, 'manual', null);
    const outcome = await worker.run({ workspace: folder, file, cls: 'Model', attemptId, timeoutMs: 20000,
      execution: executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 3], dtype: 'float32', fill: 'ones' }] } }),
      onResolvedPlan: snapshot => attempt.resolved(snapshot) });
    assert.equal(outcome.ok, true, JSON.stringify(outcome));
    attempt.captured(outcome); attempt.finish('passed'); ledger.projected(target, outcome.envelope);
    const capture = inspector.latest(target)?.capture;
    assert.ok(capture, 'Only a validated canonical capture can establish state usage');
    assert.equal(capture.status, 'complete');
    const model: McpModel = { target, current: true, projection: {
      document: { semantic: { nodes: [], edges: [] } }, spanmap: {}, traced: true,
    } as unknown as ProjectionPayload };
    const graph = new McpGraphService('state-oracle', () => model), server = new McpServer({ name: 'state-oracle', version: '1' });
    graph.register(server); registerMcpCaptureTools(server, graph, inspector);
    const client = new Client({ name: 'state-reader', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
    try {
      const find = async (text: string, kinds: string[]) =>
        (await client.callTool({ name: 'find_computation', arguments: { modelId: graph.reference(model).modelId,
          captureId: capture.id, sourceId: capture.source.id, text, kinds } })).structuredContent as any;
      const discovered = await find('left', ['module', 'parameter']);
      const moduleAnchor = discovered.items.find((item: any) => item.ref.kind === 'module');
      const stateAnchor = discovered.items.find((item: any) => item.ref.kind === 'parameter');
      assert.ok(moduleAnchor && stateAnchor, 'Readable names must reveal exact recorded anchors.');
      assert.ok(moduleAnchor.record.paths.includes('left'));
      assert.equal(stateAnchor.ref.id, 'left.weight');
      const moduleUses = (await client.callTool({ name: 'query_computation', arguments: {
        modelId: graph.reference(model).modelId, captureId: capture.id, sourceId: capture.source.id,
        anchors: [moduleAnchor.ref], depth: 0 } })).structuredContent as any;
      assert.ok(moduleUses.items.some((item: any) => item.kind === 'entity' && item.ref.kind === 'operation'));
      assert.equal((await find('unused.weight', ['parameter'])).total, 0, 'Unobserved state must not be invented.');
      const query = async (kind: 'parameter' | 'buffer', name: string, depth = 0) =>
        (await client.callTool({ name: 'query_computation', arguments: { modelId: graph.reference(model).modelId,
          captureId: capture.id, sourceId: capture.source.id, anchors: [{ kind, id: name }],
          direction: 'downstream', depth } })).structuredContent as any;
      const left = await query('parameter', 'left.weight'), right = await query('parameter', 'right.weight');
      const values = (page: any) => page.items.filter((item: any) => item.kind === 'entity' && item.ref.kind === 'value');
      assert.equal(values(left).length, 1); assert.equal(values(right).length, 1);
      const l = values(left)[0].record, r = values(right)[0].record;
      assert.equal(l.objectId === r.objectId, tied);
      assert.equal(l.id === r.id, tied);
      assert.deepEqual(l.shape, r.shape);
      assert.deepEqual(l.state.map((item: any) => item.name).sort(), tied ? ['left.weight', 'right.weight'] : ['left.weight']);
      const uses = await query('parameter', 'left.weight', 1);
      const directReads = uses.items.filter((item: any) => item.kind === 'edge' && item.relation === 'argument' && item.source.id === l.id);
      assert.equal(directReads.length, tied ? 2 : 1, 'A tied weight feeds both projections; equal independent weights do not');
      const buffer = await query('buffer', 'bias');
      assert.deepEqual(values(buffer)[0].record.shape, [2]);
      assert.equal((await query('parameter', 'unused.weight')).error.code, 'STATE_NOT_OBSERVED');
      assert.equal((await query('parameter', 'bias')).error.code, 'STATE_NOT_OBSERVED');
      assert.equal(await fs.readFile(file, 'utf8'), text);
    } finally { await client.close(); await server.close(); }
  });
});
