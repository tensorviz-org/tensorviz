import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { CaptureInspector, inspectCapture } from '@tensorviz/graph-host/captureInspection';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { registerMcpCaptureTools } from '@tensorviz/graph-host/mcpCaptures';
import type { CaptureInspectionQuery, ProjectionPayload } from '@tensorviz/contracts/protocol';
import { isCaptureInspectionPage } from '@tensorviz/contracts/protocol';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';
import { canonicalFixture, automaticFixture } from '../../../tests/fixtures/semantic.cjs';

const target = { workspaceFolder: '/ws', file: 'model.py', cls: 'Net',
  documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
function retain(ledger: AttemptLedger, id = 'attempt:0', partial = false) {
  const { capture, plan } = canonicalFixture(id);
  capture.id = `capture:${id}`;
  if (partial) { capture.status = 'partial'; capture.coverage.omissions = ['operation_limit']; }
  const handle = ledger.begin(id, target, 'manual', null);
  handle.resolved({ plan, applicabilityDigest: capture.planDigest });
  const outcome: TraceOutcome = { ok: true, exitCode: 0, envelope: { protocol_version: 1,
    result: { diagnostics: { semantic_capture: capture } }, spans: {}, error: null, worker: {} } };
  handle.captured(outcome); handle.finish('passed'); ledger.projected(target, outcome.envelope);
  return capture;
}
const query = (capture: CaptureSnapshot, fields: Partial<CaptureInspectionQuery> = {}): CaptureInspectionQuery =>
  ({ captureId: capture.id, sourceId: capture.source.id, path: [], offset: 0, limit: 2, ...fields });

suite('canonical capture inspection', () => {
  test('applied-source inspection selects only the latest finished actual capture', () => {
    const ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger), first = retain(ledger, 'first');
    assert.equal(inspector.latest(target)?.capture.id, first.id);
    const newer = ledger.begin('needs-setup', target, 'manual', null);
    assert.equal(inspector.latest(target), undefined);
    newer.finish('refused'); assert.equal(inspector.latest(target), undefined);
    const partial = retain(ledger, 'partial', true);
    assert.equal(inspector.latest(target)?.capture.id, partial.id);
    assert.equal(inspector.latest(target)?.capture.status, 'partial');
    assert.equal(inspector.latest({ ...target, cls: 'other' }), undefined);
    const capped = new AttemptLedger(100); retain(capped);
    assert.equal(new CaptureInspector(capped).latest(target), undefined);
  });

  test('candidate replay retrieves the displayed capture plan instead of the latest attempt setup', () => {
    const ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger), capture = retain(ledger, 'base');
    const newer = canonicalFixture('newer');
    ledger.begin('newer', target, 'manual', null).resolved({ plan: newer.plan, applicabilityDigest: newer.capture.planDigest });
    assert.equal(ledger.executionPlan(target)?.id, 'newer');
    const plan = inspector.retainedPlan(target, capture.id, capture.source.id)!;
    assert.equal(plan.id, 'base'); plan.source.files[0].digest = '0'.repeat(64);
    assert.notEqual(inspector.retainedPlan(target, capture.id, capture.source.id)?.source.files[0].digest, '0'.repeat(64));
    assert.throws(() => inspector.retainedPlan({ ...target, cls: 'other' }, capture.id, capture.source.id), /retained/);
  });

  test('automatic preparation is never relabelled as a replayable resolved plan', () => {
    const ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger), { capture, preparation } = automaticFixture();
    const attempt = ledger.begin(capture.attemptId, target, 'manual', null);
    attempt.prepared({ preparation, applicabilityDigest: capture.preparationDigest });
    attempt.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1, result: { diagnostics: { semantic_capture: capture } },
      worker: {}, spans: {}, error: null } }); attempt.finish('passed');
    assert.equal(inspector.retainedPlan(target, capture.id, capture.source.id), undefined);
    assert.equal(inspector.latest(target)?.capture.id, capture.id);
    assert.equal(inspector.latest(target)?.plan, undefined);
  });

  test('recursive pages reconstruct every argument, literal, state and coverage record exactly', () => {
    const { capture } = canonicalFixture();
    function read(path: Array<string | number>): unknown {
      const first = inspectCapture(capture, query(capture, { path }));
      assert.ok(isCaptureInspectionPage(first), 'the real producer must pass the browser wire gate');
      if (first.node.kind !== 'object' && first.node.kind !== 'array') return first.node.value;
      const result: any = first.node.kind === 'array' ? [] : {};
      let page = first;
      for (;;) {
        for (const item of page.items) Object.defineProperty(result, item.key!, { value: read([...path, item.key!]), enumerable: true, configurable: true, writable: true });
        if (page.nextOffset === null) break;
        page = inspectCapture(capture, query(capture, { path, offset: page.nextOffset }));
        assert.ok(isCaptureInspectionPage(page));
      }
      return result;
    }
    assert.deepEqual(read([]), capture);
    const args = inspectCapture(capture, query(capture, { entity: { kind: 'operation', id: 'add2' }, path: ['arguments', 'args'] }));
    assert.deepEqual(args.items.map(item => item.key), [0, 1], 'repeated operands occupy different paths');
    assert.equal(args.items[0].label, args.items[1].label);
  });

  test('large records and scalar collections remain accessible within the byte ceiling', () => {
    const { capture } = canonicalFixture();
    capture.operations[0].arguments.args = Array.from({ length: 256 }, () => ({ kind: 'str', value: '\u0001'.repeat(4096) }));
    capture.modules[0].paths = Array.from({ length: 256 }, (_, i) => String(i) + '\u0001'.repeat(4000));
    const whole = inspectCapture(capture, query(capture, { entity: { kind: 'operation', id: 'add1' } }));
    assert.ok(whole.items.length > 0, 'oversized operation is navigable as children');
    let offset = 0, total = 0;
    do {
      const page = inspectCapture(capture, query(capture, { path: ['modules', 0, 'paths'], offset, limit: 200 }));
      assert.ok(Buffer.byteLength(JSON.stringify(page)) <= 64 * 1024);
      assert.ok(page.items.length > 0);
      for (const item of page.items) assert.equal(item.value, capture.modules[0].paths[item.key as number]);
      total += page.items.length;
      if (page.nextOffset === null) break;
      assert.ok(page.nextOffset > offset); offset = page.nextOffset;
    } while (total < 300);
    assert.equal(total, 256);
    assert.equal(inspectCapture(capture, query(capture, { path: ['operations', 0, 'arguments', 'args', 255, 'value'] })).node.value, '\u0001'.repeat(4096));
  });

  test('rejects prototype traversal, mistyped indices, foreign references and unbounded queries', () => {
    const { capture } = canonicalFixture();
    for (const path of [['__proto__'], ['operations', '0'], ['operations', -1], ['operations', 20], ['operations', 0, 'id', 'length']]) {
      assert.throws(() => inspectCapture(capture, query(capture, { path })));
    }
    assert.throws(() => inspectCapture(capture, query(capture, { sourceId: 'other' })), /exact capture/);
    assert.throws(() => inspectCapture(capture, query(capture, { entity: { kind: 'input', id: 'other' } })), /boundary/);
    assert.throws(() => inspectCapture(capture, query(capture, { entity: { kind: 'operation', id: 'missing' } })), /entity/);
    assert.throws(() => inspectCapture(capture, query(capture, { limit: 0 })));
    assert.throws(() => inspectCapture(capture, query(capture, { path: Array(128).fill('a'.repeat(128)) })));
  });

  test('ledger scopes retained evidence to the selected model and refuses evicted/legacy/oversize captures', () => {
    const ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger);
    const first = retain(ledger, 'first');
    assert.equal(inspector.inspect({ ...target, documentVersion: 9 }, query(first)).captureId, first.id);
    const second = retain(ledger, 'second');
    assert.throws(() => inspector.inspect(target, query(first)), /no longer retained/);
    assert.throws(() => inspector.inspect({ ...target, cls: 'Other' }, query(second)), /no longer retained/);
    assert.equal(inspector.list({ ...target, cls: 'Other' }).captures.length, 0);
    const capped = new AttemptLedger(100), cappedCapture = retain(capped);
    assert.throws(() => new CaptureInspector(capped).inspect(target, query(cappedCapture)), /retention limit/);
    const handle = ledger.begin('legacy', target, 'manual', null);
    handle.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1, result: { graph: {} }, spans: {}, error: null, worker: {} } });
    handle.finish('passed');
    const legacy = inspector.list(target).lastPartialCaptureId!;
    assert.throws(() => inspector.inspect(target, { ...query(second), captureId: legacy }), /legacy/);
  });
});

suite('MCP retained canonical capture transport', () => {
  let client: Client, server: McpServer, model: McpModel, graph: McpGraphService, ledger: AttemptLedger, capture: CaptureSnapshot;
  setup(async () => {
    ledger = new AttemptLedger(); capture = retain(ledger);
    model = { target: { ...target }, current: true, projection: { document: { semantic: { nodes: [], edges: [] } }, spanmap: {}, traced: true } as unknown as ProjectionPayload };
    graph = new McpGraphService('host', () => model);
    server = new McpServer({ name: 'fixture', version: '1' }); graph.register(server);
    registerMcpCaptureTools(server, graph, new CaptureInspector(ledger));
    client = new Client({ name: 'reader', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
  });
  teardown(async () => { await client.close(); await server.close(); });
  async function call(name: string, args: object = {}) {
    return (await client.callTool({ name, arguments: { modelId: graph.reference(model).modelId, ...args } })).structuredContent as any;
  }
  test('manual and failed latest attempts remain separate; exact retained records survive a stale source graph', async () => {
    ledger.begin('timeout', target, 'mcp', null).finish('failed');
    model.current = false; model.target.documentVersion++;
    const listed = await call('list_captures');
    assert.equal(listed.current, false);
    assert.equal(listed.latestAttempt.id, 'timeout');
    assert.equal(listed.displayedCaptureId, capture.id);
    const result = await call('inspect_capture', query(capture, { entity: { kind: 'operation', id: 'add1' }, path: ['arguments', 'args', 1] }));
    assert.deepEqual(result.items.map((item: any) => [item.key, item.value]), [['kind', 'int'], ['value', '1']]);
    assert.equal((await call('get_graph', graph.reference(model))).error.code, 'STALE_TARGET', 'live topology/focus gates remain strict');
  });
  test('cross-model requests and foreign source references are refused through actual MCP', async () => {
    assert.equal((await call('list_captures', { modelId: 'other' })).error.code, 'STALE_TARGET');
    assert.equal((await call('inspect_capture', { ...query(capture), sourceId: 'other' })).error.code, 'SOURCE_MISMATCH');
    model.target.cls = 'Other';
    assert.equal((await call('list_captures')).captures.length, 0);
    assert.equal((await call('inspect_capture', query(capture))).error.code, 'UNKNOWN_CAPTURE');
  });
});
