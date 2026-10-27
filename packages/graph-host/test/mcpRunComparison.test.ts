import * as assert from 'node:assert/strict';
import { compareRetainedRuns, type ComparableRun } from '@tensorviz/graph-host/mcpRunComparison';

const copy = <T>(value: T): T => JSON.parse(JSON.stringify(value));
function fixture(): ComparableRun {
  return { runId: 'baseline', status: 'passed', graph: { target: { workspaceFolder: '/project', file: 'net.py', cls: 'Net' },
    canonicalWorkspace: '/project', traced: true, truncated: false,
    nodes: ['input_node', 'fc', 'output_node'].map(nodeId => ({ nodeId, modulePath: nodeId === 'fc' ? 'fc' : null,
      operation: nodeId === 'fc' ? 'Linear' : nodeId, parameters: nodeId === 'fc' ? { in_features: 4, out_features: 2 } : {},
      shapes: { input: [1, 4], output: [1, 2], provenance: 'observed' }, repeatCount: 1 })),
    edges: [{ edgeId: 'in', sourceNodeId: 'input_node', targetNodeId: 'fc' },
      { edgeId: 'out', sourceNodeId: 'fc', targetNodeId: 'output_node' }] },
    evidence: { setup: { inputs: { digest: 'input' }, state: { digest: 'weights' }, mode: 'eval', moduleModes: { fc: false },
      gradEnabled: false, environment: { device: 'cpu', torch: '2.7.1' }, rngDigest: 'rng', seed: 0, setup: { constructorArgs: {} } },
      execution: { truncated: [], coverage: [{ modulePath: 'fc', reached: true, completedCalls: 1 }],
        moduleCalls: [{ modulePath: 'fc', invocation: 1, completedIndex: 0, status: 'completed', inputs: [{ shape: [1, 4] }],
          outputs: [{ shape: [1, 2] }], values: { digest: 'fc-output' } }] } } };
}
function compare(a: ComparableRun, b: ComparableRun, section = 'summary', offset = 0, limit = 50): any {
  return compareRetainedRuns(a, b, section, offset, limit, 1e-6, 1e-5);
}

suite('retained run architecture and observation comparison', () => {
  test('matches actual module paths and excludes layout and annotations', () => {
    const a = fixture(), b = copy(a); b.runId = 'candidate';
    const nodes = b.graph.nodes as any[]; nodes[1].nodeId = 'new-fc-id'; nodes[1].position = { x: 99, y: 30 }; nodes[1].notes = ['review'];
    (b.graph.edges as any[])[0].targetNodeId = 'new-fc-id'; (b.graph.edges as any[])[1].sourceNodeId = 'new-fc-id';
    const result = compare(a, b);
    assert.equal(result.matchedNodeCount, 3); assert.equal(result.nodeChangeCount, 0); assert.equal(result.connectionChangeCount, 0);
    assert.equal(result.setup.recordedControlsMatch, true); assert.equal(result.outputs.allClose, null, 'no captured values cannot certify numerical equivalence');
  });
  test('reports parameters, dimensions, repeats and connection multiplicities', () => {
    const a = fixture(), b = copy(a), fc = (b.graph.nodes as any[])[1];
    fc.parameters.out_features = 3; fc.shapes.output = [1, 3]; fc.repeatCount = 2;
    (b.graph.edges as any[]).push(copy((b.graph.edges as any[])[0]));
    const result = compare(a, b, 'nodes');
    assert.deepEqual(result.changes.items[0].fields.map((f: any) => f.field), ['parameters', 'repeatCount']);
    assert.equal(result.dimensions.items[0].fields[0].field, 'output');
    const edges = compare(a, b, 'connections'); assert.equal(edges.count, 1);
    assert.equal(edges.items[0].baselineCount, 1); assert.equal(edges.items[0].candidateCount, 2);
  });
  test('ambiguous identities remain uncertain and dangling edges cannot silently match', () => {
    const a = fixture(), b = copy(a); b.runId = 'candidate';
    for (const run of [a, b]) {
      (run.graph.nodes as any[])[1].modulePath = null;
      (run.graph.edges as any[]).push({ sourceNodeId: 'missing', targetNodeId: 'output_node' });
    }
    const result = compare(a, b, 'nodes');
    assert.equal(result.uncertainMatches.count, 1); assert.ok(result.changes.items.every((c: any) => c.uncertainIdentity));
    const edges = compare(a, b, 'connections'); assert.ok(edges.items.every((e: any) => e.uncertainIdentity));
  });
  test('uses canonical source identity when module identity is unavailable', () => {
    const a = fixture(), b = copy(a);
    for (const run of [a, b]) { const node = (run.graph.nodes as any[])[1]; node.modulePath = null;
      node.source = { file: run === a ? '/project/net.py' : 'net.py', digest: 'same-span' }; }
    assert.equal(compare(a, b).nodeChangeCount, 0);
    (b.graph.nodes as any[])[1].source.digest = 'new-source';
    assert.equal(compare(a, b).uncertainMatchCount, 1);
  });
  test('retains baseline-only calls and distinguishes absent from truncated observations', () => {
    const a = fixture(), b = copy(a);
    (b.evidence!.execution as any).moduleCalls = [];
    let result = compare(a, b, 'observations');
    assert.equal(result.changes.items[0].kind, 'not_observed_in_candidate'); assert.equal(result.changes.items[0].absenceConfirmed, true);
    assert.equal(result.firstObservedDivergence, null); assert.equal(result.baselineOnlyCalls, 1);
    (b.evidence!.execution as any).truncated = ['module_calls'];
    result = compare(a, b, 'observations'); assert.equal(result.changes.items[0].absenceConfirmed, false);
    assert.deepEqual(result.candidateTruncation, ['module_calls']);
  });
  test('first observed divergence follows candidate completion and never implies tolerance failure', () => {
    const a = fixture(), b = copy(a), execution = b.evidence!.execution as any;
    execution.moduleCalls[0].values.digest = 'changed-by-one-bit';
    const result = compare(a, b);
    assert.equal(result.firstObservedDivergence.modulePath, 'fc'); assert.equal(result.firstObservedDivergence.kind, 'value_fingerprint');
    assert.equal(result.outputs.allClose, null);
    delete execution.moduleCalls[0].values;
    assert.equal(compare(a, b).callsWithoutComparableValues, 1);
  });
  test('setup differences are bounded and paginated without copying full module modes into summaries', () => {
    const a = fixture(), b = copy(a);
    (b.evidence!.setup as any).state.digest = 'new-weights'; (b.evidence!.setup as any).environment.device = 'cuda:0';
    const modes = Object.fromEntries(Array.from({ length: 10000 }, (_, i) => [`module${i}`, true]));
    (b.evidence!.setup as any).moduleModes = modes;
    const result = compare(a, b); assert.equal(result.setup.weights.status, 'different');
    assert.equal(result.setup.environment.status, 'different'); assert.equal(result.setup.recordedControlsMatch, false);
    assert.ok(JSON.stringify(result).length < 10000);
    const first = compare(a, b, 'setup', 0, 2), second = compare(a, b, 'setup', 2, 2);
    assert.equal(first.items.length, 2); assert.equal(first.nextOffset, 2); assert.ok(first.count > 10000);
    assert.notDeepEqual(first.items, second.items);
  });
  test('missing execution/setup evidence remains unknown', () => {
    const a = fixture(), b = copy(a); delete b.evidence;
    const result = compare(a, b); assert.equal(result.setup.inputs.status, 'unknown'); assert.equal(result.setup.recordedControlsMatch, false);
    assert.equal(result.observationEvidence.available.candidate, false);
  });
  test('different training losses remain an explicit setup difference', () => {
    const a = fixture(), b = copy(a);
    (a.evidence!.setup as any).training = { lossDigest: 'loss-one', clearGradients: true, optimizerUpdate: false };
    (b.evidence!.setup as any).training = { lossDigest: 'loss-two', clearGradients: true, optimizerUpdate: false };
    const result = compare(a, b);
    assert.equal(result.setup.diagnostic.status, 'different'); assert.equal(result.setup.recordedControlsMatch, false);
    assert.ok(compare(a, b, 'setup').items.some((row: any) => row.category === 'diagnostic'));
  });

});
