import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { graphDebugContext } from '@tensorviz/graph-host/graphDebug';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';

const fixture = (): CaptureSnapshot => JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/portable-capture.json'), 'utf8')).capture;
const projection = (capture: CaptureSnapshot): ProjectionPayload => ({ graphId: 'g', revision: 1, cls: 'Net', file: 'model.py',
  sourceDigest: capture.source.id, traced: true, greyed: false, viewRestored: false, spanmap: {},
  document: { semantic: { nodes: capture.invocations.map(call => ({ node_id: `node:${call.id}`,
    metadata: { captureRef: { kind: 'invocation', semanticId: call.id } } })) } },
});
suite('previous-run debugging context', () => {
  test('retains one prior document for source refresh and partial capture only', () => {
    const before = fixture(), previous = { capture: before, projection: projection(before) };
    const after = fixture(); after.id = 'next';
    assert.equal(graphDebugContext(previous, projection(after)).previousDocument, previous.projection.document);
    assert.equal(graphDebugContext(previous, projection(after), after).previousDocument, undefined);
    after.invocations[0].status = 'failed';
    assert.equal(graphDebugContext(previous, projection(after), after).previousDocument, previous.projection.document);
  });
  test('changed initializer values are observed changes, independent of failure location', () => {
    const before = fixture(), after = fixture();
    after.source.id = 'changed-source';
    const entry = { phase: 'initializer_entry' as const, status: 'observed' as const,
      positional: [{ name: 'width', value: { kind: 'int' as const, value: '8' } }] };
    before.modules[0].initializerArguments = entry;
    after.modules[0].initializerArguments = { ...entry, positional: [{ name: 'width', value: { kind: 'int', value: '4' } }] };
    const result = graphDebugContext({ capture: before, projection: projection(before) }, projection(after), after);
    assert.deepEqual(result.changed, [{ nodeId: `node:${after.invocations[0].id}`, details: ['width: 8 → 4'] }]);
    after.modules[0].initializerArguments = { phase: 'initializer_entry', status: 'unavailable', reason: 'byte_limit' };
    assert.deepEqual(graphDebugContext({ capture: before, projection: projection(before) }, projection(after), after).changed, []);
  });
  test('numeric registrations and changed tensor shapes do not invent parameter matches', () => {
    const before = fixture(), after = fixture();
    after.source.id = 'changed-source';
    before.modules[0].paths = after.modules[0].paths = ['blocks.0'];
    after.values[0].shape = [99, 1];
    assert.deepEqual(graphDebugContext({ capture: before, projection: projection(before) }, projection(after), after).changed, []);
  });
  const parameterCaptures = () => {
    const before = fixture();
    before.modules[0].paths = ['up'];
    before.modules[0].initializerArguments = null;
    before.values.push({ ...before.values[0], id: 'weight-before', origin: 'state', producer: null,
      state: [{ kind: 'parameter', name: 'up.weight' }], shape: [16, 8] });
    const after: CaptureSnapshot = JSON.parse(JSON.stringify(before));
    after.source.id = 'changed-source';
    after.values[after.values.length - 1].id = 'weight-after';
    after.values[after.values.length - 1].shape = [8, 8];
    return { before, after };
  };
  test('successful captures with the baseline source digest have no parameter badges', () => {
    const { before, after } = parameterCaptures();
    after.source.id = before.source.id;
    const previous = { capture: before, projection: projection(before) };
    const unchanged = graphDebugContext(previous, projection(after), after);
    assert.deepEqual(unchanged.changed, []);
    assert.ok(unchanged.nodePairs.length);
    after.source.id = 'changed-source';
    assert.deepEqual(graphDebugContext(previous, projection(after), after).changed,
      [{ nodeId: `node:${after.invocations[0].id}`, details: ['weight shape: [16, 8] → [8, 8]'] }]);
  });
  test('named parameter shapes identify an upstream change without constructor observations', () => {
    const { before, after } = parameterCaptures();
    after.status = 'partial';
    const result = graphDebugContext({ capture: before, projection: projection(before) }, projection(after), after);
    assert.deepEqual(result.changed, [{ nodeId: `node:${after.invocations[0].id}`, details: ['weight shape: [16, 8] → [8, 8]'] }]);
    assert.ok(result.previousDocument);
  });
  test('unknown, mutated, ambiguous and numerically registered state cannot mark a parameter change', () => {
    for (const alter of [
      (c: CaptureSnapshot) => { c.values[c.values.length - 1].shape = null; },
      (c: CaptureSnapshot) => { c.values[c.values.length - 1].logicalVersion = 1; },
      (c: CaptureSnapshot) => { c.values.push({ ...c.values[c.values.length - 1], id: 'ambiguous-weight' }); },
      (c: CaptureSnapshot) => { c.modules[0].paths = ['blocks.0']; c.values[c.values.length - 1].state[0].name = 'blocks.0.weight'; },
    ]) {
      const { before, after } = parameterCaptures();
      alter(after);
      assert.deepEqual(graphDebugContext({ capture: before, projection: projection(before) }, projection(after), after).changed, []);
    }
  });
});
