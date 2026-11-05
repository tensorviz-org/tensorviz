import * as assert from 'node:assert/strict';
import { isModelExplanationState, isViewExplanation, type ViewExplanation,
  type NodeExplanationCitation } from '@tensorviz/contracts/nodeExplanation';
import { isLocalBrowserMessage } from '@tensorviz/contracts/protocol';

const citation = (id = 'source'): NodeExplanationCitation => ({ id, file: 'layers/audio.py',
  sourceDigest: 'a'.repeat(64), startLine: 10, endLine: 30 });
const view = (): ViewExplanation => ({
  summary: 'Decode audio features, using the sample-rate bucket to condition decoder blocks.',
  nodeIds: ['features', 'sample-rate', 'decoder', 'waveform'],
  connections: [{ sourceNodeId: 'features', targetNodeId: 'decoder', label: 'Learned audio features',
    detail: 'The decoder receives these features as x.', citations: [citation()] }],
  callouts: [{ title: 'Why this branch exists', body: 'The bucket selects conditioning applied before decoder blocks.',
    nodeIds: ['sample-rate', 'decoder'], citations: [citation()] }],
});
const state = (value = view()) => ({ requestId: 'request', revision: 'revision', status: 'ready',
  completed: 4, total: 4, view: value });
const message = (value: unknown) => ({ type: 'modelExplanation', protocolVersion: 1, workspaceSessionId: 'session',
  messageId: 'message', viewId: 'view', target: { workspaceFolder: '/workspace', file: 'model.py', cls: 'Model',
    documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 }, state: value });

suite('visible-view explanation contract', () => {
  test('keeps the requested scope and partial results through each lifecycle state', () => {
    assert.ok(isViewExplanation(view()));
    assert.ok(isViewExplanation({ ...view(), connections: [], callouts: [] }));
    for (const status of ['loading', 'ready', 'unavailable', 'cancelled']) {
      const result = { ...state(), status, completed: status === 'ready' ? 4 : 1 };
      assert.ok(isModelExplanationState(result));
      assert.ok(isLocalBrowserMessage(message(result), 'session'));
    }
    assert.equal(isLocalBrowserMessage(message(state()), 'other-session'), false);
    assert.equal(isModelExplanationState({ ...state(), total: 5 }), false);
    const { view: _view, ...legacy } = state();
    assert.ok(isModelExplanationState(legacy));
  });

  test('rejects hidden or duplicate references and duplicate directed connection labels', () => {
    const value = view(), connection = value.connections[0], callout = value.callouts[0];
    for (const invalid of [
      { ...value, nodeIds: [] }, { ...value, nodeIds: ['features', 'features'] },
      { ...value, nodeIds: [''] }, { ...value, nodeIds: ['x'.repeat(1025)] },
      { ...value, connections: [{ ...connection, sourceNodeId: 'hidden' }] },
      { ...value, connections: [{ ...connection, targetNodeId: 'hidden' }] },
      { ...value, connections: [connection, connection] },
      { ...value, callouts: [{ ...callout, nodeIds: ['hidden'] }] },
      { ...value, callouts: [{ ...callout, nodeIds: [] }] },
      { ...value, callouts: [{ ...callout, nodeIds: ['decoder', 'decoder'] }] },
    ]) assert.equal(isViewExplanation(invalid), false, JSON.stringify(invalid));
    // Labels may describe both directions only when the host supplied both actual edges.
    assert.ok(isViewExplanation({ ...value, connections: [connection,
      { ...connection, sourceNodeId: connection.targetNodeId, targetNodeId: connection.sourceNodeId }] }));
  });

  test('bounds the shared scope, callouts, and directed connection collection', () => {
    const value = view();
    const nodeIds = Array.from({ length: 128 }, (_, index) => `node-${index}`);
    const connections = nodeIds.map((id, index) => ({ ...value.connections[0], sourceNodeId: id,
      targetNodeId: nodeIds[(index + 1) % nodeIds.length] }));
    const callout = { ...value.callouts[0], nodeIds };
    assert.ok(isViewExplanation({ ...value, nodeIds, connections, callouts: Array(6).fill(callout) }));
    assert.equal(isViewExplanation({ ...value, nodeIds: [...nodeIds, 'node-128'] }), false);
    assert.equal(isViewExplanation({ ...value, connections: [...connections, connections[0]] }), false);
    assert.equal(isViewExplanation({ ...value, callouts: Array(7).fill(value.callouts[0]) }), false);
  });

  test('requires source citations for connection meanings and linked commentary', () => {
    for (const section of ['connections', 'callouts'] as const) {
      for (const citations of [[], [citation(), citation()], [citation('x'.repeat(129))],
        [{ ...citation(), file: '/private/model.py' }], [{ ...citation(), startLine: 0 }],
        [{ ...citation(), execute: true }], Array.from({ length: 9 }, (_, index) => citation(`source-${index}`))]) {
        const value = view();
        assert.equal(isViewExplanation({ ...value, [section]: [{ ...value[section][0], citations }] }), false);
      }
      const value = view();
      assert.ok(isViewExplanation({ ...value, [section]: [{ ...value[section][0],
        citations: Array.from({ length: 8 }, (_, index) => citation(`source-${index}`)) }] }));
    }
  });

  test('keeps generated text bounded and plain without expanding model authority', () => {
    for (const [field, maximum] of [['summary', 640], ['label', 120], ['detail', 320], ['title', 80], ['body', 480]] as const) {
      const update = (text: unknown) => {
        const value = view();
        if (field === 'summary') return { ...value, summary: text };
        if (field === 'label' || field === 'detail') return { ...value,
          connections: [{ ...value.connections[0], [field]: text }] };
        return { ...value, callouts: [{ ...value.callouts[0], [field]: text }] };
      };
      assert.ok(isViewExplanation(update('x'.repeat(maximum))));
      assert.ok(isViewExplanation(update('🙂'.repeat(maximum / 2))));
      for (const text of ['', ' ', ' padded', 'line\nbreak', 'text\u0000', 'x'.repeat(maximum + 1),
        '🙂'.repeat(maximum / 2 + 1), null]) assert.equal(isViewExplanation(update(text)), false, `${field}: ${JSON.stringify(text)}`);
    }
    const value = view();
    for (const invalid of [{ ...value, execute: true },
      { ...value, connections: [{ ...value.connections[0], source: 'invented' }] },
      { ...value, callouts: [{ ...value.callouts[0], command: 'run model' }] },
      { ...value, connections: undefined }, { ...value, callouts: undefined }]) {
      assert.equal(isViewExplanation(invalid), false);
    }
  });
});
