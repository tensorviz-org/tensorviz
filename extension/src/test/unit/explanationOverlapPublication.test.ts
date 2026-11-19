import * as assert from 'node:assert/strict';
import { runExplanationOverlapFixture, type OverlapOrder } from '../explanationOverlapFixture';
import type { LocalBrowserModelExplanationRequest } from '@tensorviz/contracts/protocol';

suite('Actual explanation publication and selection adoption', () => {
  for (const order of ['older-first', 'newer-first', 'cancel-newer'] as OverlapOrder[]) {
    test(`retains per-job completion separately from newest selection/cache authority: ${order}`, async () => {
      const target = { workspaceFolder: '/fixture', file: 'model.py', cls: 'Model', sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 0 };
      const request: LocalBrowserModelExplanationRequest = { type: 'modelExplanationRequest', protocolVersion: 1, workspaceSessionId: 's',
        messageId: 'm', viewId: 'overlap-view', target, action: 'start', requestId: 'A', nodeIds: ['a'] };
      const trace = await runExplanationOverlapFixture([request, { ...request, requestId: 'B' }], order, 'b'.repeat(64));
      assert.deepEqual(trace.entered, ['A', 'B']);
      const older = trace.messages.filter(message => message.type === 'nodeExplanation' && message.state.requestId === 'A');
      assert.equal(older.length, 1, 'The real host publishes verified completion even for a non-authoritative job.');
      if (order !== 'older-first') assert.deepEqual(trace.after, trace.beforeOlder, 'Older output cannot replace selection/citation or cache authority.');
      if (order === 'cancel-newer') {
        assert.equal(trace.after.cached, null);
        assert.deepEqual(trace.aborted, [false, true]);
      } else {
        assert.equal(trace.after.cached?.title, 'B');
        assert.equal(trace.after.selection?.type, 'nodeExplanation');
        if (trace.after.selection?.type === 'nodeExplanation') assert.deepEqual(trace.after.selection.state.explanation, trace.after.cached);
      }
    });
  }
});
