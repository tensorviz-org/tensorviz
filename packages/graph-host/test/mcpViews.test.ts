import * as assert from 'node:assert/strict';
import { McpViewRegistry } from '@tensorviz/graph-host/mcpViews';
import { GraphToolError, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import type { LocalBrowserMessage } from '@tensorviz/contracts/protocol';

suite('MCP view ownership', () => {
  const target = { workspaceFolder: '/work', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
  let registry: McpViewRegistry, current: boolean, published: any[];
  const message = (value: object) => ({ protocolVersion: 1, workspaceSessionId: 's1', messageId: 'm1', ...value } as LocalBrowserMessage);
  setup(() => {
    current = true; published = [];
    registry = new McpViewRegistry(() => ({ target, current } as McpModel), request => { published.push(request); return request; });
    registry.handle(message({ type: 'viewState', viewId: 'view-1', connected: true }));
    registry.handle(message({ type: 'viewState', viewId: 'view-2', connected: true }));
  });
  teardown(() => registry.clear());
  test('selection remains bound to its view and revision', () => {
    registry.handle(message({ type: 'viewSelection', viewId: 'view-1', target, nodeIds: ['fc'], edgeIds: [] }));
    assert.deepEqual(registry.selection('view-1', target).nodeIds, ['fc']);
    assert.throws(() => registry.selection('view-2', target), /current graph/);
    assert.throws(() => registry.selection('view-1', { ...target, backendRevision: 2 }), /current graph/);
    registry.handle(message({ type: 'viewState', viewId: 'view-1', connected: false }));
    assert.throws(() => registry.selection('view-1', target), /connected view/);
  });
  test('focus is confirmed only by the exact view and request', async () => {
    const done = registry.focus('client-1', { viewId: 'view-1', target, nodeIds: ['fc'], edgeIds: [] }, new AbortController().signal);
    const request = published[0];
    registry.handle(message({ type: 'viewFocusResult', viewId: 'view-2', target, requestId: request.requestId, ok: true, detail: 'Focused.' }));
    registry.handle(message({ type: 'viewFocusResult', viewId: 'view-1', target, requestId: request.requestId, ok: true, detail: 'Focused.' }));
    assert.equal((await done).focused, true);
    assert.deepEqual(registry.selection('view-1', target).nodeIds, ['fc']);
  });
  test('a late receipt cannot claim success after the source changes', async () => {
    const done = registry.focus('client-1', { viewId: 'view-1', target, nodeIds: ['fc'], edgeIds: [] }, new AbortController().signal);
    current = false;
    registry.handle(message({ type: 'viewFocusResult', viewId: 'view-1', target, requestId: published[0].requestId, ok: true, detail: 'Focused.' }));
    await assert.rejects(done, (error: GraphToolError) => error.code === 'STALE_TARGET');
  });
  test('disconnecting an MCP client cancels only that client’s requests', async () => {
    const a = registry.focus('a', { viewId: 'view-1', target, nodeIds: ['fc'], edgeIds: [] }, new AbortController().signal);
    const b = registry.focus('b', { viewId: 'view-2', target, nodeIds: ['fc'], edgeIds: [] }, new AbortController().signal);
    registry.cancelOwner('a');
    await assert.rejects(a, (error: GraphToolError) => error.code === 'CANCELLED');
    registry.handle(message({ type: 'viewFocusResult', viewId: 'view-2', target, requestId: published[1].requestId, ok: true, detail: 'Focused.' }));
    assert.equal((await b).focused, true);
  });
});
