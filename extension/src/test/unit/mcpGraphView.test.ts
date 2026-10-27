import * as assert from 'node:assert/strict';
import { McpGraphView, registerMcpGraphView } from '@tensorviz/graph-host/mcpGraphView';
import { GraphToolError, McpGraphService, type McpModel, type ModelRef } from '@tensorviz/graph-host/mcpGraph';
import { McpViewRegistry } from '@tensorviz/graph-host/mcpViews';
import { captureTools } from '../../mcpHost';
import type { LocalBrowserMessage } from '@tensorviz/contracts/protocol';

suite('MCP browser graph handoff', () => {
  const signal = () => new AbortController().signal;
  const message = (data: object) => ({ protocolVersion: 1, workspaceSessionId: 's', messageId: 'm', ...data } as LocalBrowserMessage);
  function rig() {
    const model = { target: { workspaceFolder: '/w', file: 'model.py', cls: 'Net', documentVersion: 1,
      sourceDigest: 'a'.repeat(64), backendRevision: 1 }, current: true,
      projection: { projectionDigest: 'p', document: { semantic: { nodes: [], edges: [] } } } } as unknown as McpModel;
    const graph = new McpGraphService('host', () => model);
    const views = new McpViewRegistry(() => model, () => true);
    let count = 0, active = true;
    const host = { mcpViews: views, browserSessionCurrent: (id: string) => active && id === 's',
      prepareBrowserView: async () => ({ workspaceSessionId: 's', handoffId: `h-${++count}`,
        expiresAt: new Date(Date.now() + 120000).toISOString(), pageUrl: 'http://localhost/#private' }) };
    const service = new McpGraphView(graph, () => host);
    const ref = graph.reference(model);
    const poll = (handoffId: string, owner = 'a', reference: ModelRef = ref) => service.invoke(owner, { ...reference, handoffId }, signal());
    return { model, graph, views, host, service, ref, poll, get count() { return count; }, end: () => { active = false; } };
  }

  test('returns a link once and polls exact browser display, never the visible native canvas', async () => {
    const r = rig();
    const tools = captureTools(server => registerMcpGraphView(server, r.service, 'a'));
    const reply = await tools.get('open_graph_view')!.handler(r.ref, { signal: signal() });
    const first = reply.structuredContent as any;
    assert.equal(first.status, 'awaiting_browser'); assert.equal(first.displayed, false); assert.ok(first.url);
    r.views.connection('native', true);
    r.views.handle(message({ type: 'viewSelection', viewId: 'native', target: r.model.target, nodeIds: [], edgeIds: [] }));
    assert.equal((await r.poll(first.handoffId)).displayed, false);
    r.views.handle(message({ type: 'viewState', viewId: 'browser', handoffId: first.handoffId, connected: true }));
    assert.equal((await r.poll(first.handoffId)).status, 'connecting');
    r.views.handle(message({ type: 'viewSelection', viewId: 'browser', target: { ...r.model.target, backendRevision: 0 }, nodeIds: [], edgeIds: [] }));
    assert.equal((await r.poll(first.handoffId)).displayed, false);
    r.views.handle(message({ type: 'viewSelection', viewId: 'browser', target: r.model.target, nodeIds: [], edgeIds: [] }));
    const ready = await r.poll(first.handoffId);
    assert.equal(ready.displayed, true); assert.equal(ready.viewId, 'browser'); assert.equal(ready.url, undefined);
    assert.equal(r.count, 1, 'polling does not mint or execute');
    r.views.connection('browser', false);
    assert.equal((await r.poll(first.handoffId)).displayed, false);
    r.views.handle(message({ type: 'viewState', viewId: 'reconnected', handoffId: first.handoffId, connected: true }));
    assert.equal((await r.poll(first.handoffId)).viewId, 'reconnected');
  });

  test('owners cannot poll each other, and ended owners cannot reuse old handoffs', async () => {
    const r = rig(), first = await r.service.invoke('a', r.ref, signal());
    await assert.rejects(r.poll(first.handoffId as string, 'b'), /connection’s handoffId/);
    r.service.cancelOwner('a');
    await assert.rejects(r.poll(first.handoffId as string), /connection’s handoffId/);
  });

  test('rejects stale graph references and invalidated sessions', async () => {
    const r = rig(), first = await r.service.invoke('a', r.ref, signal());
    r.model.target.backendRevision++;
    await assert.rejects(r.poll(first.handoffId as string), (e: GraphToolError) => e.code === 'STALE_TARGET');
    r.model.target.backendRevision--;
    r.end();
    await assert.rejects(r.poll(first.handoffId as string), (e: GraphToolError) => e.code === 'STALE_SESSION');
  });

  test('never publishes a link after cancellation or a model change during preparation', async () => {
    for (const cancel of [true, false]) {
      const r = rig();
      const prepare = r.host.prepareBrowserView;
      r.host.prepareBrowserView = async () => {
        if (cancel) r.service.cancelOwner('a'); else r.model.target.documentVersion++;
        return prepare();
      };
      await assert.rejects(r.service.invoke('a', r.ref, signal()), (e: GraphToolError) => e.code === (cancel ? 'CANCELLED' : 'STALE_TARGET'));
    }
  });

  test('expired links cannot claim display; an already connected page outlives its link', async () => {
    const r = rig(), prepare = r.host.prepareBrowserView;
    r.host.prepareBrowserView = async () => ({ ...await prepare(), expiresAt: new Date(0).toISOString() });
    await assert.rejects(r.service.invoke('a', r.ref, signal()), (e: GraphToolError) => e.code === 'VIEW_LINK_EXPIRED');
    r.views.handle(message({ type: 'viewState', viewId: 'browser', handoffId: 'h-1', connected: true }));
    r.views.handle(message({ type: 'viewSelection', viewId: 'browser', target: r.model.target, nodeIds: [], edgeIds: [] }));
    assert.equal((await r.poll('h-1')).displayed, true);
  });
});
