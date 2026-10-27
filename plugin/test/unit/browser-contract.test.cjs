const { test } = require('node:test');
const assert = require('node:assert/strict');
const { StandaloneHost } = require('../../out/host/host.cjs');
const { ApprovalGate } = require('../../out/host/approval.cjs');
const { isLocalBrowserMessage, LOCAL_BROWSER_PROTOCOL_VERSION } = require('../../out/host/protocol.cjs');

test('browser capture replies and invalidation preserve the shared wire and attempt contracts', async () => {
  const host = new StandaloneHost('/unused', '/unused', new ApprovalGate(async () => false, '/unused'));
  const target = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Tiny', documentVersion: 0,
    sourceDigest: 'a'.repeat(64), backendRevision: 1 };
  const published = [];
  let onMessage;
  const transport = {
    status: { phase: 'connected' },
    publish(message) {
      const full = { ...message, protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION, workspaceSessionId: 'session', messageId: 'message' };
      if (['activeTarget', 'projection', 'syncState', 'graphOpening'].includes(message.type)) full.hostSequence = 1;
      assert.ok(isLocalBrowserMessage(full, 'session'), `invalid browser message: ${message.type}`);
      published.push(message);
    },
    onMessage(listener) { onMessage = listener; }, onStatusChange() {}, dispose() {},
  };
  host.backend.connect = async () => { host.backend.transport = transport; return transport; };
  host.model = { target, current: true, projection: { traced: true, sourceDigest: target.sourceDigest, spanmap: {}, document: {} } };
  try {
    await host.connectBrowser(new AbortController().signal);
    host.captures.inspect = () => ({ captureId: 'capture', sourceId: 'source', path: [],
      node: { key: null, kind: 'object', count: 0 }, items: [], total: 0, nextOffset: null });
    onMessage({ type: 'captureInspectRequest', target, requestId: 'read', query: { captureId: 'capture', sourceId: 'source', path: [] } });
    assert.equal(published.at(-1).page.captureId, 'capture', 'capture details must answer without disconnecting the view');
    const attempt = host.ledger.begin('attempt', target, 'mcp', {});
    attempt.finish('passed');
    const ready = { id: 'opening', target, phase: 'ready', hasCapture: true, detail: 'Captured.' };
    host.publishOpening(ready);
    host.publishOpening({ ...ready, phase: 'cancelled', hasCapture: false, cancellationReason: 'source_changed', detail: 'Source changed.' });
    assert.equal(published.filter(message => message.type === 'validationState').length, 1,
      'source invalidation must not turn the earlier successful attempt into a cancelled run');
    host.opening.sourceOnly({ ...target, sourceDigest: 'b'.repeat(64), backendRevision: 2 });
    host.selected = { root: target.workspaceFolder, file: target.file, cls: target.cls, environment: { interpreter: '/python' } };
    const openings = [], setups = [];
    host.openGraph.invoke = (_owner, args) => openings.push(args);
    host.opening.action = async (...args) => { setups.push(args); return true; };
    const action = (asked, kind) => onMessage({ type: 'graphSetupRequest', target: asked,
      openingId: 'earlier-opening', requestId: kind, action: { kind } });
    action(target, 'retry');
    assert.equal(openings.length, 1, 'Refresh rereads the same model even when the frame has an earlier source/revision');
    assert.equal(openings[0].file, target.file);
    action(target, 'change_inputs');
    assert.equal(setups.length, 0, 'input edits still require the exact current identity');
    for (const field of ['workspaceFolder', 'file', 'cls']) action({ ...target, [field]: 'another-model' }, 'retry');
    assert.equal(openings.length, 1, 'an old frame for another model cannot refresh the selected model');
  } finally { host.dispose(); }
});
