import * as assert from 'node:assert/strict';
import { isModelCatalog, isModelCatalogEntry, MODEL_CATALOG_LIMIT } from '@tensorviz/contracts/modelCatalog';
import { isLocalBrowserMessage } from '@tensorviz/contracts/protocol';

const entry = { id: 'model-a', cls: 'AudioVAEV2', file: 'layers/audio_vae_v2.py', evidence: 'source', opened: false };
const catalog = { status: 'ready', models: [entry], activeId: entry.id };
const target = { workspaceFolder: '/workspace', file: entry.file, cls: entry.cls, documentVersion: 1,
  sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const envelope = { protocolVersion: 1, workspaceSessionId: 'session', messageId: 'message', viewId: 'view', target, requestId: 'request' };

suite('project model catalogue contract', () => {
  test('bounds real source/capture evidence and host-issued selection identities', () => {
    assert.ok(isModelCatalog(catalog));
    for (const status of ['loading', 'ready', 'unavailable']) assert.ok(isModelCatalog({ ...catalog, status }));
    assert.ok(isModelCatalogEntry({ ...entry, evidence: 'captured', opened: true }));
    assert.ok(isModelCatalog({ status: 'ready', models: [] }));
    assert.ok(isModelCatalog({ ...catalog, detail: 'x'.repeat(320), truncated: true }));
    assert.ok(isModelCatalog({ status: 'ready', models: Array.from({ length: MODEL_CATALOG_LIMIT }, (_, index) => ({ ...entry, id: `${index}` })) }));
    for (const update of [{ models: [entry, entry] }, { activeId: 'unknown' }, { status: 'captured' }, { status: { toString: () => 'ready' } },
      { detail: 'x'.repeat(321) }, { truncated: 1 }, { models: Array(MODEL_CATALOG_LIMIT + 1).fill(entry) }, { command: 'execute' }]) {
      assert.equal(isModelCatalog({ ...catalog, ...update }), false, JSON.stringify(update).slice(0, 200));
    }
    for (const update of [{ id: '' }, { id: 'x'.repeat(129) }, { cls: '' }, { cls: 'x'.repeat(513) }, { file: 'x'.repeat(4097) },
      { file: '/etc/passwd' }, { file: '../model.py' }, { file: 'a/../model.py' }, { file: 'a//model.py' }, { file: './model.py' },
      { file: 'file:///model.py' }, { file: 'C:\\model.py' }, { file: 'model\n.py' }, { opened: 'true' }, { evidence: 'inferred' }, { uri: 'open this' }]) {
      assert.equal(isModelCatalogEntry({ ...entry, ...update }), false, JSON.stringify(update));
    }
  });

  test('list and select requests are exact, targeted and correlated without executable paths', () => {
    const request = { ...envelope, type: 'modelCatalogRequest', action: 'list' };
    assert.ok(isLocalBrowserMessage(request, 'session'));
    assert.ok(isLocalBrowserMessage({ ...request, action: 'select', modelId: entry.id }));
    assert.equal(isLocalBrowserMessage(request, 'another-session'), false);
    for (const update of [{ action: 'select' }, { modelId: entry.id }, { action: 'select', modelId: '' },
      { action: 'select', modelId: 'x'.repeat(129) }, { action: 'execute' }, { viewId: '' }, { target: null }, { requestId: '' },
      { file: '/private/model.py' }, { cls: 'Anything' }]) assert.equal(isLocalBrowserMessage({ ...request, ...update }), false, JSON.stringify(update));
    const result = { ...envelope, type: 'modelCatalog', catalog };
    assert.ok(isLocalBrowserMessage(result));
    for (const update of [{ viewId: '' }, { requestId: '' }, { target: null }, { catalog: { ...catalog, activeId: 'unknown' } }, { command: 'execute' }]) {
      assert.equal(isLocalBrowserMessage({ ...result, ...update }), false, JSON.stringify(update));
    }
  });
});
