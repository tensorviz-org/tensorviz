import * as assert from 'node:assert/strict';
import { CanvasModelCatalog, type IndexedModelClass } from '../../canvasModelCatalog';
import { isModelCatalog, MODEL_CATALOG_LIMIT } from '@tensorviz/contracts/modelCatalog';
import type { LocalBrowserModelCatalogRequest, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';

suite('CanvasModelCatalog', () => {
  const target = (): LocalBrowserTargetIdentity => ({ workspaceFolder: '/project', file: 'models/net.py', cls: 'Net',
    sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 1 });
  function rig(timeoutMs?: number) {
    let current = target(), connected = true, reviewing = false, captured = false;
    let classes: readonly IndexedModelClass[] = [{ cls: 'Net', file: 'models/net.py' }, { cls: 'Decoder', file: 'models/decoder.py' }];
    let read = async (_folder: string, _refresh: boolean) => classes;
    const reads: Array<{ folder: string; refresh: boolean }> = [], replies: any[] = [], selected: any[] = [];
    const catalog = new CanvasModelCatalog({ target: () => current, connected: () => connected, reviewing: () => reviewing,
      captured: () => captured, classes: (folder, refresh) => { reads.push({ folder, refresh }); return read(folder, refresh); },
      select: async (model, folder, native, valid) => { if (!valid()) return; selected.push({ model, folder, native }); current = { ...current, ...model }; },
      publish: message => { replies.push(message); }, timeoutMs });
    const request = (action: 'list' | 'select' = 'list', extra = {}) => catalog.request({ type: 'modelCatalogRequest', protocolVersion: 1,
      workspaceSessionId: 'session', messageId: 'message', viewId: 'native', requestId: 'request', target: { ...current }, action, ...extra } as LocalBrowserModelCatalogRequest, true);
    return { catalog, reads, replies, selected, request, set classes(value: readonly IndexedModelClass[]) { classes = value; },
      set read(value: typeof read) { read = value; }, set target(value: LocalBrowserTargetIdentity) { current = value; }, get target() { return current; },
      set connected(value: boolean) { connected = value; }, set reviewing(value: boolean) { reviewing = value; }, set captured(value: boolean) { captured = value; } };
  }

  test('discovers source classes with opaque IDs and only marks real current capture evidence', async () => {
    const r = rig(); r.captured = true; await r.request();
    const reply = r.replies.at(-1), models = reply.catalog.models;
    assert.ok(isModelCatalog(reply.catalog)); assert.equal(reply.viewId, 'native'); assert.equal(reply.requestId, 'request');
    assert.deepEqual(r.reads, [{ folder: '/project', refresh: false }]);
    assert.equal(models.length, 2); assert.equal(models.find((model: any) => model.cls === 'Net').evidence, 'captured');
    assert.equal(models.find((model: any) => model.cls === 'Decoder').evidence, 'source');
    assert.equal(models.find((model: any) => model.cls === 'Decoder').opened, false);
    assert.ok(models.every((model: any) => !model.id.includes(model.file) && !model.id.includes(model.cls)));
    assert.equal(reply.catalog.activeId, models.find((model: any) => model.cls === 'Net').id);
    assert.equal(r.selected.length, 0, 'listing does not open or execute a model');
  });

  test('switches only a listed opaque model ID after successful fresh discovery', async () => {
    const r = rig(); r.catalog.opened(r.target); await r.request();
    const id = r.replies.at(-1).catalog.models.find((model: any) => model.cls === 'Decoder').id;
    await r.request('select', { modelId: id });
    assert.deepEqual(r.reads.at(-1), { folder: '/project', refresh: true });
    assert.deepEqual(r.selected, [{ model: { cls: 'Decoder', file: 'models/decoder.py' }, folder: '/project', native: true }]);
    await r.request();
    assert.ok(r.replies.at(-1).catalog.models.every((model: any) => model.opened));
  });

  test('refuses client paths and IDs that were not listed', async () => {
    const r = rig(); await r.request('select', { modelId: '../outside.py' });
    assert.equal(r.reads.length, 0); assert.equal(r.selected.length, 0); assert.equal(r.replies.at(-1).catalog.status, 'unavailable');
  });

  test('selection handles cannot cross projects or host sessions', async () => {
    const r = rig(), other = rig(); await r.request(); await other.request();
    const id = r.replies.at(-1).catalog.models[0].id;
    assert.notEqual(id, other.replies.at(-1).catalog.models[0].id);
    r.target = { ...r.target, workspaceFolder: '/other-project' };
    await r.request('select', { modelId: id }); assert.equal(r.selected.length, 0);
  });

  test('filters invalid paths, duplicate classes and preserves same-named classes in different files', async () => {
    const r = rig(); r.classes = [{ cls: 'Net', file: 'models/net.py' }, { cls: 'Net', file: 'models/net.py' },
      { cls: 'Net', file: 'other/net.py' }, { cls: 'Escape', file: '../escape.py' }, { cls: 'Absolute', file: '/tmp/absolute.py' }];
    await r.request(); assert.equal(r.replies.at(-1).catalog.models.length, 2); assert.ok(isModelCatalog(r.replies.at(-1).catalog));
  });

  for (const failure of ['removed', 'scan-failed', 'review'] as const) test(`does not switch when ${failure}`, async () => {
    const r = rig(); await r.request(); const id = r.replies.at(-1).catalog.models[0].id;
    if (failure === 'removed') r.classes = [];
    if (failure === 'scan-failed') r.read = async () => { throw new Error('Project model refresh failed.'); };
    if (failure === 'review') r.reviewing = true;
    await r.request('select', { modelId: id }); assert.equal(r.selected.length, 0); assert.equal(r.replies.at(-1).catalog.status, 'unavailable');
  });

  for (const failure of ['stale-target', 'disconnected', 'disposed'] as const) test(`ignores ${failure} requests`, async () => {
    const r = rig(); const old = r.target;
    if (failure === 'stale-target') r.target = { ...old, documentVersion: 2 };
    if (failure === 'disconnected') r.connected = false;
    if (failure === 'disposed') r.catalog.dispose();
    await r.request('list', { target: old }); assert.equal(r.replies.length, 0); assert.equal(r.reads.length, 0);
  });

  for (const change of ['target', 'disconnect', 'new-request', 'dispose'] as const) test(`suppresses late discovery after ${change}`, async () => {
    const r = rig(); let release!: (value: readonly IndexedModelClass[]) => void;
    r.read = () => new Promise(resolve => { release = resolve; }); const pending = r.request();
    if (change === 'target') r.target = { ...r.target, backendRevision: 2 };
    if (change === 'disconnect') r.connected = false;
    if (change === 'dispose') r.catalog.dispose();
    if (change === 'new-request') { r.read = async () => []; await r.request('list', { requestId: 'new' }); }
    const count = r.replies.length; release([{ cls: 'Late', file: 'late.py' }]); await pending;
    assert.equal(r.replies.length, count); assert.equal(r.selected.length, 0);
  });

  for (const change of ['target', 'disconnect', 'review'] as const) test(`cannot select after ${change} while revalidating the index`, async () => {
    const r = rig(); await r.request(); const models = r.replies.at(-1).catalog.models;
    let release!: (value: readonly IndexedModelClass[]) => void;
    r.read = () => new Promise(resolve => { release = resolve; });
    const pending = r.request('select', { modelId: models[0].id });
    if (change === 'target') r.target = { ...r.target, documentVersion: 2 };
    if (change === 'disconnect') r.connected = false;
    if (change === 'review') r.reviewing = true;
    release(models); await pending; assert.equal(r.selected.length, 0);
  });

  test('keeps failure details valid for the bounded transport contract', async () => {
    const r = rig(); r.read = async () => { throw new Error('\tinvalid\nprovider detail'); };
    await r.request(); assert.ok(isModelCatalog(r.replies.at(-1).catalog));
  });

  test('bounds discovery time and reports a retryable unavailable result', async () => {
    const r = rig(5); r.read = () => new Promise(() => {}); await r.request();
    assert.equal(r.replies.at(-1).catalog.status, 'unavailable'); assert.match(r.replies.at(-1).catalog.detail, /taking longer/);
  });

  test('limits large catalogues while retaining the active class', async () => {
    const r = rig(); r.classes = [...Array.from({ length: MODEL_CATALOG_LIMIT + 1 }, (_, index) => ({ cls: `Class${index}`, file: `models/class${index}.py` })),
      { cls: r.target.cls, file: r.target.file }];
    await r.request(); const result = r.replies.at(-1).catalog;
    assert.equal(result.models.length, MODEL_CATALOG_LIMIT); assert.equal(result.truncated, true); assert.ok(result.activeId); assert.ok(isModelCatalog(result));
  });
});
