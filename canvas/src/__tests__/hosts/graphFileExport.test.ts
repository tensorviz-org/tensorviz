import { canExportGraphFile, downloadGraphFile, exportCurrentGraphFile, prepareCurrentGraphFile, type GraphExportState } from '../../ide/graphFileExport';
import captureDocument from '../../test/fixtures/canvas/capture-document.json';
import portable from '../../../../tests/fixtures/contracts/portable-capture.json';
import { requirePortableCapture } from '../../services/portableCapture';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import type { GraphDocumentV2 } from '../../types/graphDocument';
import { createGraphFile } from '../../services/graphFile';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE, LocalBrowserPageState } from '../../ide/localBrowserState';
import { ProjectionPayload } from '../../ide/protocol.generated';

function pageState(): LocalBrowserPageState {
  const projection: ProjectionPayload = {
    graphId: 'live-session-graph', revision: 3, cls: 'Net', file: 'models/net.py', sourceDigest: 'source-v3',
    traced: false, greyed: false, viewRestored: false, spanmap: {},
    document: createGraphFile(nestedGraph(), { name: 'Net', status: 'static' }).document,
  };
  return { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, activeTarget: { workspaceFolder: '/Users/private/project', file: 'models/net.py', cls: 'Net', documentVersion: 3, sourceDigest: 'source-v3', backendRevision: 3 },
    projection, traced: false, greyed: false, syncState: 'source-only', reason: null };
}

it('exports complete current canvas state, not a visible-node filter or old host layout', () => {
  const page = pageState();
  const graph = nestedGraph(); graph.nodes[3].position = { x: 475, y: 188 }; graph.nodes[3].hidden = true;
  const file = exportCurrentGraphFile(graph, page, page.projection);
  expect(file.document.semantic.nodes).toHaveLength(5);
  expect(file.document.view.node_positions.fc).toEqual({ x: 475, y: 188 });
  expect(file.document.semantic.name).toBe('Net');
  expect(JSON.stringify(file)).not.toMatch(/live-session|source-v3|Users|models\/net/);
});

it('refuses an export during retargeting or before the displayed projection catches up', () => {
  const page = pageState();
  expect(canExportGraphFile(page)).toBe(true);
  expect(canExportGraphFile({ ...page, syncState: null })).toBe(false);
  expect(canExportGraphFile({ ...page, activeTarget: { ...page.activeTarget!, cls: 'OtherNet' } })).toBe(false);
  expect(() => exportCurrentGraphFile(nestedGraph(), page, { ...page.projection!, revision: 2 })).toThrow(/finish opening/);
});

it.each([
  ['stale', false, false, true, 'stale'],
  ['runtime-traced', true, false, true, 'traced'],
  ['source-only', false, true, true, 'stale'],
  ['source-only', false, false, false, 'stale'],
] as const)('preserves state %s with connection=%s', (syncState, traced, greyed, connected, expected) => {
  const page = { ...pageState(), syncState, traced, greyed };
  expect(exportCurrentGraphFile(nestedGraph(), page, page.projection, connected).status).toBe(expected);
});

it('downloads a local JSON Blob and revokes its temporary URL', () => {
  jest.useFakeTimers();
  const originalCreate = URL.createObjectURL;
  const originalRevoke = URL.revokeObjectURL;
  const create = jest.fn(() => 'blob:graph-download');
  const revoke = jest.fn();
  URL.createObjectURL = create; URL.revokeObjectURL = revoke;
  const click = jest.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    expect(this.download).toBe('Net.tensorviz.json');
    expect(this.href).toBe('blob:graph-download');
  });
  try {
    const page = pageState(); downloadGraphFile(exportCurrentGraphFile(nestedGraph(), page, page.projection));
    expect(create.mock.calls).toHaveLength(1);
    expect(click).toHaveBeenCalledTimes(1);
    expect(document.querySelector('a[download]')).toBeNull();
    jest.runAllTimers(); expect(revoke).toHaveBeenCalledWith('blob:graph-download');
  } finally { click.mockRestore(); URL.createObjectURL = originalCreate; URL.revokeObjectURL = originalRevoke; jest.useRealTimers(); }
});

function captureState(): GraphExportState {
  const page = pageState();
  page.syncState = 'runtime-traced'; page.traced = true;
  return { page, displayed: page.projection, connected: true,
    graph: graphDocumentV2ToModelGraph(captureDocument as GraphDocumentV2) };
}

it('live canonical export requests retained evidence and preserves exact graph/operands with latest layout', async () => {
  const state = captureState();
  state.direction = 'horizontal';
  const prepare = jest.fn(async () => {
    state.graph.nodes[0].position = { x: 91, y: 32 };
    state.direction = 'vertical';
    return { capture: requirePortableCapture(portable), detail: 'Prepared.' };
  });
  const file = await prepareCurrentGraphFile(() => state, { prepare, cancel() {}, dispose() {} });
  expect(prepare).toHaveBeenCalledWith({ captureId: portable.capture.id, sourceId: 'source:0' });
  expect(file.capture).toEqual(portable);
  expect(file.document.semantic.edges).toEqual(captureDocument.semantic.edges);
  expect(file.document.view.node_positions[state.graph.nodes[0].id]).toEqual({ x: 91, y: 32 });
  expect(file.document.view.layout.direction).toBe('vertical');
});

it.each(['target', 'projection', 'disconnected', 'capture'])('refuses %s changes during asynchronous export', async kind => {
  const state = captureState();
  const prepare = async () => {
    if (kind === 'target') state.page = { ...state.page, activeTarget: { ...state.page.activeTarget!, backendRevision: 4 } };
    if (kind === 'projection') state.displayed = { ...state.displayed!, revision: 4 };
    if (kind === 'disconnected') state.connected = false;
    if (kind === 'capture') state.graph = { ...state.graph, nodes: state.graph.nodes.map(node => ({ ...node,
      data: { ...node.data, captureRef: { ...node.data.captureRef!, captureId: 'new-run' } } })) };
    return { capture: requirePortableCapture(portable), detail: 'Prepared.' };
  };
  await expect(prepareCurrentGraphFile(() => ({ ...state }), { prepare, cancel() {}, dispose() {} })).rejects.toThrow();
});

it('exports a stale canonical graph honestly and keeps missing-capture refusal explicit', async () => {
  const state = captureState(); state.page.syncState = 'stale'; state.page.greyed = true;
  const client = { prepare: async () => ({ capture: requirePortableCapture(portable), detail: 'Prepared.' }), cancel() {}, dispose() {} };
  expect((await prepareCurrentGraphFile(() => state, client)).status).toBe('stale');
  await expect(prepareCurrentGraphFile(() => state)).rejects.toThrow(/Connect/);
  await expect(prepareCurrentGraphFile(() => state, { ...client, prepare: async () => ({ capture: null, detail: 'Capture was evicted.' }) })).rejects.toThrow(/evicted/);
});
