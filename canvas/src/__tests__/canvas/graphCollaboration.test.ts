import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { anchorStatus, captureAnchor, edgeTarget, portableSnapshotId, updateArchitecture, WalkthroughStep } from '../../services/graphCollaboration';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';

const file = () => createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
function annotated() {
  const f = file();
  const step: WalkthroughStep = { id: 'step-1', title: 'Prediction head', note: 'First line\nSecond line', anchors: [captureAnchor(f, { kind: 'node', nodeId: 'fc' })],
    view: { camera: { x: 12, y: 20, z: .8 }, nodePositions: f.document.view.node_positions, expandedModuleIds: ['block', 'inner'], selectedNodeIds: ['fc'] } };
  f.collaboration.walkthrough.steps = [step];
  f.collaboration.threads = [{ id: 'thread-1', anchor: captureAnchor(f, edgeTarget(f.document.semantic.edges[0])), resolved: false,
    messages: [{ id: 'message-1', author: 'Sam', body: 'Why this connection?\nPlease explain.', createdAt: '2026-09-12T10:00:00Z' }] }];
  return f;
}
it('normalizes existing version 1 files with stable snapshot identity', () => {
  const f: any = file(); f.version = 1; delete f.collaboration; delete f.snapshotId;
  const a = parseGraphFile(JSON.stringify(f)); const b = parseGraphFile(JSON.stringify(f));
  expect(a.version).toBe(3); expect(a.snapshotId).toBe(b.snapshotId);
  expect(a.collaboration.threads).toEqual([]);
});
it('round trips views, multiline notes, connection context and replies without changing snapshot time', () => {
  const f = annotated();
  f.collaboration.threads[0].messages.push({ id: 'reply-1', author: 'Lee', body: '<script>plain text only</script>', createdAt: '2026-09-12T10:05:00Z' });
  expect(parseGraphFile(JSON.stringify(f))).toEqual(f);
});
it('ignores layout, edge order and sequential edge ids when checking context', () => {
  const f = annotated(); const next = file();
  next.document.view.node_positions.fc = { x: 900, y: 20 };
  next.document.semantic.edges.reverse().forEach((edge, i) => edge.edge_id = `changed-${i}`);
  expect(anchorStatus(next, f.collaboration.threads[0].anchor)).toBe('unchanged');
  expect(anchorStatus(next, f.collaboration.walkthrough.steps[0].anchors[0])).toBe('unchanged');
  expect(portableSnapshotId(next.document, f.exportedAt)).toBe(portableSnapshotId(f.document, f.exportedAt));
});
it('marks changed parameters and connections while preserving original context on update', () => {
  const f = annotated(); const next = file(); next.document.semantic.nodes.find(n => n.node_id === 'fc')!.params.out_features = 100;
  const moduleAnchor = captureAnchor(f, { kind: 'node', nodeId: 'block' });
  const updated = updateArchitecture(f, next);
  expect(anchorStatus(updated, moduleAnchor)).toBe('changed');
  expect(updated.collaboration).toEqual(f.collaboration);
  expect(anchorStatus(updated, f.collaboration.threads[0].anchor)).toBe('changed');
  expect(anchorStatus(updated, f.collaboration.walkthrough.steps[0].anchors[0])).toBe('changed');
  next.document.semantic.edges.pop();
  expect(anchorStatus(next, f.collaboration.walkthrough.steps[0].anchors[0])).toBe('changed');
});
it('keeps missing and ambiguous targets visible instead of rebinding them', () => {
  const f = annotated(); const next = file();
  next.document.semantic.edges.push({ ...next.document.semantic.edges[0], edge_id: 'duplicate-endpoints' });
  expect(anchorStatus(next, f.collaboration.threads[0].anchor)).toBe('ambiguous');
  next.document.semantic.edges = [];
  expect(anchorStatus(next, f.collaboration.threads[0].anchor)).toBe('missing');
  next.document.semantic.nodes = next.document.semantic.nodes.filter(n => n.node_id !== 'fc');
  delete next.document.view.node_positions.fc;
  const updated = parseGraphFile(JSON.stringify(updateArchitecture(f, next)));
  expect(updated.collaboration.walkthrough.steps).toHaveLength(1);
  expect(anchorStatus(updated, f.collaboration.walkthrough.steps[0].anchors[0])).toBe('missing');
});
it('refuses an annotated replacement rather than losing either set of notes', () => {
  expect(() => updateArchitecture(annotated(), annotated())).toThrow(/contains notes/);
});
it.each([
  ['overlong note', (f: any) => f.collaboration.walkthrough.steps[0].note = 'x'.repeat(4001)],
  ['invalid zoom', (f: any) => f.collaboration.walkthrough.steps[0].view.camera.z = 0],
  ['unsafe identity', (f: any) => f.collaboration.threads[0].id = '__proto__'],
  ['duplicate steps', (f: any) => f.collaboration.walkthrough.steps.push(f.collaboration.walkthrough.steps[0])],
  ['no messages', (f: any) => f.collaboration.threads[0].messages = []],
  ['too many threads', (f: any) => f.collaboration.threads = Array.from({length:201}, (_, i) => ({...f.collaboration.threads[0],id:`thread-${i}`}))],
  ['bad target', (f: any) => f.collaboration.threads[0].anchor.target.kind = 'source'],
])('rejects %s', (_, mutate) => { const f = annotated(); mutate(f); expect(() => parseGraphFile(JSON.stringify(f))).toThrow(); });
it('refuses version 1 annotations so they cannot be silently discarded', () => {
  const f: any = annotated(); f.version = 1; expect(() => parseGraphFile(JSON.stringify(f))).toThrow(/version 2/);
});

function saved() {
  const f = annotated();
  f.collaboration.savedViews = [{ ...f.collaboration.walkthrough.steps[0], id: 'view-1' }];
  f.collaboration.openingViewId = 'view-1';
  return f;
}
it('reads version 2 without losing walkthroughs or discussions', () => {
  const legacy: any = annotated(); legacy.version = 2;
  delete legacy.collaboration.savedViews; delete legacy.collaboration.openingViewId;
  const normalized = parseGraphFile(JSON.stringify(legacy));
  expect(normalized.version).toBe(3);
  expect(normalized.collaboration.walkthrough).toEqual(legacy.collaboration.walkthrough);
  expect(normalized.collaboration.threads).toEqual(legacy.collaboration.threads);
  expect(normalized.collaboration.savedViews).toEqual([]);
  expect(normalized.collaboration.openingViewId).toBeNull();
});
it('round trips an opening view with semantic edge selection and viewport dimensions', () => {
  const f = saved(); const edge = edgeTarget(f.document.semantic.edges[0]);
  if (edge.kind !== 'edge') throw new Error('Expected edge');
  f.collaboration.savedViews[0] = { ...f.collaboration.savedViews[0], anchors: [captureAnchor(f, edge)],
    view: { ...f.collaboration.savedViews[0].view, selectedNodeIds: [], selectedEdge: edge, viewport: { width: 1200, height: 800 } } };
  expect(parseGraphFile(JSON.stringify(f))).toEqual(f);
  const incoming = file(); incoming.document.semantic.edges.reverse().forEach((edge, i) => edge.edge_id = `new-${i}`);
  const updated = parseGraphFile(JSON.stringify(updateArchitecture(f, incoming)));
  expect(updated.collaboration.savedViews).toEqual(f.collaboration.savedViews);
  expect(updated.collaboration.openingViewId).toBe('view-1');
  expect(anchorStatus(updated, f.collaboration.savedViews[0].anchors[0])).toBe('unchanged');
});
it('preserves original saved view anchors when a target disappears and protects incoming saved views', () => {
  const f = saved(); const incoming = file();
  incoming.document.semantic.edges = [];
  incoming.document.semantic.nodes = incoming.document.semantic.nodes.filter(n => n.node_id !== 'fc');
  delete incoming.document.view.node_positions.fc;
  const updated = parseGraphFile(JSON.stringify(updateArchitecture(f, incoming)));
  expect(anchorStatus(updated, updated.collaboration.savedViews[0].anchors[0])).toBe('missing');
  f.collaboration.walkthrough.steps = []; f.collaboration.threads = [];
  expect(() => updateArchitecture(file(), f)).toThrow(/contains notes/);
});
it.each([
  ['dangling opening view', (f: any) => f.collaboration.openingViewId = 'missing'],
  ['duplicate saved views', (f: any) => f.collaboration.savedViews.push(f.collaboration.savedViews[0])],
  ['too many saved views', (f: any) => f.collaboration.savedViews = Array.from({length:51}, (_, i) => ({...f.collaboration.savedViews[0],id:`view-${i}`}))],
  ['unsafe saved identity', (f: any) => f.collaboration.savedViews[0].id = '__proto__'],
  ['unanchored saved target', (f: any) => f.collaboration.savedViews[0].anchors = []],
  ['invalid selected edge', (f: any) => f.collaboration.savedViews[0].view.selectedEdge = {kind:'node',nodeId:'fc'}],
  ['invalid viewport', (f: any) => f.collaboration.savedViews[0].view.viewport = {width:0,height:800}],
  ['nonfinite viewport', (f: any) => f.collaboration.savedViews[0].view.viewport = {width:Infinity,height:800}],
  ['oversized saved note', (f: any) => f.collaboration.savedViews[0].note = 'a'.repeat(4001)],
])('rejects %s', (_, mutate) => { const f = saved(); mutate(f); expect(() => parseGraphFile(JSON.stringify(f))).toThrow(); });
it('requires version 3 for saved views so older viewers cannot silently discard them', () => {
  const f: any = saved(); f.version = 2;
  expect(() => parseGraphFile(JSON.stringify(f))).toThrow(/version 3/);
});
