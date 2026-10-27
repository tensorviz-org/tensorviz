import fixture from '../../test/fixtures/canvas/nestedAssignmentCapture.json';
import containers from '../../test/fixtures/canvas/nestedContainerCaptures.json';
import { buildGraphReview } from '../../ide/graphReview';
import { draftAssignmentReview } from '../../ide/capturedGraphReview';
import { assignmentContext } from '../../ide/assignmentContext';
import { localBrowserSyncLabel } from '../../ide/localBrowserPresentation';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import { applyProjection } from '../../ide/projection';
import type { GraphReviewState, ProjectionPayload } from '../../ide/protocol.generated';
import type { ModelNode } from '../../types';

/** The parent's real trace of `self.blocks[0].relu = nn.GELU()`: Net → blocks
 * (Sequential) → Block, whose forward calls the same ReLU three times. */
const KEY = '@assignment:blocks[0].relu';
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
const semantic = (payload: { document: unknown }) => (payload.document as any).semantic;
const captureNodes = semantic(fixture.capture).nodes as any[];
const byLabel = (label: string) => captureNodes.find(node => node.metadata.captureLabel === label).node_id as string;
const RELU_CALLS = [1, 2, 3].map(i => byLabel(`blocks.0.relu · call ${i}/3`));
const RELU_OPS = captureNodes.filter(node => RELU_CALLS.includes(node.metadata.parentId)).map(node => node.node_id as string);
const BLOCK = byLabel('blocks.0'), LINEAR = byLabel('blocks.0.fc');

type Fixture = typeof fixture;
const draftState = (data: Fixture = fixture): GraphReviewState => ({ id: 'draft', proposal: { id: 'draft' }, beforeLabel: 'Current code',
  afterLabel: 'Proposed code', before: data.before as ProjectionPayload, after: data.after as ProjectionPayload, changedFiles: ['model.py'],
  stale: false, liveContext: { workspaceFolder: '/ws', file: 'model.py', cls: 'Net' } });
const draft = (mutate: (data: Fixture) => void = () => undefined) => {
  const data = clone(fixture);
  mutate(data);
  const state = draftState(data);
  return draftAssignmentReview(state, buildGraphReview(state), data.capture as ProjectionPayload);
};
const accepted = (data: Fixture = fixture): ProjectionPayload => ({ ...(data.accepted as ProjectionPayload),
  debug: { previousDocument: data.capture.document, nodePairs: [], changed: [] } });

test('a pending GELU draft replaces all three captured ReLU calls in place, and nothing else', () => {
  const review = draft()!;
  expect(review).not.toBeNull();
  expect(review.changes).toHaveLength(3);
  review.changes.forEach((change, i) => {
    expect(change.label).toBe(`blocks[0].relu → nn.GELU · previous call ${i + 1}/3`);
    expect(change.detail).toBe('Assignment set in source · 3 previous calls · not run');
    expect(review.focusNodes?.get(change.id)).toEqual([RELU_CALLS[i]]);
  });
  const graph = review.projections.diff.reviewGraph;
  const nodes = new Map(graph.nodes.map(node => [node.id, node]));
  for (const id of RELU_CALLS) {
    const data = nodes.get(id)!.data;
    expect(data).toMatchObject({ diffStatus: 'modified', review: { status: 'modified' }, shapeEvidence: 'unknown',
      sourceReplacement: { key: KEY, target: 'blocks[0].relu', before: 'ReLU' }, moduleTemplateId: 'nn.GELU',
      sourceLocation: { file: 'model.py', line: 15 }, parameters: {}, expandable: false });
    // The source callable is a label, never an observed class.
    expect(data.layerType.name).toBe('nn.GELU');
    expect(JSON.stringify(data)).not.toMatch(/torch\.nn\.modules\.activation\.GELU/);
    expect(data.capturePorts!.every(port => port.shape === null && port.dtype === null)).toBe(true);
    expect(data.outputTensor).toBeUndefined();
    expect(data.inputShape).toBeUndefined();
    expect(data.outputShape).toBeUndefined();
    expect(data.captureDisplayPorts).toBeUndefined();
    expect(data.captureLabel).toBe(captureNodes.find(node => node.node_id === id).metadata.captureLabel);
  }
  // Old aten.relu internals are gone rather than shown under GELU.
  expect(RELU_OPS).toHaveLength(3);
  for (const id of RELU_OPS) expect(nodes.has(id)).toBe(false);
  // No detached assignment card and no invented node: only captured identities remain.
  expect(graph.nodes.map(node => node.id).sort()).toEqual(captureNodes.map(node => node.node_id).filter(id => !RELU_OPS.includes(id)).sort());
  expect(graph.nodes.some(node => node.data.sourceAssignment)).toBe(false);
  // Unrelated calls keep their captured identity, unhighlighted and historical.
  for (const node of graph.nodes.filter(node => !RELU_CALLS.includes(node.id))) {
    expect(node.data.review).toBeUndefined();
    expect(node.data.sourceReplacement).toBeUndefined();
    expect(node.data).toMatchObject({ debugEvidence: 'previous', shapeEvidence: 'stale' });
  }
  expect(nodes.get(LINEAR)!.data.layerType.name).toBe('torch.nn.modules.linear.Linear');
  expect(nodes.get(BLOCK)!.data.layerType.name).toBe('nested_preview_test.Block');
  // The captured flow, not a new one: same edge set and values, re-anchored on each call's recorded port.
  const captured = semantic(fixture.capture).edges as any[];
  expect(graph.edges.map(edge => edge.id).sort()).toEqual(captured.map(edge => edge.edge_id).sort());
  for (const edge of graph.edges) {
    expect(nodes.has(edge.source) && nodes.has(edge.target)).toBe(true);
    expect(edge.data!.captureEdge!.valueId).toBe(captured.find(item => item.edge_id === edge.id).metadata.captureEdge.valueId);
  }
  for (const id of RELU_CALLS) {
    const ports = nodes.get(id)!.data.capturePorts!;
    expect(graph.edges.filter(edge => edge.target === id).map(edge => edge.targetHandle)).toEqual([ports.find(port => port.direction === 'in')!.id]);
    expect(graph.edges.filter(edge => edge.source === id).map(edge => edge.sourceHandle)).toEqual([ports.find(port => port.direction === 'out')!.id]);
  }
});

test('the draft projection keeps proposed-source authority for identity, navigation and focus', () => {
  const projection = draft()!.projections.diff;
  expect(projection).toMatchObject({ proposal: true, capturedContext: true, graphId: fixture.capture.graphId,
    file: 'model.py', cls: 'Net', sourceDigest: fixture.after.sourceDigest, revision: fixture.after.revision, traced: false });
  expect(projection.document).toEqual(fixture.after.document);
  expect(projection.spanmap[KEY]).toMatchObject({ file: 'model.py', start: [15, 8] });
  expect(projection.contextKey).toBe(`${semantic(fixture.capture).metadata.capture.id}:${semantic(fixture.after).metadata.reviewClass.structureDigest}`);
});

test.each<[string, (data: Fixture) => void]>([
  ['source outside the assignments differs', data => { semantic(data.after).metadata.reviewClass.structureDigest = 'f'.repeat(64); }],
  ['the capture predates the structure digest', data => { delete semantic(data.capture).metadata.reviewClass.structureDigest; }],
  ['the capture is stale on screen', data => { data.capture.greyed = true; }],
  ['the capture is incomplete', data => { semantic(data.capture).metadata.capture.status = 'partial'; }],
  ['an Identity or eval Dropout records no operations', data => {
    semantic(data.capture).nodes.find((node: any) => node.node_id === RELU_CALLS[0]).metadata.capturedOperationCount = 0; }],
  ['a call passes the same value through its boundary', data => {
    const ports = semantic(data.capture).nodes.find((node: any) => node.node_id === RELU_CALLS[0]).metadata.capturePorts;
    ports.find((port: any) => port.direction === 'out').valueId = ports.find((port: any) => port.direction === 'in').valueId; }],
  ['the module is a shared alias', data => {
    semantic(data.capture).metadata.sourceCorrespondence.find((fact: any) => fact.nodeId === 'blocks.0.relu').basis = 'shared_instance'; }],
  ['the write is conditional', data => { semantic(data.after).metadata.reviewClass.assignments[KEY].conditional = true; }],
  ['the write is ambiguous', data => { semantic(data.after).metadata.reviewClass.assignments[KEY].ambiguous = true; }],
  ['a legacy fact has no AST path', data => { delete semantic(data.after).metadata.reviewClass.assignments[KEY].path; }],
  ['an integer index meets a custom container', data => {
    semantic(data.capture).metadata.registeredContainers.blocks.definition = 'nested_preview_test.Blocks'; }],
  ['a container has named keys', data => { semantic(data.capture).metadata.registeredContainers.blocks.keys = ['first']; }],
  ['a numeric container has a hole', data => { semantic(data.capture).metadata.registeredContainers.blocks.keys = ['1']; }],
  ['container order differs from numeric keys', data => { semantic(data.capture).metadata.registeredContainers.blocks.keys = ['1', '0']; }],
  ['the container inventory is missing', data => { delete semantic(data.capture).metadata.registeredContainers; }],
  ['a string key meets a Sequential', data => { semantic(data.after).metadata.reviewClass.assignments[KEY].path[1].key = '0'; }],
  ['the registered path has no captured call', data => { semantic(data.after).metadata.reviewClass.assignments[KEY].path[2].key = 'act'; }],
  ['the capture belongs to another model', data => { data.capture.cls = 'Other'; }],
])('refuses captured context when %s', (_reason, mutate) => {
  expect(draft(mutate)).toBeNull();
});

test('a refused context leaves the pending source draft and its unresolved navigation intact', () => {
  const data = clone(fixture);
  semantic(data.capture).metadata.sourceCorrespondence.find((fact: any) => fact.nodeId === 'blocks.0.relu').basis = 'shared_instance';
  const state = draftState(data);
  const source = buildGraphReview(state);
  expect(draftAssignmentReview(state, source, data.capture as ProjectionPayload)).toBeNull();
  expect(source.changes.map(change => change.label)).toEqual(['blocks[0].relu → nn.GELU']);
  expect(source.projections.diff.capturedContext).toBeUndefined();
});

test('accepted source keeps the same in-context replacement, without the detached card', () => {
  const draftGraph = draft()!.projections.diff.reviewGraph;
  const context = assignmentContext(fixture.capture.document, fixture.accepted.document)!;
  expect(context.graph.nodes.map(node => node.id)).toEqual(draftGraph.nodes.map(node => node.id));
  expect(context.calls.get(KEY)).toEqual(RELU_CALLS);
  for (const id of RELU_CALLS) expect(context.graph.nodes.find(node => node.id === id)!.data.sourceReplacement)
    .toMatchObject({ key: KEY, target: 'blocks[0].relu', before: 'ReLU' });
  // A changed structure falls back to the source view rather than the stale run.
  const changed = clone(fixture.accepted);
  semantic(changed).metadata.reviewClass.structureDigest = '0'.repeat(64);
  expect(assignmentContext(fixture.capture.document, changed.document)).toBeNull();
});

describe('canvas lifecycle', () => {
  beforeAll(() => { (window as any).__TENSORVIZ_LOCAL_BROWSER__ = true; });
  afterAll(() => { delete (window as any).__TENSORVIZ_LOCAL_BROWSER__; });
  const at = (nodes: readonly ModelNode[], id: string) => nodes.find(node => node.id === id)!;

  test('Accept keeps the draft view as Not run, Discard restores the capture, and validation removes draft evidence', () => {
    const capture = fixture.capture as ProjectionPayload;
    const live = applyProjection(capture, [])!;
    const baseline = live.nodes.map((node, i) => ({ ...node, position: { x: 40 * i, y: 10 } }));

    // Pending draft: drawn on the retained capture's arrangement.
    const pending = applyProjection(draft()!.projections.diff, baseline)!;
    expect(at(pending.nodes, BLOCK).position).toEqual(at(baseline, BLOCK).position);
    // Focus opened Block; the draft's view must survive Accept.
    const focused = pending.nodes.map(node => node.id === BLOCK ? { ...node, position: { x: 900, y: 300 },
      data: { ...node.data, expanded: true } } : node);

    // The host closes the draft before writing; the old capture may be redrawn first.
    const closing = applyProjection(capture, focused)!;
    expect(at(closing.nodes, BLOCK).position).toEqual(at(baseline, BLOCK).position);
    const kept = applyProjection(accepted(), closing.nodes)!;
    expect(at(kept.nodes, BLOCK)).toMatchObject({ position: { x: 900, y: 300 }, data: { expanded: true } });
    expect(kept.nodes.some(node => node.data.sourceAssignment || RELU_OPS.includes(node.id))).toBe(false);
    for (const id of RELU_CALLS) expect(at(kept.nodes, id).data.sourceReplacement?.target).toBe('blocks[0].relu');
    // Entirely Not run: nothing claims current measured evidence.
    expect(kept.nodes.some(node => node.data.debugEvidence === 'current' || node.data.shapeEvidence === 'measured')).toBe(false);

    // Explicit validation: the new capture replaces every draft fact.
    const measured = applyProjection(fixture.validated as ProjectionPayload, kept.nodes)!;
    expect(measured.nodes.some(node => node.data.sourceReplacement || node.data.review)).toBe(false);
    const geluCalls = measured.nodes.filter(node => node.data.layerType.name === 'torch.nn.modules.activation.GELU');
    expect(geluCalls).toHaveLength(3);
    for (const node of geluCalls) expect(node.data.shapeEvidence).toBe('measured');
    expect(measured.nodes.some(node => node.data.layerType.name === 'torch.nn.modules.activation.ReLU')).toBe(false);

    // Discard: the original capture, its old ReLU internals and its arrangement return.
    const before = applyProjection(capture, measured.nodes)!.nodes.map((node, i) => ({ ...node, position: { x: 7 * i, y: 70 } }));
    const second = applyProjection(draft()!.projections.diff, before)!;
    const discarded = applyProjection(capture, second.nodes.map(node => ({ ...node, position: { x: 1, y: 1 } })))!;
    expect(discarded.nodes).toHaveLength(before.length);
    expect(new Set(discarded.nodes.map(node => node.id)).size).toBe(before.length);
    expect(new Map(discarded.nodes.map(node => [node.id, node.position]))).toEqual(new Map(before.map(node => [node.id, node.position])));
    expect(discarded.nodes.some(node => node.data.sourceReplacement || node.data.review)).toBe(false);
  });

  test('a source that is not the proven accepted draft restores the live view', () => {
    const capture = fixture.capture as ProjectionPayload;
    const live = applyProjection(capture, [])!.nodes.map((node, i) => ({ ...node, position: { x: 5 * i, y: 5 } }));
    const pending = applyProjection(draft()!.projections.diff, live)!;
    const moved = pending.nodes.map(node => ({ ...node, position: { x: 600, y: 600 } }));
    const other = clone(fixture);
    semantic(other.accepted).metadata.reviewClass.structureDigest = 'e'.repeat(64);
    const shown = applyProjection(accepted(other), moved)!;
    // Unverified placement stays a source review entry, never a detached module.
    expect(shown.nodes.some(node => node.data.captureRef)).toBe(false);
    expect(shown.nodes.some(node => node.data.sourceAssignment)).toBe(false);
    expect(shown.nodes.every(node => node.position.x !== 600)).toBe(true);
  });
});

 test('an uncalled ModuleList is resolved by its registered inventory, not an invocation', () => {
  const data = clone(fixture);
  semantic(data.capture).metadata.registeredContainers.blocks.definition = 'torch.nn.modules.container.ModuleList';
  semantic(data.capture).metadata.sourceCorrespondence = semantic(data.capture).metadata.sourceCorrespondence.filter((fact: any) => fact.nodeId !== 'blocks');
  expect(assignmentContext(data.capture.document, data.after.document)?.calls.get(KEY)).toEqual(RELU_CALLS);
});

test('accepted status distinguishes historical context from an unlocated source edit', () => {
  const projection = accepted();
  const page = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, syncState: 'source-only' as const, projection };
  expect(localBrowserSyncLabel(page)?.text).toBe('Source changes · Not run · Previous run context');
  expect(localBrowserSyncLabel({ ...page, projection: { ...projection, debug: undefined } })?.text)
    .toBe('Not run — shapes read from the source');
  const changed = clone(projection);
  semantic(changed).metadata.reviewClass.structureDigest = 'other';
  expect(localBrowserSyncLabel({ ...page, projection: changed })?.text)
    .toBe('Source edits not located in graph · Validate to update');
  const unchanged = clone(projection);
  (unchanged.debug!.previousDocument as any).semantic.metadata.reviewClass.assignments = semantic(unchanged).metadata.reviewClass.assignments;
  expect(localBrowserSyncLabel({ ...page, projection: unchanged })?.text).toBe('Not run — shapes read from the source');
});

test('several consumers collapsed onto one call boundary draw a single connection', () => {
  const data = clone(fixture);
  const edges = semantic(data.capture).edges;
  const edge = edges.find((item: any) => RELU_OPS.includes(item.target_node_id));
  expect(edge).toBeDefined();
  edges.push({ ...clone(edge), edge_id: `${edge.edge_id}:duplicate-consumer` });
  const context = assignmentContext(data.capture.document, data.after.document)!;
  const endpoints = context.graph.edges.map(item => JSON.stringify([item.source, item.sourceHandle, item.target, item.targetHandle]));
  expect(new Set(endpoints).size).toBe(endpoints.length);
});

test('a discarded draft cannot donate its layout to a different nested edit', () => {
  (window as any).__TENSORVIZ_LOCAL_BROWSER__ = true;
  try {
    const live = applyProjection(fixture.capture as ProjectionPayload, [])!;
    const pending = applyProjection(draft()!.projections.diff, live.nodes)!;
    const moved = pending.nodes.map(node => ({ ...node, position: { x: 98765, y: 98765 } }));
    const closed = applyProjection(fixture.capture as ProjectionPayload, moved)!;
    const different = { ...accepted(), sourceDigest: 'd'.repeat(64) };
    const shown = applyProjection(different, closed.nodes)!;
    expect(shown.nodes.some(node => node.position.x === 98765)).toBe(false);
  } finally { delete (window as any).__TENSORVIZ_LOCAL_BROWSER__; }
});

describe('attested dependency boundaries', () => {
  const dependencyFixture = () => {
    const data = clone(fixture);
    semantic(data.capture).metadata.reviewClass.dependencyDigests = { 'factory.py': 'factory-v1', 'base.py': 'base-v1' };
    semantic(data.after).metadata.reviewClass.dependencyDigests = { 'factory.py': 'factory-v1', 'base.py': 'base-v1' };
    return data;
  };
  test.each(['changed transitive dependency', 'missing dependency', 'legacy scope', 'legacy manifest'])(
    'refuses %s even when static structure matches', reason => {
      const data = dependencyFixture();
      const capture = semantic(data.capture).metadata.reviewClass, source = semantic(data.after).metadata.reviewClass;
      if (reason === 'changed transitive dependency') source.dependencyDigests['base.py'] = 'base-v2';
      if (reason === 'missing dependency') delete source.dependencyDigests['base.py'];
      if (reason === 'legacy scope') delete capture.dependencyScope;
      if (reason === 'legacy manifest') delete capture.dependencyDigests;
      expect(assignmentContext(data.capture.document, data.after.document)).toBeNull();
    });
  test('unrelated extra candidate sources do not invalidate attested context', () => {
    const data = dependencyFixture();
    semantic(data.after).metadata.reviewClass.dependencyDigests['another-model.py'] = 'unrelated';
    expect(assignmentContext(data.capture.document, data.after.document)?.calls.get(KEY)).toEqual(RELU_CALLS);
  });
});

test.each<[string, (data: Fixture) => void]>([
  ['removed override', data => {
    semantic(data.capture).metadata.reviewClass.assignments = clone(semantic(data.after).metadata.reviewClass.assignments);
    semantic(data.after).metadata.reviewClass.assignments = {};
  }],
  ['missing captured invocation', data => { semantic(data.capture).nodes = semantic(data.capture).nodes.filter((n: any) => n.node_id !== RELU_CALLS[2]); }],
  ['duplicate source correspondence', data => {
    const facts = semantic(data.capture).metadata.sourceCorrespondence;
    facts.push(clone(facts.find((f: any) => f.nodeId === 'blocks.0.relu')));
  }],
  ['foreign capture correspondence', data => {
    semantic(data.capture).metadata.sourceCorrespondence.find((f: any) => f.nodeId === 'blocks.0.relu').captureId = 'another-run';
  }],
  ['missing boundary port', data => {
    semantic(data.capture).nodes.find((n: any) => n.node_id === RELU_CALLS[0]).metadata.capturePorts = [];
  }],
  ['ambiguous boundary port', data => {
    const ports = semantic(data.capture).nodes.find((n: any) => n.node_id === RELU_CALLS[0]).metadata.capturePorts;
    ports.push({ ...ports.find((p: any) => p.direction === 'in'), id: 'second-input' });
  }],
  ['overlapping ancestor and descendant assignments', data => {
    semantic(data.after).metadata.reviewClass.assignments['@assignment:blocks[0]'] = {
      ...clone(semantic(data.after).metadata.reviewClass.assignments[KEY]), target: 'blocks[0]', path: [{ kind: 'attribute', key: 'blocks' }, { kind: 'index', key: 0 }],
    };
  }],
  ['one of several edits cannot be mapped', data => {
    semantic(data.after).metadata.reviewClass.assignments['@assignment:blocks[1].relu'] = {
      ...clone(semantic(data.after).metadata.reviewClass.assignments[KEY]), target: 'blocks[1].relu',
      path: [{ kind: 'attribute', key: 'blocks' }, { kind: 'index', key: 1 }, { kind: 'attribute', key: 'relu' }],
    };
  }],
])('never shows a partially truthful graph for %s', (_reason, mutate) => {
  const data = clone(fixture); mutate(data);
  expect(assignmentContext(data.capture.document, data.after.document)).toBeNull();
});

test('successive accepted edits and undo keep all three call identities and clear old measurements', () => {
  const gelu = clone(fixture.after);
  const silu = clone(fixture.after);
  const fact = semantic(silu).metadata.reviewClass.assignments[KEY];
  fact.title = 'nn.SiLU'; fact.value = 'nn.SiLU()'; fact.digest = 'silu-assignment';
  for (const source of [gelu, silu, gelu]) {
    const context = assignmentContext(fixture.capture.document, source.document)!;
    expect(context.calls.get(KEY)).toEqual(RELU_CALLS);
    for (const node of context.graph.nodes.filter(node => RELU_CALLS.includes(node.id))) {
      expect(node.data.sourceReplacement?.before).toBe('ReLU');
      expect(node.data.layerType.name).toBe(semantic(source).metadata.reviewClass.assignments[KEY].title);
      expect(node.data.shapeEvidence).toBe('unknown');
      expect(node.data.parameterCount).toBeUndefined();
    }
  }
  const revision = assignmentContext(fixture.capture.document, silu.document, gelu.document)!;
  expect(revision.graph.nodes.filter(node => RELU_CALLS.includes(node.id)).map(node => node.data.sourceReplacement?.capturedType)).toEqual(['ReLU', 'ReLU', 'ReLU']);
  expect(revision.graph.nodes.filter(node => RELU_CALLS.includes(node.id)).map(node => node.data.sourceReplacement?.before))
    .toEqual(['nn.GELU', 'nn.GELU', 'nn.GELU']);
});


test.each(['ModuleList', 'ModuleDict', 'Multifile', 'PartialMultifile', 'HostPartial'] as const)('real %s captures locate replacements despite the container never being invoked', kind => {
  const { capture, after } = containers[kind];
  const context = assignmentContext(capture, after)!;
  expect(context).not.toBeNull();
  const calls = Array.from(context.calls.values())[0];
  expect(calls).toHaveLength(3);
  const target = kind === 'ModuleDict' ? "blocks['head'].relu" : 'blocks[0].relu';
  for (const id of calls) {
    const node = context.graph.nodes.find(node => node.id === id)!;
    expect(node.data.sourceReplacement?.target).toBe(target);
    expect(node.data.shapeEvidence).toBe('unknown');
  }
});


test('partial dependency coverage produces explicitly unverified source annotations', () => {
  const data = clone(fixture);
  semantic(data.capture).metadata.reviewClass.dependencyScope = 'partial';
  const context = assignmentContext(data.capture.document, data.after.document)!;
  expect(context.calls.get(KEY)).toEqual(RELU_CALLS);
  for (const node of context.graph.nodes.filter(node => RELU_CALLS.includes(node.id))) {
    expect(node.data.sourceReplacement).toMatchObject({ coverage: 'partial', capturedType: 'ReLU' });
    expect(node.data.shapeEvidence).toBe('unknown');
    expect(node.data.debugEvidence).toBeUndefined();
  }
});


test.each(['changed', 'missing'])('partial capture still refuses a %s known candidate', change => {
  const data = clone(containers.PartialMultifile);
  const metadata = (data.capture as any).semantic.metadata.reviewClass;
  expect(metadata.dependencyScope).toBe('partial');
  const current = (data.after as any).semantic.metadata.reviewClass.dependencyDigests;
  if (change === 'missing') delete current['review_base.py'];
  else current['review_base.py'] = 'dirty-buffer';
  expect(assignmentContext(data.capture, data.after)).toBeNull();
});
