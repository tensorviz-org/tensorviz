import { buildGraphReview } from '../../ide/graphReview';
import { graphDebugPresentation } from '../../ide/graphDebugPresentation';
import { exportCurrentGraphFile } from '../../ide/graphFileExport';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import type { GraphReviewState, ProjectionPayload } from '../../ide/protocol.generated';
import type { SourceAssignmentFact } from '../../ide/sourceAssignmentCards';
import type { GraphDocumentV2 } from '../../types/graphDocument';
// Exact backend projections of a ResNet subclass over an opaque torchvision base.
import fixture from '../../test/fixtures/canvas/draftResNet.json';

const KEY = '@assignment:layer3[0].relu';
const fact = (title: string, overrides: Partial<SourceAssignmentFact> = {}): SourceAssignmentFact => ({
  target: 'layer3[0].relu', digest: `digest-${title}`, title, value: `${title}()`, ambiguous: false, conditional: false,
  span: { file: 'vision_models.py', start: [10, 8], end: [10, 42], kind: 'decl' } as SourceAssignmentFact['span'], ...overrides,
});
const reviewClass = (projection: ProjectionPayload) => (projection.document as GraphDocumentV2).semantic.metadata.reviewClass as Record<string, unknown>;

/** The current ResNet source on both sides; only the class body differs, by the
 * given literal assignments. Nothing the projector lowers changes. */
function draft(before: Record<string, SourceAssignmentFact> = {}, after: Record<string, SourceAssignmentFact> = {},
  proposal: GraphReviewState['proposal'] | null = { id: 'proposal-1' }): GraphReviewState {
  const source = JSON.parse(JSON.stringify(fixture)).before;
  const side = (assignments: Record<string, SourceAssignmentFact>, digest: string) => {
    const projection = JSON.parse(JSON.stringify(source)) as ProjectionPayload;
    reviewClass(projection).digest = Object.keys(assignments).length ? `${digest}:${Object.values(assignments).map(item => item.digest).join()}` : digest;
    if (Object.keys(assignments).length) reviewClass(projection).assignments = assignments;
    return projection;
  };
  return { ...JSON.parse(JSON.stringify(fixture)), id: 'draft-1', beforeLabel: 'Current', afterLabel: 'Draft', changedFiles: ['vision_models.py'],
    stale: false, before: side(before, 'class'), after: side(after, 'class'), ...(proposal ? { proposal } : {}) };
}
const cardIn = (result: ReturnType<typeof buildGraphReview>, id: string) => result.projections.diff.reviewGraph.nodes.find(node => node.id === id);
const GENERIC = 'Source changed outside the projected layers. Inspect the code diff; runtime impact is unknown.';

describe('draft review of a literal nested assignment', () => {
  it('shows the exact target as a review entry without a fabricated location, counted as modified, never added', () => {
    const result = buildGraphReview(draft({}, { [KEY]: fact('nn.GELU') }));
    expect(result.changes).toEqual([{ id: `review:after:${KEY}`, label: 'layer3[0].relu → nn.GELU', detail: 'Assignment set in source · not run · Location not verified in graph' }]);
    expect(result.counts).toMatchObject({ added: 0, removed: 0, modified: 1, unresolved: 0 });
    // The class-level fallback is replaced by the explained change, not shown beside it.
    expect(result.changes.some(change => change.detail === GENERIC)).toBe(false);
    expect(cardIn(result, `review:after:${KEY}`)).toBeUndefined();
    expect(result.focusNodes?.get(`review:after:${KEY}`)).toBe('review:root');
    for (const graph of Object.values(result.projections).map(projection => projection.reviewGraph)) {
      expect(graph.edges.some(edge => edge.source.includes(KEY) || edge.target.includes(KEY))).toBe(false);
    }
    // The source link navigates to the assignment line, not to the class or a captured layer.
    expect(result.projections.diff.spanmap[`review:after:${KEY}`]).toMatchObject({ file: 'vision_models.py', start: [10, 8] });
    expect(result.projections.before.reviewGraph.nodes.some(node => node.id.includes(KEY))).toBe(false);
  });

  it('keeps one navigable entry across a value revision, GELU → SiLU, and shows the value change', () => {
    const set = buildGraphReview(draft({}, { [KEY]: fact('nn.GELU') }));
    const revised = buildGraphReview(draft({ [KEY]: fact('nn.GELU') }, { [KEY]: fact('nn.SiLU') }));
    // Identity is the literal source path, so this is not a remove + add pair.
    expect(revised.changes).toEqual([{ id: `review:after:${KEY}`, label: 'layer3[0].relu → nn.SiLU', detail: 'Assignment value changed · not run · Location not verified in graph' }]);
    expect(revised.changes[0].id).toBe(set.changes[0].id);
    expect(revised.counts).toMatchObject({ added: 0, removed: 0, modified: 1, unresolved: 0 });
    expect(revised.nodes.get(`review:after:${KEY}`)).toMatchObject({ status: 'modified', before: KEY, after: KEY,
      params: { value: { before: 'nn.GELU()', after: 'nn.SiLU()' } } });
    expect(cardIn(revised, `review:after:${KEY}`)).toBeUndefined();
    // An unchanged assignment (same digest) is not a change at all.
    expect(buildGraphReview(draft({ [KEY]: fact('nn.GELU') }, { [KEY]: fact('nn.GELU') })).changes).toEqual([]);
  });

  it('reports a removed assignment as a source change, never a removed runtime layer', () => {
    const result = buildGraphReview(draft({ [KEY]: fact('nn.GELU') }, {}));
    expect(result.changes).toEqual([{ id: `review:before:${KEY}`, label: 'layer3[0].relu', detail: 'Assignment removed from source · not run · Location not verified in graph' }]);
    expect(result.counts).toMatchObject({ added: 0, removed: 0, modified: 1, unresolved: 0 });
    expect(cardIn(result, `review:before:${KEY}`)).toBeUndefined();
    expect(result.projections.after.reviewGraph.nodes.some(node => node.id.includes(KEY))).toBe(false);
    expect(result.projections.diff.spanmap[`review:before:${KEY}`]).toMatchObject({ start: [10, 8] });
  });

  it('marks a conditional assignment and a guard-only change without choosing a branch', () => {
    const conditional = buildGraphReview(draft({}, { [KEY]: fact('nn.GELU', { conditional: true }) }));
    expect(conditional.changes[0].detail).toBe('Assignment set in source · conditional · not run · Location not verified in graph');
    expect(cardIn(conditional, `review:after:${KEY}`)).toBeUndefined();
    const guarded = buildGraphReview(draft({ [KEY]: fact('nn.GELU') }, { [KEY]: fact('nn.GELU', { conditional: true, digest: 'guarded' }) }));
    expect(guarded.changes[0].detail).toBe('Execution condition changed · conditional · not run · Location not verified in graph');
    expect(guarded.nodes.get(`review:after:${KEY}`)?.params).toEqual({});
  });

  it('leaves an ambiguous assignment to the generic unresolved row, even beside an explained one', () => {
    const multiple = fact('Multiple assignments', { value: 'Multiple assignments', ambiguous: true });
    const alone = buildGraphReview(draft({}, { [KEY]: multiple }));
    expect(alone.changes).toEqual([{ id: 'review:root', label: 'ResNet152', detail: GENERIC }]);
    expect(alone.counts).toMatchObject({ added: 0, modified: 0, unresolved: 1 });
    expect(alone.projections.diff.reviewGraph.nodes.some(node => node.id.includes(KEY))).toBe(false);
    // Was a single GELU, now overlapping writes: still no value is picked for it.
    expect(buildGraphReview(draft({ [KEY]: fact('nn.GELU') }, { [KEY]: multiple })).changes.map(change => change.detail)).toEqual([GENERIC]);
    const other = '@assignment:layer4[0].relu';
    const mixed = buildGraphReview(draft({}, { [KEY]: multiple, [other]: fact('nn.SiLU', { target: 'layer4[0].relu' }) }));
    expect(mixed.changes.map(change => change.detail)).toEqual([GENERIC, 'Assignment set in source · not run · Location not verified in graph']);
    expect(mixed.counts).toMatchObject({ modified: 1, unresolved: 1 });
  });

  it('leaves Git source reviews unchanged: the assignment metadata adds no card or row', () => {
    const git = draft({}, { [KEY]: fact('nn.GELU') }, null);
    const result = buildGraphReview(git);
    const without = JSON.parse(JSON.stringify(git)) as GraphReviewState;
    delete reviewClass(without.after).assignments;
    const baseline = buildGraphReview(without);
    expect(result.changes).toEqual(baseline.changes);
    expect(result.counts).toEqual(baseline.counts);
    expect(result.projections.diff.reviewGraph.nodes.some(node => node.id.includes(KEY))).toBe(false);
  });

  it('never mutates the review state or adds the card to a semantic document', () => {
    const state = draft({ [KEY]: fact('nn.GELU') }, { [KEY]: fact('nn.SiLU') });
    const snapshot = JSON.stringify(state);
    const result = buildGraphReview(state);
    expect(JSON.stringify(state)).toBe(snapshot);
    for (const projection of Object.values(result.projections)) {
      expect((projection.document as GraphDocumentV2).semantic.nodes.some(node => node.node_id.includes(KEY))).toBe(false);
    }
  });
});

describe('live source view of a literal nested assignment', () => {
  const payload = (assignments: Record<string, SourceAssignmentFact>, traced = false): ProjectionPayload => {
    const projection = JSON.parse(JSON.stringify(fixture.after)) as ProjectionPayload;
    reviewClass(projection).assignments = assignments;
    return { ...projection, graphId: 'g', revision: 1, cls: 'ResNet152', file: 'vision_models.py', sourceDigest: 'a'.repeat(64),
      traced, greyed: false, viewRestored: false };
  };
  const present = (projection: ProjectionPayload, existing: Parameters<typeof graphDebugPresentation>[2] = []) =>
    graphDebugPresentation(projection, graphDocumentV2ToModelGraph(projection.document as GraphDocumentV2), existing);

  it('does not invent a module location without verified context, before or after a capture', () => {
    for (const traced of [false, true]) {
      const source = present(payload({ [KEY]: fact('nn.GELU') }, traced));
      expect(source.graph.nodes.some(node => node.data.sourceAssignment)).toBe(false);
      expect(source.graph.edges.some(edge => edge.source === KEY || edge.target === KEY)).toBe(false);
    }
  });

  it('exports the authoritative document only and leaves the payload unchanged', () => {
    const projection = payload({ [KEY]: fact('nn.GELU') });
    const snapshot = JSON.stringify(projection);
    const presented = present(projection).graph;
    expect(JSON.stringify(projection)).toBe(snapshot);
    const page = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection, syncState: 'source-only' as const,
      activeTarget: { workspaceFolder: '/workspace', cls: 'ResNet152', file: 'vision_models.py', documentVersion: 1, backendRevision: 1, sourceDigest: projection.sourceDigest } };
    const exported = exportCurrentGraphFile(presented, page, projection);
    expect(exported.document.semantic.nodes.map(node => node.node_id).sort()).toEqual(['@inherited-forward', 'fc2', 'input', 'output']);
    expect(JSON.stringify(exported.document.semantic.nodes)).not.toContain('sourceAssignment');
  });
});
