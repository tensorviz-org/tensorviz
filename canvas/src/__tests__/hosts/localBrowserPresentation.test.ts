import {
  graphOpeningPresentation,
  diagnosticSummary,
  localBrowserSyncLabel,
  orderedDiagnostics,
  projectionForCanvas,
} from '../../ide/localBrowserPresentation';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE, type LocalBrowserPageState } from '../../ide/localBrowserState';
import { IdeProjectionDiagnostic, ProjectionPayload } from '../../ide/protocol.generated';

test('a failed capture names the layer instead of hiding the detail', () => {
  const graph = projection();
  graph.traced = true;
  (graph.document as any).semantic.metadata.capture = { failureNodeId: 'head' };
  (graph.document as any).semantic.nodes.find((node: { node_id: string }) => node.node_id === 'head').params._module_path = 'net.head';
  const page: LocalBrowserPageState = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: graph, syncState: 'runtime-traced',
    opening: { id: 'run', phase: 'failed', hasCapture: true, detail: 'expected [?, 10], received [2, 8].', failure: { kind: 'execution' },
      target: { workspaceFolder: '/project', file: 'models/net.py', cls: 'Net', sourceDigest: 'root-digest', documentVersion: 1, backendRevision: 1 } } };
  expect(localBrowserSyncLabel(page)).toEqual({ tone: 'warn', text: 'Capture stopped at net.head · expected [?, 10], received [2, 8].' });
});

test('a JAX capture labels inferred shapes even without a setup draft', () => {
  const graph = projection();
  graph.document = { semantic: { metadata: { capture: { backend: 'jaxpr' } } } };
  expect(localBrowserSyncLabel({ ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: graph, syncState: 'runtime-traced' }))
    .toEqual({ tone: 'ok', text: 'JAX traced program — shapes inferred' });
});

test.each(['opaque_kernel', 'byte_limit'])('partial JAX coverage stays a warning with inferred shapes (%s)', omission => {
  const graph = projection();
  graph.document = { semantic: { metadata: { capture: { backend: 'jaxpr', status: 'partial', omissions: [omission] } } } };
  expect(localBrowserSyncLabel({ ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: graph, syncState: 'runtime-traced' }))
    .toEqual({ tone: 'warn', text: 'Partial JAX program — shapes inferred' });
});

test('environment review keeps its cancellation and progress panel visible over a source preview', () => {
  const target = { workspaceFolder: '/project', file: 'models/net.py', cls: 'Net', sourceDigest: 'root-digest', documentVersion: 1, backendRevision: 1 };
  const opening = { id: 'opening', target, phase: 'resolving' as const, detail: 'Preparing the reviewed library repair.', hasCapture: false,
    environment: { reason: 'missing_libraries' as const, candidates: [], requirements: ['scipy'], detail: 'Missing SciPy.' } };
  const page = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: projection(), opening };
  expect(graphOpeningPresentation(page).compact).toBe(false);
  expect(graphOpeningPresentation({ ...page, opening: { ...opening, environment: { ...opening.environment, reason: 'ready' } } }).compact).toBe(true);
});

const diagnostics: IdeProjectionDiagnostic[] = [
  {
    code: 'UNRESOLVED_DYNAMIC',
    severity: 'warning',
    file: 'models/dynamic.py',
    line: 9,
    import_text: 'self.inner = block_cls()',
    detail: 'The class is selected dynamically, so this instance remains opaque.',
    node: 'dyn.inner',
  },
  {
    code: 'DEPENDENCY_SYNTAX_ERROR',
    severity: 'error',
    file: 'models/head.py',
    line: 4,
    import_text: '',
    detail: 'This dependency does not parse.',
    node: 'head',
  },
];

function projection(): ProjectionPayload {
  return {
    graphId: 'g',
    revision: 1,
    cls: 'Net',
    file: 'models/net.py',
    sourceDigest: 'root-digest',
    traced: false,
    greyed: false,
    viewRestored: false,
    spanmap: {},
    diagnostics,
    document: {
      schemaVersion: 2,
      semantic: {
        graph_id: 'g', name: 'Net', revision: 1, inputs: [], edges: [], modules: [], operations: [], metadata: {},
        nodes: [
          { node_id: 'dyn.inner', kind: 'module', op: 'custom_code', params: {}, in_ports: [], out_ports: [], metadata: {} },
          { node_id: 'head', kind: 'module', op: 'custom_code', params: {}, in_ports: [], out_ports: [], metadata: {} },
        ],
      },
      view: {
        graph_id: 'g', node_positions: {}, collapsed_modules: [], selection: [],
        layout: { node_data: { 'dyn.inner': { expanded: false } } },
      },
    },
  };
}

describe('multi-file local-browser presentation', () => {
  it.each([
    [['byte_limit'], 'Partial capture — recording size limit reached'],
    [['operation_limit'], 'Partial capture — some execution details are missing'],
  ])('does not present a bounded partial capture as a complete trace (%s)', (omissions, text) => {
    const captured = { ...projection(), traced: true };
    (captured.document as any).semantic.metadata.capture = { status: 'partial', omissions };
    const page: LocalBrowserPageState = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE,
      projection: captured, syncState: 'runtime-traced' };
    expect(localBrowserSyncLabel(page)).toEqual({ tone: 'warn', text });
    expect(localBrowserSyncLabel({ ...page, syncState: 'stale', reason: 'Source changed' })?.text)
      .toBe('Stale — Source changed');
    (captured.document as any).semantic.metadata.capture = { status: 'complete', omissions: [] };
    expect(localBrowserSyncLabel(page)).toEqual({ tone: 'ok', text: 'Traced — shapes measured' });
  });

  it.each([false, true])('uses the exact host source span for clickable node locations (traced=%s)', traced => {
    const original = { ...projection(), traced, diagnostics: [], spanmap: {
      'dyn.inner': { file: 'models/block.py', start: [42, 4] },
      head: { file: 'models/head.py', start: [0, 0] },
    } };
    const nodes = (projectionForCanvas(original).document as any).semantic.nodes;
    expect(nodes[0].metadata.sourceLocation).toEqual({ file: 'models/block.py', line: 42 });
    expect(nodes[1].metadata.sourceLocation).toBeUndefined();
    expect((original.document as any).semantic.nodes[0].metadata).toEqual({});
  });
  it('orders errors before warnings and summarizes every severity honestly', () => {
    expect(orderedDiagnostics(diagnostics).map((item) => item.code)).toEqual([
      'DEPENDENCY_SYNTAX_ERROR',
      'UNRESOLVED_DYNAMIC',
    ]);
    expect(diagnosticSummary(diagnostics)).toBe('1 error · 1 warning');
  });

  it('annotates the affected canvas cards without mutating the backend document', () => {
    const original = projection();
    const presented = projectionForCanvas(original);
    const document = presented.document as any;

    expect(presented).not.toBe(original);
    expect((original.document as any).semantic.nodes[0].metadata).toEqual({});
    expect(document.semantic.nodes[0].metadata).toMatchObject({
      resolutionCode: 'UNRESOLVED_DYNAMIC',
      resolutionSeverity: 'warning',
      resolutionDetail: expect.stringContaining('remains opaque'),
    });
    expect(document.view.layout.node_data['dyn.inner']).toMatchObject({
      expanded: false,
      moduleStatus: 'warning',
      resolutionCode: 'UNRESOLVED_DYNAMIC',
    });
    expect(document.view.layout.node_data.head.moduleStatus).toBe('error');
  });

  it('says a current source graph after an edit was not run, instead of only that source changed', () => {
    const target = { workspaceFolder: '/w', cls: 'Net', file: 'models/net.py', documentVersion: 2, backendRevision: 2, sourceDigest: 'b' };
    const page: LocalBrowserPageState = { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, activeTarget: target,
      projection: projection(), syncState: 'source-only',
      opening: { id: 'o', target, phase: 'cancelled', cancellationReason: 'source_changed', hasCapture: false,
        detail: 'Source changed. Refresh the saved model to capture it.' } };
    expect(localBrowserSyncLabel(page)).toEqual({ tone: 'muted', text: 'Not run — shapes read from the source' });
    // Without a current source graph the cancellation remains the explanation.
    expect(localBrowserSyncLabel({ ...page, syncState: 'stale', greyed: true })?.text).toMatch(/^Source changed/);
    expect(localBrowserSyncLabel({ ...page, opening: { ...page.opening!, cancellationReason: undefined } })?.text)
      .toBe('Not run — shapes read from the source');
  });
});
