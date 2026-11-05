import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { ModuleShapeUtil, statusLabelForModule } from '../../canvas/ModuleShapeUtil';
import { reconcileModuleView } from '../../services/canvasLayout';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { initialGraphFileView, restoreGraphFileView } from '../../viewer/graphFileView';
import { modelGraphToDocumentV2, graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { CapturePort, ModelGraph, ModelNodeData } from '../../types';
import { moduleCardHeader, moduleCardHeight, moduleCardWidth } from '../../canvas/cardMetrics';
import { NodeUnderstanding } from '../../canvas/NodeUnderstanding';
import captureDocument from '../../test/fixtures/canvas/capture-document.json';
import { currentProjection, reportProjectExpand, reportNodeClick, showingProjectGraph } from '../../ide/projection';


jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: any) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: any) => component,
}));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
let mockDetail = 'inspection';
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => mockDetail }));
afterEach(() => { mockDetail = 'inspection'; });
jest.mock('../../canvas/graphAccess', () => ({ graphIsReadOnly: () => true }));
jest.mock('../../ide/projection', () => ({ localBrowserMode: () => true,
  currentProjection: jest.fn(() => ({ spanmap: { 'encoder.layers.0': {} } })),
  showingProjectGraph: jest.fn(() => false), reportProjectExpand: jest.fn(), reportNodeClick: jest.fn() }));

// F3's actual depth-2 trace topology: one repeated boundary, no invented members.
const graph = (): ModelGraph => ({
  nodes: [
    { id: 'encoder', type: 'moduleNode', position: { x: 40, y: 40 }, data: {
      layerType: { id: 'transformerencoder', name: 'TransformerEncoder' },
      parameters: { _shape_provenance: 'observed', _out_shape: '1x10x32' },
      moduleInstanceId: 'encoder', moduleLayerCount: 1, expanded: true, isGroup: true,
    } },
    { id: 'encoder.layers.0', type: 'moduleNode', position: { x: 24, y: 64 }, data: {
      layerType: { id: 'custom_code', name: 'custom_code' },
      parameters: { _class: 'CustomBlock', _import_path: 'models.custom', _module_path: 'encoder.layers.0',
        _shape_provenance: 'observed', _in_shape: '1x10x32', _out_shape: '1x10x32', _repeat_count: 2, _repeat_reason: 'module-list' },
      inputShape: [1, 10, 32], outputShape: [1, 10, 32], shapeEvidence: 'measured',
      moduleInstanceId: 'encoder.layers.0', moduleTemplateId: 'CustomBlock x2', isGroup: true,
      moduleLayerCount: 0, moduleCapture: { status: 'depth-limited', maxDepth: 2 },
      parentId: 'encoder', repeatCount: 2, expanded: true,
    } },
  ] as ModelGraph['nodes'],
  edges: [],
});

function renderCard(data: ModelNodeData) {
  const util = new ModuleShapeUtil();
  (util as any).editor = { getSelectedShapeIds: () => [], getHoveredShapeId: () => null };
  return render(util.component({ id: 'shape:encoder.layers.0', props: { nodeData: data, w: moduleCardWidth(data), h: moduleCardHeight(data) } } as any));
}

function inspectParameters(data: ModelNodeData) {
  render(<NodeUnderstanding node={{ id: 'encoder.layers.0', type: 'moduleNode', position: { x: 0, y: 0 }, data }} />);
  fireEvent.click(screen.getByText('More details'));
  return within(screen.getByRole('region', { name: 'Parameter values' }));
}

const boundaryPort = (direction: 'in' | 'out', extra: Partial<CapturePort> = {}): CapturePort => ({
  ...captureDocument.semantic.nodes.flatMap(node => node.metadata.capturePorts ?? [])[0],
  id: direction, direction, argumentKind: 'value', shape: [1, 10, 32], dtype: 'torch.float32',
  displayName: null,
  paths: direction === 'in' ? [['inputs', 'args', 0]] : [['output']], ...extra,
} as CapturePort);

const capturedModule = (): ModelNodeData => ({ ...graph().nodes[1].data, expanded: false,
  repeatCount: 0, moduleCapture: undefined, moduleLayerCount: 2, expandable: true,
  captureRef: { captureId: 'run', kind: 'invocation', invocationId: 'call:fc' } as ModelNodeData['captureRef'],
  capturedOperationCount: 2, capturePorts: [boundaryPort('in'), boundaryPort('out')],
  sourceLocation: { file: 'models/blocks.py', line: 14 },
});

it.each(['expanded', 'collapsed', 'passthrough'])('labels rewired %s modules without claiming their source changed', mode => {
  const { container } = renderCard({ ...capturedModule(), expanded: mode === 'expanded', capturePassThrough: mode === 'passthrough',
    diffStatus: 'rewired', review: { status: 'rewired', params: {}, inputChanges: { added: 1, removed: 1 },
      summary: 'Input connections changed (+1 / −1). Source and parameters unchanged.' } });
  expect(screen.getByLabelText('Graph change: inputs changed')).toBeVisible();
  expect(container.querySelector('.diff-rewired')).not.toBeNull();
  expect(screen.queryByText('Input unchanged')).not.toBeInTheDocument();
});

it.each(['overview', 'working', 'inspection'])('keeps module shapes visible at %s zoom without routine capture badges', detail => {
  mockDetail = detail;
  renderCard(capturedModule());
  expect(within(screen.getByRole('group', { name: 'Input and output shapes' }))
    .getAllByTitle('(1, 10, 32) · float32')).toHaveLength(2);
  expect(screen.queryByText(/captured operations?|Measured boundary/)).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Expand encoder.layers.0' })).toBeEnabled();
  expect(screen.queryByRole('button', { name: 'Expand Block' })).not.toBeInTheDocument();
});

it('expanded headers omit routine capture badges', () => {
  renderCard({ ...capturedModule(), expanded: true });
  expect(screen.queryByText(/captured operations?|Measured boundary/)).not.toBeInTheDocument();
});

it('sizes captured cards for their content and reserves extra room for warnings and multiple ports', () => {
  const data = capturedModule();
  const height = moduleCardHeight(data);
  // Complete identity, body padding, wrapped tensor values and source.
  expect(height).toBe(moduleCardHeader(data).height + 132);
  expect(moduleCardHeight({ ...data, shapeEvidence: 'stale' })).toBeGreaterThan(height);
  expect(moduleCardHeight({ ...data, shapeEvidence: 'unknown' })).toBe(height);
  const ports = Array.from({ length: 12 }, (_, i) => boundaryPort(i < 6 ? 'in' : 'out', { id: `port-${i}` }));
  // Unnamed interfaces use concise labels; full values wrap onto two lines.
  expect(moduleCardHeight({ ...data, capturePorts: ports })).toBe(height - 2 * 44 + 12 * 44);
});

it.each([false, true])('leaves static-analysis notes to the source-notes list (expanded=%s)', expanded => {
  const note = 'transformers is external to the workspace; its classes stay opaque on the static graph unless they are canonical framework operations';
  const plain = { ...capturedModule(), expanded, parameters: { ...capturedModule().parameters, _defined_in: 'models/blocks.py' } };
  const noted = { ...plain, resolutionCode: 'EXTERNAL_PACKAGE', resolutionSeverity: 'info' as const, resolutionDetail: note };
  const { container } = renderCard(noted);
  expect(container).not.toHaveTextContent('external to the workspace');
  expect(screen.getByText('defined in models/blocks.py')).toBeVisible();
  expect(moduleCardHeight(noted)).toBe(moduleCardHeight(plain));
});

it('shows readable names and reserves space for every boundary value', () => {
  const ports = [
    boundaryPort('in', { id: 'x', paths: [['x']] }),
    boundaryPort('in', { id: 'tokens', paths: [['context', 'tokens']] }),
    boundaryPort('out', { id: 'y', paths: [['y']] }),
    boundaryPort('out', { id: 'states', paths: [['hidden', 'states']] }),
  ];
  const data = { ...capturedModule(), capturePorts: ports };
  expect(moduleCardHeight(data)).toBe(moduleCardHeader(data).height + 212);
  renderCard(data);
  const rows = within(screen.getByRole('group', { name: 'Input and output shapes' }));
  for (const [label, path] of [['In x', 'x'], ['In context.tokens', 'context.tokens'], ['Out y', 'y'], ['Out hidden.states', 'hidden.states']]) {
    expect(rows.getByText(label)).toHaveAttribute('title', path);
  }
  expect(rows.getAllByTitle('(1, 10, 32) · float32')).toHaveLength(4);
  expect(data.capturePorts).toEqual(ports);
});

it('does not invent constructor arguments from capture metadata or tensor dimensions', () => {
  renderCard(capturedModule());
  expect(screen.queryByLabelText('Constructor arguments')).not.toBeInTheDocument();
  expect(screen.queryByText('in features')).not.toBeInTheDocument();
  expect(screen.queryByText('out features')).not.toBeInTheDocument();
});

it('keeps actual constructor values and unevaluated source expressions in selection details', () => {
  const data = { ...capturedModule(), parameters: { ...capturedModule().parameters,
    in_features: 8, out_features: 16, bias: false, activation: null,
    _source_arguments: { out_features: 'cfg.width' } } };
  renderCard(data);
  expect(screen.queryByLabelText('Constructor arguments')).not.toBeInTheDocument();
  const values = inspectParameters(data);
  expect(values.getByText('in_features').parentElement).toHaveTextContent('in_features8');
  expect(values.getByText('out_features').parentElement).toHaveTextContent('out_featurescfg.width');
  expect(values.getByText('bias').parentElement).toHaveTextContent('biasfalse');
  expect(values.getByText('activation').parentElement).toHaveTextContent('activationnull');
  expect(values.queryByText('_shape_provenance')).not.toBeInTheDocument();
});

it('retains a source constructor argument named name rather than treating it as a display label', () => {
  const data = { ...capturedModule(), moduleTemplateId: 'NamedBlock', parameters: {
    ...capturedModule().parameters, _source_arguments: { name: "'stem'" } } };
  renderCard(data);
  expect(screen.queryByLabelText('Constructor arguments')).not.toBeInTheDocument();
  const values = inspectParameters(data);
  expect(values.getByText('name').parentElement).toHaveTextContent("name'stem'");
});

it('keeps every source argument available in selection without adding an argument grid to the card', () => {
  const parameters = Object.fromEntries(Array.from({ length: 7 }, (_, i) => [`arg_${i}`, i + 1]));
  const data = { ...capturedModule(), parameters: { ...capturedModule().parameters, ...parameters } };
  renderCard(data);
  expect(screen.queryByLabelText('Constructor arguments')).not.toBeInTheDocument();
  const values = inspectParameters(data);
  for (const [key, value] of Object.entries(parameters)) expect(values.getByText(key).parentElement).toHaveTextContent(`${key}${value}`);
  expect(values.queryByText(/more/)).not.toBeInTheDocument();
  expect(data.parameters).toMatchObject(parameters);
});

it('retains the bounded constructor comparison grid on review cards', () => {
  const parameters = Object.fromEntries(Array.from({ length: 7 }, (_, i) => [`arg_${i}`, i + 1]));
  renderCard({ ...capturedModule(), parameters: { ...capturedModule().parameters, ...parameters },
    review: { status: 'modified', params: { arg_0: { before: 0, after: 1 } } } });
  const grid = screen.getByLabelText('Constructor arguments');
  expect(grid.children).toHaveLength(4);
  expect(within(grid).getByText('+4 more')).toBeInTheDocument();
  for (const key of ['arg_0', 'arg_1', 'arg_2']) expect(within(grid).getByTitle(key)).toBeInTheDocument();
  expect(within(grid).queryByTitle('arg_3')).not.toBeInTheDocument();
  expect(grid.querySelector('del')).toHaveTextContent('0');
  expect(grid.querySelector('ins')).toHaveTextContent('1');
});

it.each([[1, 28], [2, 28], [3, 48], [4, 48], [7, 48]])(
  'keeps %i arguments out of ordinary card geometry while reserving their review rows',
  (count, addedHeight) => {
    const data = capturedModule();
    const parameters = Object.fromEntries(Array.from({ length: count }, (_, i) => [`arg_${i}`, i]));
    expect(moduleCardHeight({ ...data, parameters: { ...data.parameters, ...parameters } }))
      .toBe(moduleCardHeight(data));
    const review = { status: 'modified' as const, params: {} };
    expect(moduleCardHeight({ ...data, review, parameters: { ...data.parameters, ...parameters } }))
      .toBe(moduleCardHeight({ ...data, review }) + addedHeight);
  },
);

it('keeps issue evidence visible while constructor details move to selection; older timing adds no row', () => {
  const data = { ...capturedModule(), parameters: { ...capturedModule().parameters, in_features: 8, out_features: 16 } };
  const diagnostic: ModelNodeData['moduleDiagnostics'] = [{ id: 'shape', nodeId: 'encoder.layers.0',
    label: 'CustomBlock', severity: 'error', message: 'Expected the recorded input width.' }];
  const warning = { ...data, moduleDiagnosticStatus: 'error' as const, moduleDiagnostics: diagnostic };
  const timed = { ...warning, moduleLatency: { cpuTotalUs: [10, 20, 30], cpuSelfUs: [1, 2, 3],
    deviceTotalUs: null, deviceMethod: null } };
  expect(moduleCardHeight(data)).toBe(moduleCardHeight(capturedModule()));
  expect(moduleCardHeight(warning)).toBe(moduleCardHeight(data) + 35);
  expect(moduleCardHeight(timed)).toBe(moduleCardHeight(warning));
  renderCard(timed);
  expect(screen.queryByLabelText('Constructor arguments')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: '1 error in encoder.layers.0' })).toBeInTheDocument();
  expect(screen.queryByLabelText(/Module latency/)).toBeNull();
});

it('reserves a current-run change note alongside constructor values and measured boundaries', () => {
  const data = { ...capturedModule(), parameters: { ...capturedModule().parameters, in_features: 8, out_features: 16 } };
  const changed = { ...data, debugEvidence: 'current' as const, debugChanges: ['Output values changed'] };
  expect(moduleCardHeight(changed)).toBe(moduleCardHeight(data) + 37);
  renderCard(changed);
  expect(screen.getByText('Changed')).toBeInTheDocument();
  expect(screen.getByText('Output values changed')).toBeInTheDocument();
  expect(screen.queryByLabelText('Constructor arguments')).not.toBeInTheDocument();
  expect(screen.getByRole('group', { name: 'Input and output shapes' })).toBeInTheDocument();
});

it('does not reserve a hidden change note from previous-run evidence', () => {
  const data = { ...capturedModule(), debugEvidence: 'previous' as const };
  const changed = { ...data, debugChanges: ['Output values changed'] };
  expect(moduleCardHeight(changed)).toBe(moduleCardHeight(data));
  renderCard(changed);
  expect(screen.queryByText('Changed')).not.toBeInTheDocument();
  expect(screen.getByText('Previous run')).toBeInTheDocument();
});

it('reserves routed display-port spacing while retaining only the call\'s own boundary rows', () => {
  const data = { ...capturedModule(), captureDisplayPorts: Array.from({ length: 40 }, (_, i) =>
    boundaryPort('in', { id: `child-${i}`, paths: [['child', i]] })) };
  expect(moduleCardHeight(data)).toBe(688);
  renderCard(data);
  const rows = screen.getByRole('group', { name: 'Input and output shapes' });
  expect(rows.children).toHaveLength(2);
  expect(within(rows).getByText('In')).toBeInTheDocument();
  expect(within(rows).getByText('Out')).toBeInTheDocument();
  expect(data.capturePorts).toEqual(capturedModule().capturePorts);
});

it('shows captured module inputs and outputs instead of cached or routed child shapes', () => {
  renderCard({ ...graph().nodes[1].data, expanded: false,
    inputShape: [999], outputShape: [999],
    capturePorts: [boundaryPort('in'), boundaryPort('out')],
    captureDisplayPorts: [boundaryPort('in', { id: 'child', shape: [777] })],
  });
  const rows = within(screen.getByRole('group', { name: 'Input and output shapes' }));
  expect(rows.getByText('In')).toBeInTheDocument();
  expect(rows.getByText('Out')).toBeInTheDocument();
  expect(rows.getAllByTitle('(1, 10, 32) · float32')).toHaveLength(2);
  expect(rows.queryByText(/999|777/)).not.toBeInTheDocument();
});

it('preserves every boundary value, including scalar, unknown and non-tensor outputs', () => {
  renderCard({ ...graph().nodes[1].data, expanded: false, capturePorts: [
    boundaryPort('in'),
    boundaryPort('in', { id: 'tokens', paths: [['inputs', 'kwargs', 'tokens']], shape: [1, 4], dtype: 'torch.int64' }),
    boundaryPort('out', { id: 'scalar', paths: [['output', 0]], shape: [] }),
    boundaryPort('out', { id: 'unknown', paths: [['output', 1]], shape: null, dtype: null }),
    boundaryPort('out', { id: 'literal', paths: [['output', 2]], argumentKind: 'int', valueId: null, shape: null, dtype: null }),
  ] });
  const rows = within(screen.getByRole('group', { name: 'Input and output shapes' }));
  expect(rows.getByText('In 1')).toHaveAttribute('title', 'Input 1');
  expect(rows.getByText('In tokens')).toHaveAttribute('title', 'tokens');
  expect(rows.getByText('Out 1')).toHaveAttribute('title', 'Output 1');
  expect(rows.getByTitle('(1, 4) · int64')).toBeInTheDocument();
  expect(rows.getByTitle('() · float32')).toBeInTheDocument();
  expect(rows.getByTitle('shape unknown · dtype unknown')).toBeInTheDocument();
  expect(rows.getByText('int')).toBeInTheDocument();
});

it('keeps stale legacy boundary shapes labelled and respects missing captured ports', () => {
  const view = renderCard({ ...graph().nodes[1].data, expanded: false,
    inputShape: [999], outputShape: [999], shapeEvidence: 'stale' });
  expect(screen.getByText('Stale shapes')).toBeInTheDocument();
  expect(within(screen.getByRole('group', { name: 'Input and output shapes' })).getAllByText('(1, 10, 32)')).toHaveLength(2);
  view.unmount();
  renderCard({ ...graph().nodes[1].data, expanded: false, capturePorts: [] });
  expect(screen.queryByRole('group', { name: 'Input and output shapes' })).not.toBeInTheDocument();
});

it('reserves room for every module boundary row in nested layout without changing semantic ports', () => {
  const source = graph();
  const ports = Array.from({ length: 12 }, (_, i) => boundaryPort(i < 6 ? 'in' : 'out', {
    id: `port-${i}`, paths: [['values', i]],
  }));
  source.nodes[1].data.capturePorts = ports;
  const view = reconcileModuleView(source.nodes, source.edges);
  const child = view.nodes.find(node => node.id === source.nodes[1].id)!;
  const parent = view.nodes.find(node => node.id === source.nodes[0].id)!;
  // Each numbered label/value row takes 22px, beyond the 80px header and body frame.
  expect(Number(child.style!.height)).toBeGreaterThanOrEqual(80 + ports.length * 22);
  expect(Number(parent.style!.height)).toBeGreaterThan(Number(child.position.y) + Number(child.style!.height));
  expect(child.data.capturePorts).toEqual(ports);
});

test('an unchanged call has a compact contextual name, source navigation and no unavailable-contents warning', () => {
  (currentProjection as jest.Mock).mockReturnValue({ spanmap: { 'encoder.layers.0': {} } });
  const data = { ...graph().nodes[1].data, moduleTemplateId: 'Identity', captureLabel: 'blocks.0.ls1',
    capturePassThrough: { valueId: 'v', inputPortId: 'in', sourceNodeId: 'producer', sourcePortId: 'out' } };
  renderCard(data);
  expect(screen.getByTitle('Identity')).toHaveTextContent('Identity');
  expect(screen.getByText('blocks.0.ls1')).toBeInTheDocument();
  expect(screen.getByText('Input unchanged')).toBeInTheDocument();
  expect(screen.queryByText('Contents unavailable')).not.toBeInTheDocument();
  fireEvent.pointerDown(screen.getByRole('button', { name: 'Open source: blocks.0.ls1' }));
  expect(reportNodeClick).toHaveBeenCalledWith('encoder.layers.0');
});

test('an unmapped captured call keeps its path without promising source navigation', () => {
  (currentProjection as jest.Mock).mockReturnValueOnce({ spanmap: {} });
  renderCard({ ...graph().nodes[1].data, moduleTemplateId: 'Identity', captureLabel: 'blocks.0.ls1',
    capturePassThrough: { valueId: 'v', inputPortId: 'in', sourceNodeId: 'producer', sourcePortId: 'out' } });
  expect(screen.getByTitle('blocks.0.ls1 · Source location not mapped')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'ls1 · Identity' })).not.toBeInTheDocument();
});

it.each([false, true])(
  'shows declared ModuleList members without opening an empty frame (raw parser=%s)', raw => {
  const onToggleExpand = jest.fn();
  renderCard({ layerType: { id: 'modulelist', name: 'ModuleList' },
    parameters: { _shape_provenance: 'static', _class: 'ModuleList', _opaque: true,
      _modules: [{ op: 'Linear', params: {} }, { op: 'Linear', params: {} }] },
    ...(!raw ? { moduleInstanceId: 'blocks', moduleTemplateId: 'ModuleList', moduleLayerCount: 0, moduleMembers: [] } : {}),
    expanded: true, onToggleExpand } as ModelNodeData);
  expect(screen.getByText('ModuleList')).toBeInTheDocument();
  if (raw) expect(screen.queryByTitle('encoder.layers.0')).not.toBeInTheDocument();
  else expect(screen.getByTitle('blocks')).toHaveTextContent('blocks');
  expect(screen.getByText('Declared: 2 × Linear')).toBeInTheDocument();
  expect(screen.queryByText('Source-only')).not.toBeInTheDocument();
  expect(screen.getByRole('img', { name: 'Read from code; tensor shapes have not been recorded.' })).toBeInTheDocument();
  expect(screen.queryByText(/captured layers/)).toBeNull();
  expect(screen.queryByRole('button', { name: `Expand ${raw ? 'encoder.layers.0' : 'blocks'}` })).not.toBeInTheDocument();
  expect(onToggleExpand).not.toHaveBeenCalled();
  expect(reportProjectExpand).not.toHaveBeenCalled();
});

it('renders the measured bounded card and refuses empty expansion', () => {
  const onToggleExpand = jest.fn();
  renderCard({ ...graph().nodes[1].data, onToggleExpand });
  expect(screen.getByText('Depth-limited')).toBeInTheDocument();
  expect(screen.queryByText('Measured boundary')).not.toBeInTheDocument();
  expect(screen.queryByText(/Internal trace contents not captured at depth 2/)).not.toBeInTheDocument();
  expect(screen.queryByText(/0 layers/)).toBeNull();
  expect(screen.queryByText('Source-only')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Expand encoder.layers.0' })).not.toBeInTheDocument();
  expect(onToggleExpand).not.toHaveBeenCalled();
  expect(reportProjectExpand).not.toHaveBeenCalled();
});

it('still expands real captured contents as a local view', () => {
  const onToggleExpand = jest.fn();
  renderCard({ ...graph().nodes[0].data, expanded: false, onToggleExpand });
  expect(screen.queryByText('Measured boundary')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Expand Block' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Expand encoder' }));
  expect(onToggleExpand).toHaveBeenCalledWith('encoder');
});

it.each([
  [{ parameters: {} }, ''],
  [{ parameters: { _shape_provenance: 'unreached', _out_shape: '1x8' }, shapeEvidence: 'measured' }, 'Not observed'],
  [{ parameters: { _shape_provenance: 'declared', _out_shape: '1x8' } }, 'Declared shapes'],
  [{ parameters: { _shape_provenance: 'observed', _out_shape: '1x8' }, shapeEvidence: 'stale' }, 'Stale shapes'],
  [{ parameters: { _shape_provenance: 'observed' } }, 'Shapes unavailable'],
  [{ parameters: { _shape_provenance: 'observed', _in_shape: '1x?', _out_shape: [[1, 2], [1, 3]] } }, 'Shapes unavailable'],
] as const)('labels only node-local evidence: %s', (data, expected) => {
  expect(statusLabelForModule('ok', data as ModelNodeData)).toBe(expected);
});

it('keeps warnings alongside measured evidence without certifying the module', () => {
  expect(statusLabelForModule('warning', graph().nodes[1].data)).toBe('Warnings');
});

it('keeps a source-only empty ModuleList compact without unavailable badges or dead controls', () => {
  const data = { layerType: { id: 'modulelist', name: 'ModuleList' }, parameters: {}, moduleLayerCount: 0,
    moduleTemplateId: 'ModuleList', debugEvidence: 'source' } as ModelNodeData;
  const { container } = renderCard(data);
  expect(container).not.toHaveTextContent(/Contents unavailable|Source-only|Source only|Collapsed block view|No internal contents/);
  expect(screen.queryByRole('button', { name: /Expand|Reset module layout/ })).not.toBeInTheDocument();
  expect(screen.getByRole('img', { name: 'Read from code; tensor shapes have not been recorded.' })).toBeVisible();
  expect(moduleCardHeight(data)).toBe(moduleCardHeader(data).height + 2);
});

it('grows for the full source filename and complete long boundary values', () => {
  (currentProjection as jest.Mock).mockReset();
  (currentProjection as jest.Mock).mockReturnValue({ spanmap: { 'encoder.layers.0': {} } });
  const original = capturedModule();
  const file = `models/${'long_source_filename_'.repeat(8)}.py`;
  const data = { ...original, sourceLocation: { file, line: 127 }, capturePorts: [
    boundaryPort('in', { paths: [['inputs', 'kwargs', 'audio_representations_for_cross_attention']],
      shape: Array.from({ length: 20 }, (_, index) => index + 1) }),
  ] };
  expect(moduleCardHeight(data)).toBeGreaterThan(moduleCardHeight(original));
  const { container } = renderCard(data);
  const source = screen.getByRole('button', { name: `${file}:127` });
  expect(source.textContent).toBe(`${file.split('/').pop()}:127`);
  expect(parseFloat(source.style.height)).toBeGreaterThan(18);
  const value = container.querySelector('.module-boundary-value')!;
  expect(value.textContent).toBe(`(${data.capturePorts[0].shape!.join(', ')}) · float32`);
  expect(container.querySelector('.module-boundary-row')!.getAttribute('style')).not.toContain('36px');
});

it('round trips bounds, repeats, custom Python identity and real topology; restored open state cannot expand nothing', () => {
  const source = graph();
  source.nodes.push({ id: 'head', type: 'layerNode', position: { x: 800, y: 40 }, data: {
    layerType: source.nodes[0].data.layerType, parameters: {},
  } });
  source.edges.push({ id: 'observed-edge', source: 'encoder.layers.0', target: 'head' });
  const before = JSON.stringify(source);
  const document = modelGraphToDocumentV2(source, { name: 'Net', revision: 1 });
  expect(graphDocumentV2ToModelGraph(document).nodes[1].data).toMatchObject({
    moduleCapture: source.nodes[1].data.moduleCapture, parameters: source.nodes[1].data.parameters,
  });
  const file = parseGraphFile(JSON.stringify(createGraphFile(source, { name: 'Net', status: 'traced' })));
  const view = initialGraphFileView(file);
  expect(view.nodes.map(n => n.id).sort()).toEqual(source.nodes.map(n => n.id).sort());
  expect(view.edges.map(e => [e.source, e.target])).toEqual([['encoder.layers.0', 'head']]);
  const block = view.nodes.find(n => n.id === 'encoder.layers.0')!;
  expect(view.nodes[0].data.expanded).toBe(true);
  expect(block.data).toMatchObject({ expanded: false, expandable: false, repeatCount: 2,
    moduleCapture: { status: 'depth-limited', maxDepth: 2 }, moduleTemplateId: 'CustomBlock x2' });
  // Portable snapshots retain the label/evidence, not executable source/import paths.
  expect(block.data.parameters).toEqual({});
  expect(statusLabelForModule('ok', block.data)).toBe('');
  const restored = restoreGraphFileView(view, { expandedModuleIds: ['encoder', 'encoder.layers.0'],
    selectedNodeIds: [], nodePositions: {} } as any);
  expect(restored.nodes.find(n => n.id === block.id)?.data.expanded).toBe(false);
  expect(restored.edges.map(e => [e.source, e.target])).toEqual([['encoder.layers.0', 'head']]);
  expect(JSON.stringify(source)).toBe(before);
});

it('keeps legacy empty traces bounded without guessing their capture depth', () => {
  const source = graph();
  delete source.nodes[1].data.moduleCapture;
  const block = reconcileModuleView(source.nodes, source.edges).nodes[1];
  expect(block.data.expanded).toBe(false);
  expect(block.data.moduleCapture).toBeUndefined();
  renderCard(block.data);
  expect(screen.queryByText('No internal contents are available in this graph.')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Expand encoder.layers.0' })).not.toBeInTheDocument();
});

it('preserves closing an empty host-owned project frame', () => {
  (showingProjectGraph as jest.Mock).mockReturnValue(true);
  try {
    const source = graph().nodes[0];
    source.data.parameters = { _kind: 'model' };
    source.data.moduleLayerCount = 0;
    const frame = reconcileModuleView([source], []).nodes[0];
    expect(frame.data.expanded).toBe(true);
    renderCard(frame.data);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse encoder' }));
    expect(reportProjectExpand).toHaveBeenCalledWith(expect.any(String), false);
  } finally {
    (showingProjectGraph as jest.Mock).mockReturnValue(false);
    (reportProjectExpand as jest.Mock).mockClear();
  }
});

it('allows materialized static children under a bounded boundary without claiming they were measured', () => {
  const source = graph();
  source.nodes.push({ id: 'encoder.layers.0.fc', type: 'layerNode', position: { x: 24, y: 64 }, data: {
    layerType: source.nodes[0].data.layerType, parameters: { _shape_provenance: 'unreached' }, parentId: 'encoder.layers.0',
  } });
  const view = reconcileModuleView(source.nodes, source.edges);
  expect(view.nodes[1].data.expanded).toBe(true);
  expect(view.nodes[2].hidden).toBe(false);
  expect(statusLabelForModule('ok', view.nodes[2].data)).toBe('Not observed');
  expect(view.edges).toEqual([]);
});

it('refuses malformed portable capture metadata', () => {
  const file = createGraphFile(graph(), { name: 'Net', status: 'traced' });
  file.document.semantic.nodes[1].metadata.moduleCapture = { status: 'depth-limited', maxDepth: -1 };
  expect(() => parseGraphFile(JSON.stringify(file))).toThrow('Invalid module capture limit');
});


const olderTiming = { cpuTotalUs: [10, 20, 30], cpuSelfUs: [1, 2, 3], deviceTotalUs: null, deviceMethod: null };

it.each(['collapsed', 'expanded', 'passthrough'])('shows no older timing in the %s module presentation', presentation => {
  const plain = { ...capturedModule(), expanded: presentation === 'expanded',
    capturePassThrough: presentation === 'passthrough' ? {} as never : undefined };
  const data = { ...plain, moduleLatency: olderTiming };
  const { container } = renderCard(data);
  expect(screen.queryByLabelText(/Module latency/)).toBeNull();
  expect(container.textContent).not.toMatch(/CPU|µs/);
  expect(moduleCardHeight(data)).toBe(moduleCardHeight(plain));
});

it('drops older timing evidence when the graph is saved again', () => {
  const saved = { nodes: [{ id: 'call', type: 'moduleNode', position: { x: 0, y: 0 }, data: { ...capturedModule(), moduleLatency: olderTiming } }], edges: [] };
  const document = modelGraphToDocumentV2(saved, { name: 'Net', revision: 1 });
  expect(document.semantic.nodes[0].metadata).not.toHaveProperty('moduleLatency');
});
