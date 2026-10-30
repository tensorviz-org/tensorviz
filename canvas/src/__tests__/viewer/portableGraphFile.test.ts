import fixture from '../../test/fixtures/canvas/capture-document.json';
import portable from '../../../../tests/fixtures/contracts/portable-capture.json';
import type { GraphDocumentV2 } from '../../types/graphDocument';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { argumentLeaves, requirePortableCapture } from '../../services/portableCapture';
import { graphDocumentV2ToModelGraph, modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { initialGraphFileView, revealGraphNode } from '../../viewer/graphFileView';
import { reconcileModuleView } from '../../services/canvasLayout';
import { anchorStatus, captureAnchor } from '../../services/graphCollaboration';
import { compareGraphFiles } from '../../services/graphComparison';
import contractCases from '../../test/fixtures/portable-contract-cases.json';
import { capturedFile } from '../../test/fixtures/capturedGraphFileFixture';
import editFixture from '../../../../tests/fixtures/contracts/captured-source-edit.json';
import initializerFixture from '../../../../tests/fixtures/contracts/initializer-arguments.json';

const clone = <T,>(item: T): T => JSON.parse(JSON.stringify(item));
const graph = () => graphDocumentV2ToModelGraph(clone(fixture) as GraphDocumentV2);

it('recovers dictionary names from an older file and ignores unverified saved labels', () => {
  const file = capturedFile();
  const capture = file.capture!.capture;
  const output = { kind: 'dict' as const, entries: [
    { key: { kind: 'str' as const, value: 'patch_tokens' }, value: capture.output! },
  ] };
  capture.output = output;
  capture.invocations[0].output = clone(output);
  for (const node of file.document.semantic.nodes) {
    for (const port of node.metadata.capturePorts as any[]) {
      delete port.displayName;
      if ((node.kind === 'output' || node.kind === 'module') && port.paths[0]?.[0] === 'output') {
        port.paths = [['output', 'entries', 0, 'value']];
        port.displayName = 'invented_name';
      }
    }
    if (node.kind === 'output' || node.kind === 'module') {
      delete node.metadata.outputShape;
      delete node.metadata.outputTensor;
    }
  }
  const imported = parseGraphFile(JSON.stringify(file));
  const names = imported.document.semantic.nodes.filter(node => node.kind === 'output' || node.kind === 'module')
    .flatMap(node => (node.metadata.capturePorts as any[]).filter(port => port.paths[0]?.[0] === 'output').map(port => port.displayName));
  expect(names).toEqual(['patch_tokens', 'patch_tokens']);
  expect(imported.document.semantic.edges).toEqual(file.document.semantic.edges);
  expect(imported.capture).toEqual(file.capture);
});

it('keeps nested sequence positions, typed dictionary keys, and empty containers in readable names', () => {
  const leaves = argumentLeaves({ kind: 'dict', entries: [
    { key: { kind: 'str', value: 'states' }, value: { kind: 'tuple', items: [{ kind: 'value', id: 'v' }, { kind: 'none' }] } },
    { key: { kind: 'str', value: 'extra.key' }, value: { kind: 'dict', entries: [
      { key: { kind: 'int', value: '2' }, value: { kind: 'list', items: [] } },
      { key: { kind: 'bool', value: false }, value: { kind: 'none' } },
      { key: { kind: 'none' }, value: { kind: 'dict', entries: [] } },
    ] } },
  ] }, ['output']);
  expect(leaves.map(leaf => leaf.displayName)).toEqual(['states[0]', 'states[1]', '["extra.key"][2]', '["extra.key"][False]', '["extra.key"][None]']);
  expect(leaves[2].path).toEqual(['output', 'entries', 1, 'value', 'entries', 0, 'value']);
});

function initializerCapture() {
  const value = clone(portable) as any, capture = value.capture;
  const binding = { id: 'constructor:0', snapshotId: capture.source.id, kind: 'constructor', quality: 'exact',
    anchors: [{ file: capture.source.files[0].path, digest: capture.source.files[0].digest,
      startLine: 1, startColumn: 0, endLine: 1, endColumn: 5, scope: 'Model.__init__', tokenDigest: 'a'.repeat(64) }] };
  capture.sourceBindings.push(binding);
  Object.assign(capture.modules[0], { constructorClassVerified: true, constructorBindingIds: [binding.id],
    initializerArguments: clone(initializerFixture.entry) });
  return value;
}

it.each(initializerFixture.cases)('initializer agreement with Python: $name', item => {
  const value = initializerCapture();
  for (const patch of item.patches) {
    let target = value.capture.modules[0].initializerArguments;
    for (const key of patch.path.slice(0, -1)) target = target[key];
    target[patch.path[patch.path.length - 1]] = patch.value;
  }
  if (item.portable) expect(requirePortableCapture(value)).toEqual(value);
  else expect(() => requirePortableCapture(value)).toThrow();
});

it('initializer entry survives a graph-file round trip without creating write authority', () => {
  const capture = requirePortableCapture(initializerCapture());
  const file = createGraphFile(graph(), { name: 'Model', status: 'traced', capture });
  expect(parseGraphFile(JSON.stringify(file)).capture!.capture.modules).toEqual(capture.capture.modules);
});

it('refuses unverified initializer class identity and numeric negative zero', () => {
  const value = initializerCapture();
  value.capture.modules[0].constructorClassVerified = false;
  expect(() => requirePortableCapture(value)).toThrow();
  value.capture.modules[0].constructorClassVerified = true;
  value.capture.modules[0].initializerArguments.varArgs[0].value = -0;
  expect(() => requirePortableCapture(value)).toThrow();
});

it('exports and reimports all captured computation, original ports/edges and layout without live source identity', () => {
  const source = graph();
  source.nodes[0].position = { x: 370, y: 122 };
  const file = createGraphFile(source, { name: 'Model', status: 'traced', capture: requirePortableCapture(clone(portable)) });
  expect(file.version).toBe(4);
  expect(file.capture).toEqual(portable);
  expect(file.document.semantic.nodes.map(n => n.metadata.capturePorts)).toEqual(fixture.semantic.nodes.map(n => n.metadata.capturePorts));
  expect(file.document.semantic.edges).toEqual(fixture.semantic.edges);
  expect(file.document.view.node_positions[source.nodes[0].id]).toEqual(source.nodes[0].position);
  expect(file.document.semantic.nodes.every(node => (node.metadata.captureRef as any).sourceId === portable.capture.source.id)).toBe(true);
  expect(parseGraphFile(JSON.stringify(file))).toEqual(file);
  expect(JSON.stringify(file)).not.toContain('model.py');
});

it('never exports live source-edit authority or its local source anchors', () => {
  const source = graph();
  source.nodes[0].data.sourceEditCapabilities = [editFixture.capability] as any;
  source.nodes[0].data.sourceEditLimitation = 'source_unavailable';
  const file = createGraphFile(source, { name: 'Model', status: 'traced', capture: requirePortableCapture(clone(portable)) });
  expect(JSON.stringify(file)).not.toContain('sourceEditCapabilities');
  expect(JSON.stringify(file)).not.toContain('sourceEditLimitation');
  expect(JSON.stringify(file)).not.toContain(editFixture.capability.id);
});

it('preserves JAX inferred shapes and refuses invented runtime measurements', () => {
  const source = graph();
  const value = clone(portable) as any;
  Object.assign(value.capture, { backend: 'jaxpr', jaxVersion: '0.10.2', torchVersion: null });
  value.capture.coverage.scope = 'traced_program';
  source.nodes.forEach(node => { node.data.shapeEvidence = 'inferred'; });
  const file = createGraphFile(source, { name: 'JAX program', status: 'traced', capture: requirePortableCapture(value) });
  const reopened = parseGraphFile(JSON.stringify(file));
  expect(reopened.document.semantic.nodes.every(node => node.metadata.shapeEvidence === 'inferred'
    && node.params._shape_provenance === 'inferred')).toBe(true);
  reopened.document.semantic.nodes[0].metadata.shapeEvidence = 'measured';
  expect(() => parseGraphFile(JSON.stringify(reopened))).toThrow(/inconsistent/);
});

it('collapse/reveal and annotations never rewrite computation or operand identity', () => {
  const file = capturedFile(), before = JSON.stringify(file.capture);
  const initial = initialGraphFileView(file);
  const module = initial.nodes.find(n => n.data.captureRef?.kind === 'invocation')!;
  const collapsed = reconcileModuleView(initial.nodes.map(n => n.id === module.id ? { ...n, data: { ...n.data, expanded: false } } : n), initial.edges);
  const add = collapsed.nodes.find(n => n.data.captureRef?.semanticId === 'add2')!;
  const revealed = revealGraphNode(collapsed, add.id);
  expect(revealed.nodes.find(n => n.id === add.id)?.hidden).toBe(false);
  expect(modelGraphToDocumentV2(revealed, { name: 'Model', revision: 0 }).semantic.edges).toEqual(file.document.semantic.edges);
  const roundtrip = createGraphFile(collapsed, { name: 'Model', status: 'stale', capture: file.capture });
  expect(JSON.stringify(roundtrip.capture)).toBe(before);
  expect(roundtrip.document.semantic.edges).toEqual(file.document.semantic.edges);
  expect(roundtrip.document.semantic.nodes.every(n => n.metadata.shapeEvidence === 'stale')).toBe(true);
});

it('refuses to silently downgrade a canonical graph when evidence is missing', () => {
  expect(() => createGraphFile(graph(), { name: 'Model', status: 'traced' })).toThrow(/retained canonical/);
  const file = capturedFile();
  expect(() => parseGraphFile(JSON.stringify({ ...file, version: 3 }))).toThrow(/version 4/);
  expect(() => parseGraphFile(JSON.stringify({ ...file, capture: undefined }))).toThrow(/Lossless import refused/);
});

it.each([
  ['missing event', (file: any) => { file.document.semantic.nodes.pop(); }],
  ['missing edge', (file: any) => { file.document.semantic.edges.pop(); }],
  ['repeated operand collapsed', (file: any) => { file.document.semantic.edges[2].target_port = file.document.semantic.edges[1].target_port; }],
  ['invented edge', (file: any) => { file.document.semantic.edges.push({ ...file.document.semantic.edges[0], edge_id: 'invented' }); }],
  ['redirected source', (file: any) => { file.document.semantic.edges[1].source_port = file.document.semantic.edges[0].source_port; }],
  ['wrong tensor shape', (file: any) => { file.document.semantic.nodes[0].metadata.capturePorts[0].shape = [99]; }],
  ['fake metric', (file: any) => { file.document.semantic.nodes[0].metadata.parameterCount = 900; }],
  ['foreign capture', (file: any) => { file.document.semantic.nodes[0].metadata.captureRef.captureId = 'other'; }],
  ['bad owner', (file: any) => { file.document.semantic.nodes.find((n: any) => n.metadata.captureRef.kind === 'operation').metadata.parentId = undefined; }],
  ['false completeness', (file: any) => { file.capture.capture.coverage.omissions.push('operation_limit'); }],
  ['unknown producer', (file: any) => { file.capture.capture.values[1].producer.operationId = 'missing'; }],
  ['private operand', (file: any) => { file.capture.capture.operations[0].arguments.kwargs.token = { kind: 'str', value: 'SECRET' }; }],
  ['source identity', (file: any) => { file.capture.capture.source.workspaceId = 'live-workspace'; }],
  ['unbound constructor class', (file: any) => { file.capture.capture.modules[0].constructorClassVerified = true; }],
])('rejects %s instead of displaying a plausible altered graph', (_name, change) => {
  const file = clone(capturedFile());
  change(file);
  expect(() => parseGraphFile(JSON.stringify(file))).toThrow();
});

it('preserves arbitrary typed literals exactly, including large integers and nested repeated values', () => {
  const capture = clone(portable) as any;
  capture.capture.operations[0].arguments.kwargs.alpha = { kind: 'tuple', items: [
    { kind: 'int', value: '9007199254740993' }, { kind: 'float', value: '-0' },
    { kind: 'dict', entries: [{ key: { kind: 'str', value: 'pair' }, value: { kind: 'list', items: [{ kind: 'value', id: 'h' }, { kind: 'value', id: 'h' }] } }] },
  ] };
  expect(requirePortableCapture(capture)).toEqual(capture);
  capture.capture.operations[0].arguments.kwargs.alpha.items[2].entries[0].value.items[1].id = 'b';
  expect(() => requirePortableCapture(capture)).toThrow(/inconsistent/);
});

it('detects changed literal context for saved notes and canonical comparison, ignoring layout', () => {
  const file = capturedFile(), node = file.document.semantic.nodes.find(n => (n.metadata.captureRef as any).semanticId === 'add1')!;
  const anchor = captureAnchor(file, { kind: 'node', nodeId: node.node_id });
  const next = clone(file);
  (next.capture!.capture.operations[0].arguments.args[1] as any).value = '2';
  const parsed = parseGraphFile(JSON.stringify(next));
  expect(anchorStatus(parsed, anchor)).toBe('changed');
  file.document.view.node_positions[node.node_id].x += 150;
  expect(anchorStatus(file, anchor)).toBe('unchanged');
  expect(compareGraphFiles(file, parsed).architectureChanged).toBe(true);
});

it.each(contractCases)('agrees with the Python contract: $name', ({ patches, valid }) => {
  const capture = clone(portable);
  for (const patch of patches) {
    let target: any = capture.capture;
    for (const key of patch.path.slice(0, -1)) target = target[key];
    target[patch.path[patch.path.length - 1]] = patch.value;
  }
  if (valid) expect(requirePortableCapture(capture)).toEqual(capture);
  else expect(() => requirePortableCapture(capture)).toThrow();
});


it('opens an older file with module timing and call-site digests by dropping them', () => {
  const value = clone(portable) as any;
  for (const call of value.capture.invocations) {
    call.latency = { cpuTotalUs: [4, 8], cpuSelfUs: [1, 2], deviceTotalUs: null, deviceMethod: null };
    call.callSiteDigest = null;
  }
  value.capture.latencyProfile = { status: 'measured', reason: null, device: 'cpu', warmupIterations: 3,
    iterations: 2, measuredInvocations: value.capture.invocations.length, benchmarkSamplesMs: [0.1, 0.2] };
  const capture = requirePortableCapture(value);
  expect(capture).toEqual(portable);
  const imported = parseGraphFile(JSON.stringify(createGraphFile(graph(), { name: 'Older model', status: 'traced', capture })));
  expect(JSON.stringify(imported)).not.toMatch(/latency|callSiteDigest/i);
});
