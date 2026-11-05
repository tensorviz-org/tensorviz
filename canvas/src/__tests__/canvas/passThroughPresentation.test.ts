import { mergeProjection } from '../../ide/applyProjection';
import { createDropoutInsertionReveal } from '../../ide/dropoutInsertion';
import fixture from '../../test/fixtures/canvas/capture-document.json';
import type { CapturePort, GraphDocumentV2, ModelGraph } from '../../types';
import { graphDocumentV2ToModelGraph, modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { showPassThroughCalls } from '../../services/passThroughPresentation';
import { reconcileModuleView } from '../../services/canvasLayout';

function example() {
  const document: GraphDocumentV2 = JSON.parse(JSON.stringify(fixture));
  const root = document.semantic.nodes.find(node => node.kind === 'module')!;
  const input = document.semantic.nodes.find(node => node.kind === 'input')!;
  const port = (input.metadata.capturePorts as CapturePort[])[0];
  const call = { ...root, node_id: 'identity', op: 'torch.nn.modules.linear.Identity',
    in_ports: ['identity:in'], out_ports: ['identity:out'], params: { _shape_provenance: 'observed' },
    metadata: { ...root.metadata, parentId: root.node_id, capturedOperationCount: 0,
      moduleLayerCount: 0, moduleMembers: [], moduleTemplateId: 'Identity', captureLabel: 'blocks.0.ls1',
      captureRef: { ...root.metadata.captureRef as object, semanticId: 'identity' },
      capturePorts: [
        { ...port, id: 'identity:in', nodeId: 'identity', direction: 'in', paths: [['arguments', 'args', 0]] },
        { ...port, id: 'identity:out', nodeId: 'identity', direction: 'out', paths: [['output']] },
      ],
    } };
  document.semantic.nodes.push(call);
  return { document, call, root };
}
const semantic = (graph: ModelGraph) => modelGraphToDocumentV2(graph, { name: 'Model', revision: 0 }).semantic;

test.each(['torch.nn.modules.linear.Identity', 'torch.nn.modules.dropout.Dropout'])('%s is folded only with unchanged tensor evidence', type => {
  const { document, call } = example(); call.op = type;
  const graph = graphDocumentV2ToModelGraph(document);
  expect(graph.nodes.find(node => node.id === 'identity')!.data.capturePassThrough?.valueId).toBe('h');
  const hidden = reconcileModuleView(graph.nodes, graph.edges, { relayout: true });
  expect(hidden.nodes.find(node => node.id === 'identity')!.hidden).toBe(true);
  expect(semantic({ ...graph, ...hidden })).toEqual(semantic(graph));
  const shown = reconcileModuleView(showPassThroughCalls(hidden.nodes, true), hidden.edges, { relayout: true });
  const identity = shown.nodes.find(node => node.id === 'identity')!;
  expect(identity.hidden).toBe(false);
  expect(identity.style?.height).toBe(88);
  const connection = shown.edges.find(edge => edge.target === 'identity')!;
  expect(connection).toMatchObject({ source: graph.nodes.find(node => node.type === 'inputNode')!.id,
    targetHandle: 'identity:in', data: { synthetic: true, _relationship: 'unchanged input' } });
  expect(connection.data?.captureEdge).toBeUndefined();
  expect(shown.edges.filter(edge => !edge.data?.synthetic)).toEqual(graph.edges.map(edge => ({ ...edge, hidden: false })));
  expect(semantic({ ...graph, ...shown })).toEqual(semantic(graph));
  expect(reconcileModuleView(shown.nodes, shown.edges).edges).toEqual(shown.edges);
  const roundTrip = graphDocumentV2ToModelGraph(modelGraphToDocumentV2({ ...graph, ...shown }, { name: 'Model', revision: 0 }));
  expect(roundTrip.nodes.find(node => node.id === 'identity')!.data.passThroughShown).toBe(true);
});

test.each(['custom', 'failed', 'partial', 'omitted', 'changed tensor', 'literal', 'container', 'child', 'operation', 'root', 'untrusted hint'])('%s never qualifies just because its operations are empty', reason => {
  const { document, call } = example();
  if (reason === 'custom') call.op = 'my.Identity';
  if (reason === 'failed') call.params._shape_provenance = 'failed';
  if (reason === 'partial') (document.semantic.metadata.capture as any).status = 'partial';
  if (reason === 'omitted') (document.semantic.metadata.capture as any).omissions = ['dispatch disabled'];
  if (reason === 'changed tensor') call.metadata.capturePorts[1].valueId = 'other';
  if (reason === 'literal') call.metadata.capturePorts[0].argumentKind = 'int';
  if (reason === 'container') call.metadata.capturePorts[0].paths = [['arguments', 'args', 0, 'items', 0]];
  if (reason === 'child') document.semantic.nodes.find(node => node.kind === 'layer')!.metadata.parentId = 'identity';
  if (reason === 'operation') call.metadata.capturedOperationCount = 1;
  if (reason === 'root') delete (call.metadata as any).parentId;
  if (reason === 'untrusted hint') {
    call.op = 'my.Custom'; (call.metadata as any).capturePassThrough = { valueId: 'h' };
  }
  const graph = graphDocumentV2ToModelGraph(document);
  expect(graph.nodes.find(node => node.id === 'identity')!.data.capturePassThrough).toBeUndefined();
});

test('an annotation is hidden with its parent and uses a real boundary port when the producer collapses', () => {
  const { document, call, root } = example();
  const operation = document.semantic.nodes.filter(node => node.kind === 'layer')[1];
  const port = (operation.metadata.capturePorts as CapturePort[]).find(port => port.direction === 'out')!;
  call.metadata.capturePorts.forEach(own => { own.valueId = port.valueId; });
  // A sibling frame can be collapsed while the annotation's parent stays open.
  const sibling = { ...root, node_id: 'sibling', metadata: { ...root.metadata, parentId: root.node_id } };
  document.semantic.nodes.push(sibling);
  operation.metadata.parentId = sibling.node_id;
  document.view.collapsed_modules.push('sibling');
  const graph = graphDocumentV2ToModelGraph(document);
  const shown = reconcileModuleView(showPassThroughCalls(graph.nodes, true), graph.edges, { relayout: true });
  const edge = shown.edges.find(edge => edge.target === 'identity')!;
  expect(edge.source).toBe('sibling');
  expect(shown.nodes.find(node => node.id === 'sibling')!.data.captureDisplayPorts?.some(port => port.id === edge.sourceHandle)).toBe(true);
  const collapsed = reconcileModuleView(shown.nodes.map(node => node.id === root.node_id
    ? { ...node, data: { ...node.data, expanded: false } } : node), shown.edges);
  expect(collapsed.edges.some(edge => edge.id.startsWith('passthrough:'))).toBe(false);
});

test('accepted Dropout becomes visible after classification while ordinary opens stay hidden', () => {
  const { document, call } = example();
  call.op = 'torch.nn.modules.dropout.Dropout';
  const before = JSON.parse(JSON.stringify(document));
  before.semantic.nodes = before.semantic.nodes.filter((node: any) => node.node_id !== call.node_id);
  const after = { file: 'model.py', cls: 'MLP', sourceDigest: 'accepted', document, traced: true } as any;
  const graph = graphDocumentV2ToModelGraph(document);
  const reveal = createDropoutInsertionReveal();
  expect(reconcileModuleView(reveal.apply(after, graph.nodes), graph.edges).nodes.find(node => node.id === call.node_id)!.hidden).toBe(true);
  reveal.accept({ before: { document: before }, after } as any, { document: before } as any);
  const shown = reconcileModuleView(reveal.apply(after, graph.nodes), graph.edges);
  expect(shown.nodes.find(node => node.id === call.node_id)).toMatchObject({ hidden: false, data: { passThroughShown: true } });
  expect(shown.edges.some(edge => edge.id === `passthrough:${call.node_id}`)).toBe(true);
  const refreshed = mergeProjection(graph, shown.nodes, { trustOpenness: true });
  expect(refreshed.nodes.find(node => node.id === call.node_id)!.data.passThroughShown).toBe(true);
});
