import type { ModelGraph } from '../../types';
import type { GraphDocumentV2 } from '../../types/graphDocument';
import portedFixture from '../../test/fixtures/canvas/ported-document.json';
import {
  graphDocumentV2ToModelGraph,
  legacyModelGraphToDocumentV2,
  modelGraphToDocumentV2,
} from '../../services/graphDocumentAdapter';

it('preserves exact ports, repeated operands and ownership from the actual Python converter', () => {
  const fixture = portedFixture as unknown as GraphDocumentV2;
  const graph = graphDocumentV2ToModelGraph(fixture);
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  expect(byId.get('add')!.data.parentId).toBe('block');
  expect(byId.get('add')!.data.inputPorts).toEqual(['args:0', 'args:1', 'alpha']);
  expect(byId.get('factory')!.data.inputPorts).toEqual([]);
  const saved = modelGraphToDocumentV2(graph, { graphId: 'ported', name: 'Ported', revision: 0 });
  expect(saved.semantic.nodes.map(node => [node.node_id, node.in_ports, node.out_ports]))
    .toEqual(fixture.semantic.nodes.map(node => [node.node_id, node.in_ports, node.out_ports]));
  expect(saved.semantic.edges).toEqual(fixture.semantic.edges);
});

it('uses semantic ownership when an older canvas duplicate disagrees', () => {
  const fixture = JSON.parse(JSON.stringify(portedFixture)) as GraphDocumentV2;
  fixture.view.layout.node_data!.add.parentId = 'wrong-parent';
  expect(graphDocumentV2ToModelGraph(fixture).nodes.find(node => node.id === 'add')!.data.parentId).toBe('block');
  fixture.semantic.nodes.find(node => node.node_id === 'add')!.metadata.parentId = null;
  expect(graphDocumentV2ToModelGraph(fixture).nodes.find(node => node.id === 'add')!.data.parentId).toBeNull();
});

it('preserves a parser ModuleList container kind even without a redundant isGroup flag', () => {
  const source: ModelGraph = { nodes: [{ id: 'blocks', type: 'moduleNode', position: { x: 12, y: 20 }, data: {
    layerType: { id: 'modulelist', name: 'ModuleList' }, parameters: { _modules: [{ op: 'Linear', params: { in_features: 8, out_features: 8 } }] },
  } }] as ModelGraph['nodes'], edges: [] };
  const document = modelGraphToDocumentV2(source, { name: 'Net', revision: 0 });
  expect(document.semantic.nodes[0]).toMatchObject({ node_id: 'blocks', kind: 'module', op: 'ModuleList', params: source.nodes[0].data.parameters });
  expect(document.view.collapsed_modules).toEqual(['blocks']);
  expect(graphDocumentV2ToModelGraph(document).nodes[0]).toMatchObject(source.nodes[0]);
});

const graph: ModelGraph = {
  nodes: [
    {
      id: 'input',
      type: 'layerNode',
      position: { x: 10, y: 20 },
      selected: true,
      data: {
        layerType: { id: 'input', name: 'Input', displayName: 'Input', description: '', icon: '', category: 'io', parameters: [] },
        parameters: { shape: '1,3,32,32' },
        isSelected: true,
        error: 'derived error',
      },
    },
    {
      id: 'module',
      type: 'moduleNode',
      position: { x: 200, y: 20 },
      data: {
        layerType: { id: 'module', name: 'ResidualBlock', displayName: 'Residual', description: '', icon: '', category: 'module', parameters: [] },
        parameters: { channels: 64 },
        isGroup: true,
        expanded: false,
        moduleMembers: ['conv'],
        moduleLayoutState: 'custom',
      },
    },
  ] as any,
  edges: [
    { id: 'real', source: 'input', target: 'module', sourceHandle: 'out', targetHandle: 'in' },
    { id: 'display-only', source: 'input', target: 'module', data: { synthetic: true } },
  ] as any,
};

it('separates semantic meaning from view state and drops transient canvas data', () => {
  const document = modelGraphToDocumentV2(graph, {
    graphId: 'g1', name: 'Demo', description: 'A demo graph', revision: 4,
  });

  expect(document.schemaVersion).toBe(2);
  expect(document.semantic.revision).toBe(4);
  expect(document.semantic.metadata.description).toBe('A demo graph');
  expect(document.semantic.nodes[0].params).toEqual({ shape: '1,3,32,32' });
  expect(document.semantic.nodes[0].metadata).not.toHaveProperty('error');
  expect(document.semantic.edges.map((edge) => edge.edge_id)).toEqual(['real']);
  expect(document.view.node_positions.input).toEqual({ x: 10, y: 20 });
  expect(document.view.selection).toEqual(['input']);
  expect(document.view.collapsed_modules).toEqual(['module']);
});

it('projects native V2 documents back to the canvas boundary', () => {
  const document = modelGraphToDocumentV2(graph, { graphId: 'g1', name: 'Demo', revision: 4 });
  const reopened = graphDocumentV2ToModelGraph(document);

  expect(reopened.nodes.map((node) => node.position)).toEqual([{ x: 10, y: 20 }, { x: 200, y: 20 }]);
  expect(reopened.nodes[0].data.parameters).toEqual({ shape: '1,3,32,32' });
  expect(reopened.nodes[0].selected).toBe(true);
  expect(reopened.nodes[1].data.expanded).toBe(false);
  expect(reopened.edges).toHaveLength(1);
  expect(reopened.edges[0]).toMatchObject({ sourceHandle: 'out', targetHandle: 'in' });
});

it('synthesizes canvas layer metadata for a backend-authored V2 node', () => {
  const document: GraphDocumentV2 = {
    schemaVersion: 2,
    semantic: {
      graph_id: 'g-native',
      name: 'Native',
      revision: 1,
      inputs: [{ name: 'input-1', shape: [1, 8] }],
      nodes: [
        {
          node_id: 'input-1',
          kind: 'input',
          op: 'Input',
          params: { shape: [1, 8] },
          in_ports: [],
          out_ports: ['y'],
          metadata: {},
        },
        {
          node_id: 'linear-1',
          kind: 'layer',
          op: 'nn.Linear',
          params: { in_features: 8, out_features: 4 },
          in_ports: ['x'],
          out_ports: ['y'],
          metadata: {},
        },
      ],
      edges: [],
      modules: [],
      operations: [],
      metadata: {},
    },
    view: {
      graph_id: 'g-native',
      node_positions: {
        'input-1': { x: 0, y: 60 },
        'linear-1': { x: 40, y: 60 },
      },
      collapsed_modules: [],
      selection: [],
      layout: {},
    },
  };

  const [input, node] = graphDocumentV2ToModelGraph(document).nodes;

  expect(input).toMatchObject({ type: 'inputNode', data: { layerType: { id: 'input' } } });
  expect(node.data.layerType).toMatchObject({
    id: 'linear',
    name: 'Linear',
    displayName: 'Linear',
    category: 'linear',
  });
  expect(node.data.parameters).toEqual({ in_features: 8, out_features: 4 });
});

it('reopens a migrated legacy ModelGraph exactly', () => {
  const legacy = JSON.parse(JSON.stringify(graph)) as ModelGraph;
  const document = legacyModelGraphToDocumentV2(legacy, { graphId: 'g1', name: 'Legacy', revision: 3 });

  expect(graphDocumentV2ToModelGraph(document)).toEqual(legacy);
});

it('does not let a preserved migration payload mask a newer semantic commit', () => {
  const legacy = JSON.parse(JSON.stringify(graph)) as ModelGraph;
  const document = legacyModelGraphToDocumentV2(legacy, {
    graphId: 'g1', name: 'Legacy', revision: 3,
  });
  document.semantic.revision = 4;
  document.semantic.nodes[0].params = { shape: '1,3,64,64' };
  document.view.node_positions.input = { x: 80, y: 90 };

  const reopened = graphDocumentV2ToModelGraph(document);

  expect(reopened.nodes[0].data.parameters).toEqual({ shape: '1,3,64,64' });
  expect(reopened.nodes[0].position).toEqual({ x: 80, y: 90 });
  expect(document.semantic.metadata.migration).toHaveProperty('legacyPayload');
});
