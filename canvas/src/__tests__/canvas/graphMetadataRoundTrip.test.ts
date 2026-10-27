import {
  graphDocumentV2ToModelGraph,
  modelGraphToDocumentV2,
} from '../../services/graphDocumentAdapter';
import type { GraphDocumentV2 } from '../../types/graphDocument';

/**
 * `template_instances` must survive a load/save round trip.
 *
 * It records which nodes belong to which instantiated block, and two separate
 * consumers depend on it: module presentation draws collapsed cards from it,
 * and the backend resolver resolves an instance id to that block's entry/exit
 * nodes with it.
 *
 * Both directions dropped it. Measured on a real AI-built graph — 107 nodes, 12
 * declared `full_transformer_block` instances — the canvas showed 105 flat cards
 * and zero "Expand Block" affordances (a 1568x25px ribbon), because ModelBuilder
 * could not see the declaration and fell back to autoModularize's repeat
 * detection, which produced 3 op-pattern groups instead of the 12 real blocks.
 * The save path then rebuilt semantic.metadata from scratch and erased the record
 * outright, so a later AI turn could no longer reference a block by name.
 */
const INSTANCES = [
  { instance_id: 'encoder_block_01', template_id: 'full_transformer_block', entry_node_id: 'encoder_block_01__norm1', exit_node_id: 'encoder_block_01__add2' },
  { instance_id: 'encoder_block_02', template_id: 'full_transformer_block', entry_node_id: 'encoder_block_02__norm1', exit_node_id: 'encoder_block_02__add2' },
];

function documentWith(metadata: Record<string, unknown>): GraphDocumentV2 {
  return {
    schemaVersion: 2,
    semantic: {
      graph_id: 'g1', name: 'n', revision: 3, inputs: [],
      nodes: [
        { node_id: 'input_node', kind: 'input', op: 'Input', params: { shape: '1,16' }, in_ports: [], out_ports: ['y'], metadata: {} },
        { node_id: 'encoder_block_01__norm1', kind: 'layer', op: 'LayerNorm', params: {}, in_ports: ['x'], out_ports: ['y'], metadata: {} },
      ],
      edges: [], modules: [], operations: [], metadata,
    },
    view: { graph_id: 'g1', node_positions: {}, collapsed_modules: [], selection: [], layout: {} },
  } as unknown as GraphDocumentV2;
}

describe('graph-level metadata round trip', () => {
  it('carries template_instances onto the loaded ModelGraph', () => {
    const graph = graphDocumentV2ToModelGraph(documentWith({ template_instances: INSTANCES }));
    expect(graph.metadata?.template_instances).toHaveLength(2);
    expect((graph.metadata?.template_instances as typeof INSTANCES)[0].template_id)
      .toBe('full_transformer_block');
  });

  it('does not erase template_instances when saving', () => {
    const loaded = graphDocumentV2ToModelGraph(documentWith({ template_instances: INSTANCES }));
    const saved = modelGraphToDocumentV2(loaded, { name: 'n', revision: 3 });
    expect((saved.semantic.metadata as { template_instances?: unknown[] })?.template_instances)
      .toHaveLength(2);
  });

  it('still records a description without clobbering the rest', () => {
    const loaded = graphDocumentV2ToModelGraph(documentWith({ template_instances: INSTANCES }));
    const saved = modelGraphToDocumentV2(loaded, { name: 'n', revision: 3, description: 'my model' });
    const meta = saved.semantic.metadata as { template_instances?: unknown[]; description?: string };
    expect(meta.description).toBe('my model');
    expect(meta.template_instances).toHaveLength(2);
  });

  it('survives two consecutive round trips', () => {
    // A user saving twice must not lose it on the second save either.
    const once = modelGraphToDocumentV2(
      graphDocumentV2ToModelGraph(documentWith({ template_instances: INSTANCES })),
      { name: 'n', revision: 3 },
    );
    const twice = modelGraphToDocumentV2(
      graphDocumentV2ToModelGraph(once as GraphDocumentV2),
      { name: 'n', revision: 4 },
    );
    expect((twice.semantic.metadata as { template_instances?: unknown[] })?.template_instances)
      .toHaveLength(2);
  });

  it('leaves a graph with no metadata alone', () => {
    const graph = graphDocumentV2ToModelGraph(documentWith({}));
    expect(graph.metadata).toEqual({});
    const saved = modelGraphToDocumentV2(graph, { name: 'n', revision: 3 });
    expect(saved.semantic.metadata).toEqual({});
  });

  it('does not alias the document, so mutating the graph cannot corrupt it', () => {
    const document = documentWith({ template_instances: INSTANCES });
    const graph = graphDocumentV2ToModelGraph(document);
    (graph.metadata as { template_instances: unknown[] }).template_instances.pop();
    expect((document.semantic.metadata as { template_instances: unknown[] }).template_instances)
      .toHaveLength(2);
  });
});
