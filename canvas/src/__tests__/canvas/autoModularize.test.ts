import { autoModularize } from '../../services/autoModularize';
import { reconcileModuleView } from '../../services/canvasLayout';
import { buildInferablePayload } from '../../services/shapeInferencePayload';
import { ModelNode, ModelEdge } from '../../types';

/**
 * Build a graph with F5-TTS's exact topology (smaller depth/convLayers): a
 * timestep MLP whose output `t_emb` fans into every DiT block, a ConvNeXt-V2
 * text encoder, an input-embedding fusion, a conv position embedding, and a DiT
 * backbone — including the cross-block conditioning and residual edges that are
 * the whole reason F5-TTS ships flat. Mirrors generative_templates._build_f5tts_structure.
 */
function buildF5Like(convLayers: number, depth: number): { nodes: ModelNode[]; edges: ModelEdge[] } {
  const nodes: ModelNode[] = [];
  const edges: ModelEdge[] = [];
  const node = (id: string, type: string, opId: string, name: string) =>
    nodes.push({
      id,
      type,
      position: { x: 0, y: 0 },
      data: {
        layerType: { id: opId, name, displayName: name, description: '', icon: '', category: 'misc', parameters: [] },
        parameters: { name },
      },
    } as ModelNode);
  const link = (s: string, t: string) => edges.push({ id: `${s}__${t}`, source: s, target: t } as ModelEdge);

  node('in_text', 'inputNode', 'input', 'Text');
  node('in_noisy', 'inputNode', 'input', 'Noisy');
  node('in_cond', 'inputNode', 'input', 'Cond');
  node('in_time', 'inputNode', 'input', 'Time');

  // timestep branch → t_emb (the global conditioning signal)
  node('t_sinus', 'layerNode', 'sinusoidal_pos_embedding', 'SinusPositionEmbedding');
  node('t_lin1', 'layerNode', 'linear', 'Time MLP 1');
  node('t_silu', 'layerNode', 'silu', 'SiLU');
  node('t_lin2', 'layerNode', 'linear', 'Time MLP 2');
  link('in_time', 't_sinus'); link('t_sinus', 't_lin1'); link('t_lin1', 't_silu'); link('t_silu', 't_lin2');
  const tEmb = 't_lin2';

  // text encoder: Char Embedding + ConvNeXt-V2 blocks (with residual)
  node('text_embed', 'layerNode', 'embedding', 'Char Embedding');
  link('in_text', 'text_embed');
  let prev = 'text_embed';
  for (let i = 0; i < convLayers; i++) {
    const b = `cnx${i}`;
    node(`${b}_dw`, 'layerNode', 'conv1d', `ConvNeXt-V2 ${i}: DWConv`);
    node(`${b}_ln`, 'layerNode', 'layernorm', 'LayerNorm');
    node(`${b}_pw1`, 'layerNode', 'linear', 'PWConv1');
    node(`${b}_gelu`, 'layerNode', 'gelu', 'GELU');
    node(`${b}_grn`, 'layerNode', 'grn', 'GRN');
    node(`${b}_pw2`, 'layerNode', 'linear', 'PWConv2');
    node(`${b}_add`, 'layerNode', 'add', `ConvNeXt Residual ${i}`);
    link(prev, `${b}_dw`); link(`${b}_dw`, `${b}_ln`); link(`${b}_ln`, `${b}_pw1`);
    link(`${b}_pw1`, `${b}_gelu`); link(`${b}_gelu`, `${b}_grn`); link(`${b}_grn`, `${b}_pw2`);
    link(prev, `${b}_add`); link(`${b}_pw2`, `${b}_add`);
    prev = `${b}_add`;
  }
  const textFeat = prev;

  node('in_concat', 'layerNode', 'cat', 'InputEmbedding Concat');
  link('in_noisy', 'in_concat'); link('in_cond', 'in_concat'); link(textFeat, 'in_concat');
  node('in_proj', 'layerNode', 'linear', 'Input Projection');
  link('in_concat', 'in_proj');

  node('cpe_c1', 'layerNode', 'conv1d', 'ConvPosEmbed C1');
  node('cpe_m1', 'layerNode', 'mish', 'Mish');
  node('cpe_c2', 'layerNode', 'conv1d', 'ConvPosEmbed C2');
  node('cpe_m2', 'layerNode', 'mish', 'Mish');
  node('cpe_add', 'layerNode', 'add', 'ConvPos Residual');
  link('in_proj', 'cpe_c1'); link('cpe_c1', 'cpe_m1'); link('cpe_m1', 'cpe_c2'); link('cpe_c2', 'cpe_m2');
  link('in_proj', 'cpe_add'); link('cpe_m2', 'cpe_add');
  let x = 'cpe_add';

  for (let i = 0; i < depth; i++) {
    const b = `dit${i}`;
    node(`${b}_mod`, 'layerNode', 'linear', `DiT ${i}: AdaLN-Zero Proj`); // off-spine modulation
    node(`${b}_adaln1`, 'layerNode', 'ada_layer_norm', `DiT ${i}: AdaLayerNorm-Zero`);
    node(`${b}_rope`, 'layerNode', 'rotary_embedding', 'RoPE');
    node(`${b}_attn`, 'layerNode', 'multiheadattention', 'Self-Attention');
    node(`${b}_add1`, 'layerNode', 'add', 'Gated Residual (attn)');
    node(`${b}_adaln2`, 'layerNode', 'ada_layer_norm', 'FF AdaLN');
    node(`${b}_ff1`, 'layerNode', 'linear', 'FeedForward In');
    node(`${b}_ffg`, 'layerNode', 'gelu', 'GELU');
    node(`${b}_ffd`, 'layerNode', 'dropout', 'Dropout');
    node(`${b}_ff2`, 'layerNode', 'linear', 'FeedForward Out');
    node(`${b}_add2`, 'layerNode', 'add', 'Gated Residual (ff)');
    link(tEmb, `${b}_mod`);                                      // timestep -> modulation
    link(x, `${b}_adaln1`); link(`${b}_mod`, `${b}_adaln1`);     // sequence + modulation
    link(`${b}_adaln1`, `${b}_rope`); link(`${b}_rope`, `${b}_attn`);
    link(x, `${b}_add1`); link(`${b}_attn`, `${b}_add1`);        // residual
    link(`${b}_add1`, `${b}_adaln2`); link(`${b}_mod`, `${b}_adaln2`); // reuse modulation
    link(`${b}_adaln2`, `${b}_ff1`); link(`${b}_ff1`, `${b}_ffg`);
    link(`${b}_ffg`, `${b}_ffd`); link(`${b}_ffd`, `${b}_ff2`);
    link(`${b}_add1`, `${b}_add2`); link(`${b}_ff2`, `${b}_add2`);
    x = `${b}_add2`;
  }
  node('final_mod', 'layerNode', 'linear', 'AdaLN-Final Proj');
  node('final_norm', 'layerNode', 'ada_layer_norm_final', 'AdaLayerNorm-Final');
  node('out_proj', 'layerNode', 'linear', 'Output Projection');
  node('out_node', 'outputNode', 'output', 'Output');
  link(tEmb, 'final_mod');
  link(x, 'final_norm'); link('final_mod', 'final_norm');
  link('final_norm', 'out_proj'); link('out_proj', 'out_node');

  return { nodes, edges };
}

const idSet = (ns: ModelNode[]) => ns.map((n) => n.id).sort();
const edgeSet = (es: ModelEdge[]) => es.map((e) => `${e.source}->${e.target}`).sort();

describe('autoModularize', () => {
  it('detects the repeated DiT and ConvNeXt stacks as ×N modules', () => {
    const flat = buildF5Like(3, 4);
    const { groups } = autoModularize(flat.nodes, flat.edges);

    // The DiT backbone: one module, repeatCount=4, holding all 4×11 block nodes
    // (10 spine nodes + the off-spine AdaLayerNorm-Zero modulation Linear).
    const dit = groups.find((g) => g.repeatCount === 4 && g.memberIds.length === 4 * 11);
    expect(dit).toBeTruthy();

    // The ConvNeXt-V2 text encoder: repeatCount=3, all 3×7 nodes (no off-spine).
    const cnx = groups.find((g) => g.repeatCount === 3 && g.memberIds.length === 3 * 7);
    expect(cnx).toBeTruthy();

    // Block labels come from the template's node names, not the raw op.
    expect(dit!.label).toBe('DiT');
    expect(cnx!.label).toBe('ConvNeXt-V2');
  });

  it('an expanded ×N module renders ONE representative block, not all N copies', () => {
    // The DiT stack is ×4 of a 10-node block. Expanded, the canvas must show 10
    // visible nodes (one block) — the other 3 instances are kept for the
    // payload but flagged repeatHidden so they never render (matches the
    // exported nn.ModuleList).
    const flat = buildF5Like(3, 4);
    const m = autoModularize(flat.nodes, flat.edges);
    const dit = m.nodes.find((n) => n.type === 'moduleNode' && n.data.repeatCount === 4)!;
    expect(dit.data.moduleLayerCount).toBe(11); // one block = 10 spine + mod Linear

    const ditChildren = m.nodes.filter((n) => n.data.parentId === dit.id);
    expect(ditChildren.length).toBe(44); // all 4 instances still in the graph
    expect(ditChildren.filter((n) => n.data.repeatHidden).length).toBe(33); // 3 extra ×11

    // Expand the module and reconcile: only the canonical 11 are visible.
    const logical = m.nodes.map((n) =>
      n.id === dit.id ? ({ ...n, data: { ...n.data, expanded: true } } as ModelNode) : n
    );
    const r = reconcileModuleView(logical, m.edges, { relayout: false });
    const visible = (r.nodes as ModelNode[]).filter((n) => n.data.parentId === dit.id && !n.hidden);
    expect(visible.length).toBe(11);
  });

  it('never pulls the global conditioning signal (t_emb) into a block', () => {
    // t_emb fans into every DiT block; folding it into one block would break the
    // others. It must stay top-level (ungrouped) — this is the safety invariant.
    const flat = buildF5Like(3, 4);
    const { groups } = autoModularize(flat.nodes, flat.edges);
    const allMembers = new Set(groups.flatMap((g) => g.memberIds));
    expect(allMembers.has('t_lin2')).toBe(false);
  });

  it('keeps the verify/validate payload byte-identical to the flat graph', () => {
    // The whole point: grouping is display-only. The graph the backend sees
    // (after reconcile + payload-build) must equal the original flat graph, so
    // F5-TTS's carefully-tuned verify/validate gates are never disturbed.
    const flat = buildF5Like(3, 4);
    const flatPayload = buildInferablePayload(flat.nodes, flat.edges);

    const m = autoModularize(flat.nodes, flat.edges);
    const rendered = reconcileModuleView(m.nodes, m.edges, { relayout: false });
    const groupedPayload = buildInferablePayload(rendered.nodes as ModelNode[], rendered.edges as ModelEdge[]);

    expect(idSet(groupedPayload.inferableNodes)).toEqual(idSet(flatPayload.inferableNodes));
    expect(edgeSet(groupedPayload.inferableEdges)).toEqual(edgeSet(flatPayload.inferableEdges));
    // And the wrapper modules themselves must not leak into the payload.
    expect(groupedPayload.inferableNodes.some((n) => n.type === 'moduleNode')).toBe(false);
  });

  it('collapsed modules stay visually connected via synthetic display edges', () => {
    const flat = buildF5Like(3, 4);
    const m = autoModularize(flat.nodes, flat.edges);
    const rendered = reconcileModuleView(m.nodes, m.edges, { relayout: false });
    const synth = (rendered.edges as ModelEdge[]).filter((e) => e.data?.synthetic);
    // Every collapsed module box has at least one synthetic in/out connector.
    const moduleIds = new Set(m.nodes.filter((n) => n.type === 'moduleNode').map((n) => n.id));
    for (const id of moduleIds) {
      expect(synth.some((e) => e.source === id || e.target === id)).toBe(true);
    }
  });

  it('reconcile is idempotent — synthetic edges do not accumulate', () => {
    const flat = buildF5Like(3, 4);
    const m = autoModularize(flat.nodes, flat.edges);
    const once = reconcileModuleView(m.nodes, m.edges, { relayout: false });
    const twice = reconcileModuleView(m.nodes, once.edges as ModelEdge[], { relayout: false });
    const count = (es: ModelEdge[]) => es.filter((e) => e.data?.synthetic).length;
    expect(count(twice.edges as ModelEdge[])).toBe(count(once.edges as ModelEdge[]));
  });

  it('is a no-op on a small non-repetitive graph', () => {
    const nodes: ModelNode[] = [
      { id: 'a', type: 'inputNode', position: { x: 0, y: 0 }, data: { layerType: { id: 'input', name: 'In', displayName: 'In', description: '', icon: '', category: 'io', parameters: [] }, parameters: {} } },
      { id: 'b', type: 'layerNode', position: { x: 0, y: 0 }, data: { layerType: { id: 'linear', name: 'L', displayName: 'L', description: '', icon: '', category: 'linear', parameters: [] }, parameters: {} } },
      { id: 'c', type: 'outputNode', position: { x: 0, y: 0 }, data: { layerType: { id: 'output', name: 'Out', displayName: 'Out', description: '', icon: '', category: 'io', parameters: [] }, parameters: {} } },
    ] as ModelNode[];
    const edges: ModelEdge[] = [
      { id: 'a__b', source: 'a', target: 'b' },
      { id: 'b__c', source: 'b', target: 'c' },
    ] as ModelEdge[];
    const res = autoModularize(nodes, edges);
    expect(res.groups).toHaveLength(0);
    expect(res.nodes).toBe(nodes);
  });

  it('leaves an already-modular graph untouched', () => {
    const flat = buildF5Like(3, 4);
    const m = autoModularize(flat.nodes, flat.edges); // now has moduleNodes
    const again = autoModularize(m.nodes, m.edges);
    expect(again.groups).toHaveLength(0);
    expect(again.nodes).toBe(m.nodes);
  });
});

describe('autoModularize on the real F5-TTS template (290 nodes / 382 edges)', () => {
  // Fixture dumped from generative_templates._build_f5tts_structure() (defaults:
  // depth=22, conv_layers=4). Guards the headline case: the full-fidelity F5-TTS.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const real = require('../../test/fixtures/canvas/f5tts_real.json') as { nodes: ModelNode[]; edges: ModelEdge[] };

  it('collapses the 22 DiT blocks and 4 ConvNeXt blocks into ×N modules', () => {
    const { nodes, groups } = autoModularize(real.nodes, real.edges);
    expect(groups.find((g) => g.repeatCount === 22 && g.memberIds.length === 22 * 11)).toBeTruthy();
    expect(groups.find((g) => g.repeatCount === 4 && g.memberIds.length === 4 * 7)).toBeTruthy();

    // The 267-node flat graph becomes a handful of top-level boxes.
    const topLevel = nodes.filter((n) => n.type === 'moduleNode' || !n.data.parentId);
    expect(topLevel.length).toBeLessThan(20);
  });

  it('keeps the verify/validate payload identical to the flat F5-TTS graph', () => {
    const flatPayload = buildInferablePayload(real.nodes, real.edges);
    const m = autoModularize(real.nodes, real.edges);
    const rendered = reconcileModuleView(m.nodes, m.edges, { relayout: false });
    const grouped = buildInferablePayload(rendered.nodes as ModelNode[], rendered.edges as ModelEdge[]);

    expect(idSet(grouped.inferableNodes)).toEqual(idSet(flatPayload.inferableNodes));
    expect(edgeSet(grouped.inferableEdges)).toEqual(edgeSet(flatPayload.inferableEdges));
    expect(grouped.inferableNodes).toHaveLength(290);
    expect(grouped.inferableEdges).toHaveLength(382);
  });
});

describe('declared blocks beat inferred repeats', () => {
  // The shape the resolver actually writes (ai/domain/operation_resolver.py):
  // { instance_id, template_id, entry_node_id, exit_node_id, config }, with member
  // node ids prefixed `<instance_id>__`.
  const declaredGraph = (instanceCount: number, perBlock = 4) => {
    const nodes: any[] = [];
    const edges: any[] = [];
    let prev = '';
    for (let i = 0; i < instanceCount; i += 1) {
      const inst = `encoder_block_${String(i).padStart(2, '0')}`;
      for (let k = 0; k < perBlock; k += 1) {
        const id = `${inst}__n${k}`;
        nodes.push({
          id,
          type: 'layerNode',
          position: { x: nodes.length * 280, y: 0 },
          data: { layerType: { id: k === 0 ? 'linear' : 'relu', name: 'L', parameters: [] }, parameters: {} },
        });
        if (prev) edges.push({ id: `e${edges.length}`, source: prev, target: id });
        prev = id;
      }
    }
    const metadata = {
      template_instances: Array.from({ length: instanceCount }, (_v, i) => {
        const inst = `encoder_block_${String(i).padStart(2, '0')}`;
        return {
          instance_id: inst,
          template_id: 'full_transformer_block',
          entry_node_id: `${inst}__n0`,
          exit_node_id: `${inst}__n${perBlock - 1}`,
          config: {},
        };
      }),
    };
    return { nodes, edges, metadata };
  };

  it('groups one card per declared instance, not per inferred repeat', () => {
    const { nodes, edges, metadata } = declaredGraph(12);

    const declared = autoModularize(nodes, edges, { metadata });

    // 12 declared blocks -> 12 cards. Signature inference on this graph collapses
    // everything into ONE ×12 run, which is why declared structure has to win.
    expect(declared.groups.length).toBe(12);
    declared.groups.forEach((g) => {
      const inst = g.moduleId.replace('auto-mod-', '');
      expect(g.memberIds.every((id) => id.startsWith(`${inst}__`))).toBe(true);
      expect(g.label).toBe('full_transformer_block');
    });
  });

  it('cuts the visible card count on a large build', () => {
    const { nodes, edges, metadata } = declaredGraph(12);

    const grouped = autoModularize(nodes, edges, { metadata });
    const visible = grouped.nodes.filter(
      (n: any) => !n.data?.parentId && !n.data?.repeatHidden,
    ).length;

    // 48 flat cards on one row becomes 12 block cards.
    expect(nodes.length).toBe(48);
    expect(visible).toBe(12);
  });

  it('is unchanged when the graph declares nothing', () => {
    const { nodes, edges } = declaredGraph(12);

    const withoutMetadata = autoModularize(nodes, edges);
    const withEmpty = autoModularize(nodes, edges, { metadata: {} });

    // Compare the grouping SHAPE, not the ids: moduleSeq is module-level, so
    // `auto-mod-N` differs between two calls by construction.
    expect(withEmpty.groups.map((g) => g.memberIds)).toEqual(
      withoutMetadata.groups.map((g) => g.memberIds),
    );
    expect(withEmpty.groups.map((g) => g.repeatCount)).toEqual(
      withoutMetadata.groups.map((g) => g.repeatCount),
    );
  });

  it('ignores a malformed template_instances instead of throwing', () => {
    const { nodes, edges } = declaredGraph(3);

    expect(() => autoModularize(nodes, edges, { metadata: { template_instances: 'nope' } })).not.toThrow();
    expect(() => autoModularize(nodes, edges, { metadata: { template_instances: [null, 7] } })).not.toThrow();
  });

  it('skips a declared instance whose members were deleted down to one', () => {
    const { edges, metadata } = declaredGraph(2);
    // Keep only a single member of the first instance.
    const nodes = [
      {
        id: 'encoder_block_00__n0',
        type: 'layerNode',
        position: { x: 0, y: 0 },
        data: { layerType: { id: 'linear', name: 'L', parameters: [] }, parameters: {} },
      },
    ] as any[];

    const result = autoModularize(nodes, edges, { metadata });

    expect(result.groups.filter((g) => g.moduleId === 'auto-mod-encoder_block_00')).toHaveLength(0);
  });
});
