/**
 * A project frame's children reach the reconciler as children.
 *
 * Measured in the running extension, three times over one afternoon: an open
 * `TinyLM` said "9 layers" and drew an empty box. The children were in the
 * document, parented correctly, in the right order. The document says which frame
 * a child belongs to with `parentNode`; the canvas — `reconcileModuleView`,
 * `repeatBlocks`, the grouping code — reads `data.parentId`. Nothing translated
 * between the two, so the reconciler saw nine top-level cards and the frame it
 * was supposed to fill stayed empty.
 *
 * Fixed at the seam where the document becomes a ModelGraph, so the wire contract
 * keeps its field and every render path keeps theirs.
 */
import { applyProjection } from '../../ide/projection';
import { reconcileModuleView } from '../../services/canvasLayout';
import type { ModelNode } from '../../types';

const card = (id: string, extra: Record<string, unknown> = {}) => ({
  id,
  type: 'layerNode',
  position: { x: 0, y: 0 },
  data: { label: id, layerType: { id: 'custom_code', name: 'model' }, parameters: { _kind: 'model' } },
  ...extra,
});

const document = {
  experiment_id: 'exp:1',
  graph_id: 'g',
  name: 'train',
  nodes: [
    { ...card('lm'), type: 'moduleNode', data: { ...card('lm').data, expanded: true, isGroup: true, moduleInstanceId: 'lm', moduleTemplateId: 'TinyLM', moduleLayerCount: 2 } },
    card('lm/embed', { parentNode: 'lm' }),
    card('lm/head', { parentNode: 'lm' }),
  ],
  edges: [],
  spanmap: {},
  expanded: ['lm'],
};

describe('a project document’s children', () => {
  it('carry data.parentId, which is the field the canvas reads', () => {
    const merged = applyProjection(document as never, []);
    expect(merged).not.toBeNull();
    const byId = new Map(merged!.nodes.map((node) => [node.id, node]));
    expect(byId.get('lm/embed')!.data.parentId).toBe('lm');
    expect(byId.get('lm/head')!.data.parentId).toBe('lm');
    // The frame itself is nobody's child.
    expect(byId.get('lm')!.data.parentId).toBeUndefined();
  });

  it('are attached to their frame by the reconciler, not laid out beside it', () => {
    // This is the assertion that was false for the whole life of the project
    // canvas. Without it the frame reports a count and draws nothing.
    const merged = applyProjection(document as never, []);
    const { nodes } = reconcileModuleView(merged!.nodes as ModelNode[], merged!.edges);
    const children = nodes.filter((node) => node.parentNode === 'lm' || node.data.parentId === 'lm');
    expect(children.map((node) => node.id).sort()).toEqual(['lm/embed', 'lm/head']);
    for (const child of children) {
      expect(child.hidden).not.toBe(true);
    }
  });

  it('does not overwrite a parentId the document already set', () => {
    const explicit = {
      ...document,
      nodes: [document.nodes[0], { ...card('lm/x', { parentNode: 'lm' }), data: { ...card('lm/x').data, parentId: 'other' } }],
    };
    const merged = applyProjection(explicit as never, []);
    expect(merged!.nodes.find((node) => node.id === 'lm/x')!.data.parentId).toBe('other');
  });
});
