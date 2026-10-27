import { searchGraph } from '../../canvas/graphSearch';
import type { ModelNode } from '../../types';

const node = (id: string, extra: Partial<ModelNode['data']> = {}, hidden = false): ModelNode => ({
  id,
  type: 'layerNode',
  position: { x: 0, y: 0 },
  hidden,
  data: {
    layerType: { id: 'conv2d', name: 'Conv2d', displayName: 'Conv2D', category: 'conv', icon: 'Box', description: '', parameters: [] },
    parameters: {},
    ...extra,
  },
} as ModelNode);

/**
 * Why these matter: the graph is only navigable by eye up to about a screenful.
 * Past that, "which layer is the 256-wide one" has no answer on this canvas, and
 * a search that forces every container open to show a hit has given up the
 * hierarchy to answer the question.
 */
describe('searchGraph', () => {
  it.each(['Max pool', 'type=Max pool', 'aten.max_pool2d', 'type=aten.max_pool2d'])('finds imported operations by readable or exact names: %s', query => {
    const pool = node('fx_42', { layerType: { name: 'aten.max_pool2d_with_indices.default' } as ModelNode['data']['layerType'] }, true);
    pool.data.parentId = 'block';
    const result = searchGraph([node('block'), pool], query);
    expect(result.hits.map(hit => hit.id)).toEqual(['fx_42']);
    expect(result.groups.map(group => group.visibleId)).toEqual(['block']);
  });

  it('matches a layer type and reports which text matched', () => {
    const result = searchGraph([node('stem'), node('pool', { layerType: { id: 'avgpool', name: 'AvgPool', displayName: 'AvgPool', category: 'pool', icon: 'Box', description: '', parameters: [] } })], 'type=Conv2d');
    expect(result.hits.map(hit => hit.id)).toEqual(['stem']);
    expect(result.hits[0].field).toBe('type');
    expect(result.hits[0].matched).toBe('Conv2D');
  });

  it('searches the tensor text the card actually shows', () => {
    const nodes = [
      node('expand', { outputShape: [2, 16, 256] }),
      node('project', { outputShape: [2, 16, 64] }),
    ];
    expect(searchGraph(nodes, 'shape=256').hits.map(h => h.id)).toEqual(['expand']);
    expect(searchGraph(nodes, 'shape=\\[2, 16, 64\\]').hits.map(h => h.id)).toEqual(['project']);
  });

  it('matches a constructor parameter by name', () => {
    const nodes = [node('a', { parameters: { padding: 1 } }), node('b', { parameters: { padding: 0 } })];
    expect(searchGraph(nodes, 'attr:padding=1').hits.map(h => h.id)).toEqual(['a']);
  });

  it('reports a hit inside a closed module on the module, not on nothing', () => {
    // The whole point of containers: searching must not force them open.
    const nodes = [
      node('stage1', { parameters: {} }),
      node('stage1.conv1', { parentId: 'stage1' }, true),
      node('stage1.conv2', { parentId: 'stage1' }, true),
    ];
    const result = searchGraph(nodes, 'type=Conv2d');
    expect(result.hits).toHaveLength(3);
    expect(result.groups.map(group => group.visibleId)).toEqual(['stage1']);
    expect(result.groups[0].hits.map(hit => hit.id)).toEqual(['stage1', 'stage1.conv1', 'stage1.conv2']);
    expect([...result.visibleIds]).toEqual(['stage1']);
  });

  it('reports a hit on itself once its module is open', () => {
    const nodes = [
      node('stage1'),
      node('stage1.conv1', { parentId: 'stage1' }, false),
    ];
    const result = searchGraph(nodes, 'type=Conv2d');
    expect(result.groups.map(group => group.visibleId).sort()).toEqual(['stage1', 'stage1.conv1']);
  });

  it('rolls a doubly nested hit up to the outermost closed frame', () => {
    const nodes = [
      node('stage1'),
      node('stage1.block', { parentId: 'stage1' }, true),
      node('stage1.block.conv', { parentId: 'stage1.block' }, true),
    ];
    expect(searchGraph(nodes, 'type=Conv2d').visibleIds).toEqual(new Set(['stage1']));
  });

  it('says a regex is broken rather than reporting zero matches', () => {
    const result = searchGraph([node('stem')], 'type=[unclosed');
    expect(result.error).toBeTruthy();
    expect(result.hits).toHaveLength(0);
  });

  it('treats an empty query as no search at all', () => {
    expect(searchGraph([node('stem')], '   ').hits).toHaveLength(0);
    expect(searchGraph([node('stem')], '   ').error).toBeUndefined();
  });

  it('survives a cycle in parent links instead of hanging', () => {
    const a = node('a', { parentId: 'b' }, true);
    const b = node('b', { parentId: 'a' }, true);
    expect(() => searchGraph([a, b], 'type=Conv2d')).not.toThrow();
  });
});
