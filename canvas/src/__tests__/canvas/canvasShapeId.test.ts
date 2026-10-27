import { nodeIdToShapeId, edgeIdToShapeId, shapeIdToNodeId } from '../../canvas/canvasShapeId';

test('all model identities survive a canvas round trip without collisions', () => {
  const ids = ['branch.fc', 'branch_fc', 'block:1', 'block_1', 'add@Net#0', 'add_Net_0', '层', 'branch%2Efc', 'edge:e1'];
  expect(new Set(ids.map(nodeIdToShapeId)).size).toBe(ids.length);
  expect(ids.map(id => shapeIdToNodeId(nodeIdToShapeId(id)))).toEqual(ids);
});

test('edges cannot replace a similarly named layer or another connection', () => {
  const edge = { id: 'a.b', source: 'branch.fc', target: 'output' };
  const ids = [
    edgeIdToShapeId(edge),
    edgeIdToShapeId({ ...edge, id: 'a_b' }),
    edgeIdToShapeId({ ...edge, source: 'branch_fc' }),
    nodeIdToShapeId('edge_a_b'),
  ];
  expect(new Set(ids).size).toBe(ids.length);
});
