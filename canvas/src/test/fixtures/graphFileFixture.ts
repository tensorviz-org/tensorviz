import { ModelGraph } from '../../types';

export function nestedGraph(): ModelGraph {
  const layer = (id: string, type: string, parentId?: string) => ({
    id, type: type === 'module' ? 'moduleNode' : 'layerNode', position: { x: id.length * 50, y: 80 },
    data: {
      layerType: { id: type, name: type, displayName: type, description: '', category: 'misc' as const, icon: 'Box', parameters: [] },
      parameters: { in_features: 8, out_features: 4 },
      ...(parentId ? { parentId } : {}),
      ...(type === 'module' ? { expanded: false, moduleInstanceId: id } : {}),
    },
  });
  return {
    nodes: [layer('input', 'input'), layer('block', 'module'), layer('inner', 'module', 'block'), layer('fc', 'linear', 'inner'), layer('output', 'output')],
    edges: [{ id: 'a', source: 'input', target: 'fc', sourceHandle: 'y', targetHandle: 'x' }, { id: 'b', source: 'fc', target: 'output' }],
  };
}
