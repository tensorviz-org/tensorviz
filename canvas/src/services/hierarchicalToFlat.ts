/**
 * Convert a backend `HierarchicalGraph` (root_nodes + root_edges + modules)
 * into the flat ModelGraph shape that ModelBuilder consumes via initialGraph.
 *
 * The canonical hierarchical payload from /api/popular-models/{id}/instantiate
 * and /api/templates/{id}/instantiate is:
 *
 *   {
 *     root_nodes: Node[],
 *     root_edges: Edge[],
 *     modules: {
 *       [moduleId]: {
 *         module_id, parent_id,
 *         internal_nodes: Node[],   // children of the module
 *         internal_edges: Edge[],
 *         input_node_id, output_node_id,
 *         parameters: { ..., repeat_count?, repeat_block_label? }
 *       }
 *     }
 *   }
 *
 * Backends declare a placeholder moduleNode at the root level (e.g.
 * `encoder_stack`, `layer1`) AND a matching entry in `modules` carrying the
 * internals. We must MERGE the module metadata onto that root placeholder
 * (layer count, repeat badge, group flag) — earlier this code skipped the
 * placeholder, so template modules rendered as bare, action-less boxes.
 *
 * ReactFlow wants every node in one flat list. Child nodes get
 * `data.parentId` pointing to their containing moduleNode; the canvas layout
 * pass (services/canvasLayout) maps that to `parentNode` + `extent: 'parent'`
 * only while the module is expanded, and hides children while collapsed.
 */
import { ModelGraph, ModelNode, ModelEdge } from '../types';

type RawNode = any;
type RawEdge = any;
type RawModule = {
  module_id?: string;
  parent_id?: string | null;
  internal_nodes?: RawNode[];
  internal_edges?: RawEdge[];
  input_node_id?: string | null;
  output_node_id?: string | null;
  parameters?: Record<string, any>;
};

export interface HierarchicalGraphPayload {
  root_nodes?: RawNode[];
  root_edges?: RawEdge[];
  nodes?: RawNode[];      // some endpoints use bare nodes/edges keys
  edges?: RawEdge[];
  modules?: Record<string, RawModule>;
}

const normalizeNode = (n: RawNode, parentId?: string | null): ModelNode => {
  const data = n.data || {};
  const layerType = data.layerType || data.layer_type;
  return {
    id: String(n.id),
    type: n.type || (layerType?.category === 'io' ? 'inputNode' : 'layerNode'),
    position: n.position || { x: 0, y: 0 },
    data: {
      ...data,
      layerType,
      parameters: data.parameters || {},
      // Preserve parent linkage for sub-flow rendering. Child nodes carry
      // parentId so the layout pass can anchor them inside the parent frame.
      parentId: parentId || data.parentId,
    },
  } as ModelNode;
};

const normalizeEdge = (e: RawEdge): ModelEdge => ({
  id: String(e.id),
  source: String(e.source),
  target: String(e.target),
  sourceHandle: e.sourceHandle || e.source_handle || undefined,
  targetHandle: e.targetHandle || e.target_handle || undefined,
  data: {
    sourceHandle: e.sourceHandle || e.source_handle || undefined,
    targetHandle: e.targetHandle || e.target_handle || undefined,
  },
});

/** Stamp module metadata (group flag, layer count, repeat badge) onto a
 *  moduleNode's data so the canvas can render it as a real collapsible block. */
const applyModuleMeta = (node: ModelNode, moduleId: string, mod: RawModule): void => {
  const params = mod.parameters || {};
  const childCount = (mod.internal_nodes || []).length;
  node.type = 'moduleNode';
  node.data = {
    ...node.data,
    parameters: { ...params, ...node.data.parameters },
    isGroup: true,
    expanded: node.data.expanded ?? false,
    moduleInstanceId: node.data.moduleInstanceId || moduleId,
    moduleTemplateId: node.data.moduleTemplateId || params.name || moduleId,
    moduleLayerCount: childCount,
    moduleStatus: node.data.moduleStatus || 'ok',
    moduleLayoutState: 'deterministic',
    moduleEntryNodeId: mod.input_node_id || node.data.moduleEntryNodeId,
    moduleExitNodeId: mod.output_node_id || node.data.moduleExitNodeId,
    repeatCount: typeof params.repeat_count === 'number' ? params.repeat_count : node.data.repeatCount,
    repeatBlockLabel: params.repeat_block_label || node.data.repeatBlockLabel,
    parentId: mod.parent_id || node.data.parentId,
  };
};

export function hierarchicalToFlat(payload: HierarchicalGraphPayload): ModelGraph {
  const nodes: ModelNode[] = [];
  const edges: ModelEdge[] = [];
  const byId = new Map<string, ModelNode>();

  // 1. Root-level nodes (top of model)
  const roots = payload.root_nodes ?? payload.nodes ?? [];
  for (const rn of roots) {
    const node = normalizeNode(rn);
    nodes.push(node);
    byId.set(node.id, node);
  }

  // 2. Modules: merge metadata onto the matching root placeholder (or
  //    synthesize a moduleNode if the backend didn't emit one), then append
  //    the internals as children carrying parentId.
  const modules = payload.modules || {};
  for (const [moduleId, mod] of Object.entries(modules)) {
    const moduleNodeId = mod.module_id || moduleId;
    let moduleNode = byId.get(moduleNodeId);
    if (moduleNode) {
      applyModuleMeta(moduleNode, moduleNodeId, mod);
    } else {
      moduleNode = {
        id: moduleNodeId,
        type: 'moduleNode',
        position: { x: 0, y: 0 },
        data: {
          layerType: {
            id: 'module',
            name: 'Module',
            displayName: (mod.parameters?.name as string) || moduleNodeId,
            description: 'nn.Module group',
            icon: 'Package',
            category: 'module',
            parameters: [],
          },
          parameters: { name: moduleNodeId },
        },
      } as ModelNode;
      applyModuleMeta(moduleNode, moduleNodeId, mod);
      nodes.push(moduleNode);
      byId.set(moduleNodeId, moduleNode);
    }

    // children
    for (const child of mod.internal_nodes || []) {
      const childNode = normalizeNode(child, moduleNodeId);
      if (!byId.has(childNode.id)) {
        nodes.push(childNode);
        byId.set(childNode.id, childNode);
      }
    }

    // internal edges
    for (const e of mod.internal_edges || []) edges.push(normalizeEdge(e));
  }

  // 3. Root edges
  const rootEdges = payload.root_edges ?? payload.edges ?? [];
  for (const e of rootEdges) edges.push(normalizeEdge(e));

  return { nodes, edges };
}
