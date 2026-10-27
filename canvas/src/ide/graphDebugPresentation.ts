import type { ModelEdge, ModelGraph, ModelNode } from '../types';
import type { ProjectionPayload } from './protocol.generated';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import type { GraphDocumentV2 } from '../types/graphDocument';
import { assignmentContext, previousData } from './assignmentContext';

const historicalId = (id: string) => `previous-run:${id}`;

export function showsPreviousRunContext(payload: ProjectionPayload | null, nodeCount: number): boolean {
  return !!payload?.debug?.previousDocument && (payload.traced || nodeCount === 0);
}

/** Compose display context without changing the authoritative projection. Old
 * edges remain dashed historical facts, even when their endpoints correspond. */
export function graphDebugPresentation(payload: ProjectionPayload, current: ModelGraph,
  existing: readonly ModelNode[]): { graph: ModelGraph; existing: ModelNode[] } {
  const context = payload.debug;
  const changes = new Map(context?.changed.map(change => [change.nodeId, change.details]));
  const pairs = new Map(context?.nodePairs.map(pair => [pair.beforeId, pair.afterId]));
  const reverse = new Map(context?.nodePairs.map(pair => [pair.afterId, pair.beforeId]));
  let graph: ModelGraph = { ...current, nodes: current.nodes.map(node => ({ ...node,
    data: { ...(payload.greyed && payload.traced ? previousData(node) : { ...node.data,
      debugEvidence: payload.traced ? 'current' as const : 'source' as const, debugChanges: changes.get(node.id) }),
      debugPreviousId: reverse.get(node.id) } })) };
  // A current source projection is the architecture to show, even after a
  // capture: a prior run never replaces it, and only fills an empty projection.
  if (context?.previousDocument && showsPreviousRunContext(payload, current.nodes.length)) {
    const previous = graphDocumentV2ToModelGraph(context.previousDocument as GraphDocumentV2);
    const id = (nodeId: string) => pairs.get(nodeId) ?? historicalId(nodeId);
    const ghosts = previous.nodes.filter(node => !pairs.has(node.id)).map(node => ({ ...node, id: id(node.id),
      parentNode: node.parentNode ? id(node.parentNode) : undefined,
      data: { ...previousData(node), debugPreviousId: node.id,
        capturePorts: node.data.capturePorts?.map(port => ({ ...port, nodeId: id(node.id) })),
        parentId: node.data.parentId ? id(node.data.parentId) : undefined,
        moduleInstanceId: node.data.moduleInstanceId ? id(node.data.moduleInstanceId) : undefined,
        moduleMembers: node.data.moduleMembers?.map(id),
        moduleEntryNodeId: node.data.moduleEntryNodeId ? id(node.data.moduleEntryNodeId) : undefined,
        moduleExitNodeId: node.data.moduleExitNodeId ? id(node.data.moduleExitNodeId) : undefined,
        capturePassThrough: undefined,
      } }));
    const oldNodes = new Map(previous.nodes.map(node => [node.id, node]));
    const newNodes = new Map(graph.nodes.map(node => [node.id, node]));
    const port = (nodeId: string, handle: string | null | undefined) => {
      if (!pairs.has(nodeId)) return handle;
      const old = oldNodes.get(nodeId)?.data.capturePorts?.find(port => port.id === handle);
      const matches = old ? newNodes.get(id(nodeId))?.data.capturePorts?.filter(port =>
        port.direction === old.direction && port.argumentKind === old.argumentKind
        && JSON.stringify(port.paths) === JSON.stringify(old.paths)) : [];
      return matches?.length === 1 ? matches[0].id : undefined;
    };
    const edges: ModelEdge[] = previous.edges.flatMap(edge => {
      if (pairs.has(edge.source) && pairs.has(edge.target)) return [];
      const sourceHandle = port(edge.source, edge.sourceHandle), targetHandle = port(edge.target, edge.targetHandle);
      if ((pairs.has(edge.source) && !sourceHandle) || (pairs.has(edge.target) && !targetHandle)) return [];
      return [{ ...edge, id: historicalId(edge.id), source: id(edge.source), target: id(edge.target), sourceHandle, targetHandle,
        data: { ...edge.data, debugPrevious: true }, label: 'Previous run',
        style: { ...edge.style, strokeDasharray: '5 5', opacity: .4 } }];
    });
    // Without source nodes keep the captured structure, explicitly stale. The
    // source document still owns edits, navigation and export.
    graph = { ...graph, nodes: payload.traced ? [...graph.nodes, ...ghosts] : ghosts,
      edges: payload.traced ? [...graph.edges, ...edges] : edges };
  }
  // Accepted nested assignments stay on the calls they replace, inside the last
  // run's hierarchy, when the rest of the source provably still matches it.
  const assigned = !payload.traced && context?.previousDocument ? assignmentContext(context.previousDocument, payload.document) : null;
  if (assigned) graph = assigned.graph;
  if (payload.greyed) graph = { ...graph, edges: graph.edges.map(edge => ({ ...edge,
    data: { ...edge.data, debugPrevious: true }, style: { ...edge.style, strokeDasharray: '5 5', opacity: .4 } })) };
  // Transfer only proven counterparts, including a ghost retained by an earlier
  // partial run. Never use runtime ordinals or tensor shapes as identity.
  const oldById = new Map(existing.map(node => [node.id, node]));
  const oldByPreviousId = new Map(existing.filter(node => node.data.debugPreviousId)
    .map(node => [node.data.debugPreviousId!, node]));
  const matched = graph.nodes.flatMap(node => {
    const key = node.data.debugPreviousId;
    const old = oldById.get(node.id) ?? (key ? oldById.get(key) ?? oldByPreviousId.get(key) : undefined);
    return old ? [{ ...old, id: node.id }] : [];
  });
  return { graph, existing: matched };
}
