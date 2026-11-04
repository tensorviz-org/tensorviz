import { object, type JsonObject, type McpGraphService, type ModelRef } from '@tensorviz/graph-host/mcpGraph';

/** Attach a bounded first graph page using the same node/edge serializers as MCP. */
export function agentGraphContext(graph: McpGraphService, ref: ModelRef, maxBytes = 64 * 1024): JsonObject {
  const { model, nodes: allNodes, edges: allEdges } = graph.require(ref);
  const page = { ...graph.reference(model), target: model.target, traced: model.projection.traced,
    projectionDigest: model.projection.projectionDigest, truncated: model.projection.truncated ?? false,
    nodeCount: allNodes.length, edgeCount: allEdges.length,
    ...(object(object(object(model.projection.document).semantic).metadata).capture
      ? { capture: object(object(object(model.projection.document).semantic).metadata).capture } : {}),
    nodes: allNodes.slice(0, 200).map(node => graph.node(node, model.projection)),
    edges: allEdges.slice(0, 200).map(edge => graph.edge(edge)) };
  const nodes = page.nodes as JsonObject[], edges = page.edges as JsonObject[];
  const result = { ...page, nodes: [] as JsonObject[], edges: [] as JsonObject[],
    nextNodeOffset: 0 as number | null, nextEdgeOffset: 0 as number | null, contextLimited: true };
  // Reserve space for commas and continuation offsets before taking whole records.
  let remaining = maxBytes - Buffer.byteLength(JSON.stringify(result)) - 32;
  if (remaining < 0) return { ...ref, contextLimited: true, omitted: 'Graph metadata exceeds the context byte limit.',
    nextNodeOffset: 0, nextEdgeOffset: 0 };
  const take = (items: JsonObject[]) => {
    const included: JsonObject[] = [];
    for (const item of items) {
      const bytes = Buffer.byteLength(JSON.stringify(item)) + 1;
      if (bytes > remaining) break;
      remaining -= bytes; included.push(item);
    }
    return included;
  };
  result.nodes = take(nodes); result.edges = take(edges);
  result.nextNodeOffset = result.nodes.length < Number(page.nodeCount) ? result.nodes.length : null;
  result.nextEdgeOffset = result.edges.length < Number(page.edgeCount) ? result.edges.length : null;
  result.contextLimited = result.nextNodeOffset !== null || result.nextEdgeOffset !== null;
  return result;
}
