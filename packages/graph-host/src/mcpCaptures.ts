import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { CaptureInspector, captureQuerySchema } from './captureInspection';
import { McpGraphService, graphToolResult, modelRefSchema } from './mcpGraph';

export function registerMcpCaptureTools(server: McpServer, graph: McpGraphService, inspector: CaptureInspector): void {
  const annotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
  server.registerTool('list_captures', {
    description: 'List the selected model’s latest attempt and independently retained complete, partial and displayed captures. Includes browser/manual runs. Readable after source changes; retention lasts only in this host session. Never executes Python.',
    inputSchema: { modelId: modelRefSchema.modelId }, annotations,
  }, args => graphToolResult(() => {
    const model = graph.requireModel(args.modelId);
    return { ...graph.reference(model), current: model.current, ...inspector.list(model.target) };
  }));
  server.registerTool('inspect_capture', {
    description: 'Page the immutable canonical capture tree, including operations, invocations, ordered arguments, literals, values, state, effects, coverage and diagnostics. Start at path=[]; open an object/array child by appending its key, or select an entity using the node captureRef kind/semanticId. Containers return counts and scalar children return exact typed values; use nextOffset for every page. Paths are relative to entity when supplied. Records stay inspectable after source changes; focus still requires the current graph revision. Never executes Python.',
    inputSchema: { modelId: modelRefSchema.modelId, ...captureQuerySchema.shape }, annotations,
  }, args => graphToolResult(() => {
    const model = graph.requireModel(args.modelId);
    const { modelId, ...query } = args;
    return { ...graph.reference(model), current: model.current, ...inspector.inspect(model.target, query) };
  }));
}
