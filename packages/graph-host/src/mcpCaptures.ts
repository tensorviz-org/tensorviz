import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { CaptureInspector, captureQuerySchema } from './captureInspection';
import { McpGraphService, graphToolResult, modelRefSchema } from './mcpGraph';
import { computationQuerySchema, computationFindSchema } from './computationQuery';
import { computationSourceContext, type ComputationSourceReader } from './computationSource';
import { computationChangeSchema } from './computationChange';
import { getModelContext, MODEL_CONTEXT_GUIDANCE, modelContextQuerySchema } from './modelContext';

export function registerMcpCaptureTools(server: McpServer, graph: McpGraphService, inspector: CaptureInspector,
  readSource?: ComputationSourceReader): void {
  const annotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
  server.registerTool('get_model_context', {
    description: 'Read compact canonical architecture facts for the exact current model revision from its latest completed capture: aliases, unique registered parameter counts, full retained per-module call lists, ordered tensor shapes, modes, unknowns and coverage. Defaults to 4000 bytes; focus is an exact module ID, path, qualified type or invocation ID. Included call lists are atomic; use focus/larger maxBytes for omitted modules, inspect_capture for oversized lists. Source-only, stale-source, evicted and no-capture states return explicit unavailability. Awaits the host freshness guard; never executes model code, starts a capture or edits. ' + MODEL_CONTEXT_GUIDANCE,
    inputSchema: { ...modelRefSchema, ...modelContextQuerySchema.shape }, annotations,
  }, ({ modelId, revision, focus, maxBytes }) => graph.currentToolResult(() => getModelContext(graph, inspector, { modelId, revision }, { focus, maxBytes })));
  server.registerTool('find_computation', {
    description: 'Find exact computation anchors by case-sensitive literal substrings of captured module paths/type names, definition names, operation targets or observed parameter/buffer registration names. Returns every matching entity with bounded pages and exact inspection pointers; follow nextOffset. Use a returned ref with query_computation instead of guessing an ID from a name. Repeated names stay separate; aliases of one recorded module stay together. No match does not prove absence. Describes retained evidence, not current runtime state. Never reads new files, executes or edits.',
    inputSchema: { modelId: modelRefSchema.modelId, ...computationFindSchema.shape }, annotations,
  }, args => graphToolResult(() => {
    const model = graph.requireModel(args.modelId), { modelId, ...query } = args;
    return { ...graph.reference(model), current: model.current, ...inspector.findComputation(model.target, query) };
  }));
  server.registerTool('analyze_computation_change', {
    description: 'Analyze hypothetical shape changes to exact values in a retained canonical capture. Traverses bounded downstream paths and applies a closed set of deterministic ATen dimension rules. Returns conditional compatible/incompatible/unknown checks beside unchanged observations. Arguments, other operand shapes and control flow are held fixed; unsupported operators propagate unknown shapes. Aliasing, missing paths, dtype/layout, numerics and training are not certified. Follow pages and explicit boundaries; use query_computation for optional exact source. Never edits or executes.',
    inputSchema: { modelId: modelRefSchema.modelId, ...computationChangeSchema.shape }, annotations,
  }, args => graphToolResult(() => {
    const model = graph.requireModel(args.modelId), { modelId, ...query } = args;
    return { ...graph.reference(model), current: model.current,
      ...inspector.analyzeComputation(model.target, query) };
  }));
  server.registerTool('query_computation', {
    description: 'Trace a bounded upstream/downstream slice of a retained canonical capture. Anchor exact operation/value IDs, or an invocation/module/definition to select its recorded operations. Parameter/buffer anchors use the exact registration name as id and select all recorded value versions carrying that association; tied names remain on the values. Missing state means unobserved, not absent from the model. This is not a complete state inventory. Each depth step crosses an operation-value or explicit value-version edge. Preserves repeated operand paths, mutation effects, ownership, source bindings and full-record inspection pointers. includeSource attaches bounded workspace source excerpts only when the current file matches an exact captured binding. Pages include boundaries and omitted counts; follow nextOffset, then re-anchor boundary refs. Evidence describes only the recorded execution/traced program, never all branches or current runtime state. Use list_captures for capture identities and find_computation for exact anchors from recorded names; inspect_capture exposes original records. Never executes Python or grants edits.',
    inputSchema: { modelId: modelRefSchema.modelId, ...computationQuerySchema.shape, includeSource: z.boolean().default(false) }, annotations,
  }, args => graphToolResult(async () => {
    const model = graph.requireModel(args.modelId), ref = graph.reference(model), { modelId, includeSource, ...query } = args;
    const { page, source } = inspector.queryComputation(model.target, query);
    const sourceContext = includeSource ? await computationSourceContext(page, source, ref,
      model.current ? readSource : undefined, model.current ? 'reader_unavailable' : 'graph_stale') : undefined;
    if (sourceContext) {
      // Async reads cannot keep an evicted capture or a foreign model alive.
      const current = graph.requireModel(args.modelId);
      inspector.assertRetained(current.target, query.captureId, query.sourceId);
      if (sourceContext.filesRead && (!current.current || graph.reference(current).revision !== ref.revision)) {
        sourceContext.status = 'unavailable'; sourceContext.reason = 'stale_target';
        sourceContext.evidence = null;
        sourceContext.omittedEntries += sourceContext.entries.length; sourceContext.entries = [];
      }
    }
    return { ...ref, current: model.current, ...page, ...(sourceContext ? { sourceContext } : {}) };
  }));
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
