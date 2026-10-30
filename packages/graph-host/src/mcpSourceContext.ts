import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { McpGraphService, GraphToolError, graphToolResult, modelRefSchema } from './mcpGraph';
import type { CaptureInspector } from './captureInspection';
import { sourceContextQuerySchema, sourceCaptureContext, type SourceContextReader } from './sourceContext';

export function registerMcpSourceContextTool(server: McpServer, graph: McpGraphService, inspector: CaptureInspector, read: SourceContextReader): void {
  const captureId = z.string().min(1).max(128).optional();
  server.registerTool('query_source_context', {
    description: 'Follow bounded lexical source/configuration dependencies in the current model and graph-recorded workspace Python files. Select file plus symbol (optional line), or anchorIds plus indexId from a previous page. Optional assumeValue with a downstream symbol selector analyzes one direct integer definition and supported integer expressions; before/after scalar and call-argument results are conditional source arithmetic, never observed values. Defaults, ambiguous/conditional bindings and unsupported expressions remain unknown. Imports, assignments, argument positions, default-only and possible receiver-attribute links remain distinct from runtime facts. Follow nextOffset and re-anchor explicit boundaries. Optional captureId/sourceId adds exact source correspondence to retained canonical records. With assumeValue, matched initializer observations and baseline state can supply conditional Linear/scalar-LayerNorm parameter dimensions and recorded sharing; unknowns stay explicit. Proven direct reshape/view argument maps also supply operationImpact dimensions and localCheck under fixed captured operand and referenced tensor shapes; exact observed tensor-dimension reads are supported, other dynamic dimensions stay unknown. computationImpact separately propagates supported constructor state, owning standard LayerNorm dimensions, observed reshape reads and flatten arguments on the bounded recorded path. Inspect its substitutions, unknowns and coverage; other arguments and control flow are held fixed. Neither result is a verdict about the complete source edit. This establishes neither forward/checkpoint compatibility nor runtime applicability. Reads live base buffers, never a pending proposal. Partial file coverage and unresolved relationships are explicit. Never imports or executes project code, edits files, or expands file access.',
    inputSchema: { ...modelRefSchema, ...sourceContextQuerySchema.shape, captureId, sourceId: captureId },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, args => graphToolResult(async () => {
    const { modelId, revision, captureId, sourceId, ...raw } = args, ref = { modelId, revision };
    const query = sourceContextQuerySchema.safeParse(raw);
    if (!query.success || !!captureId !== !!sourceId) throw new GraphToolError('INVALID_QUERY', 'Use a bounded source selector and both capture identities when linking a capture.');
    const { model } = graph.require(ref);
    if (captureId && sourceId) inspector.retained(model.target, captureId, sourceId);
    const page = await read(ref, query.data);
    const current = graph.require(ref);
    const capture = captureId && sourceId ? inspector.retained(current.model.target, captureId, sourceId) : undefined;
    return { ...ref, ...page, ...(capture ? { captureContext: sourceCaptureContext(page, capture) } : {}) };
  }));
}
