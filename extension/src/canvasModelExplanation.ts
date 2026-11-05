import { createHash } from 'node:crypto';
import { z } from 'zod';
import type { LocalBrowserModelExplanationRequest, LocalBrowserTargetIdentity, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';
import { isNodeExplanation, isNodeSourceDocumentation, type NodeSourceDocumentation, type ModelExplanationState, type NodeExplanation, type NodeExplanationCitation, type NodeExplanationState } from '@tensorviz/contracts/nodeExplanation';
import { GraphToolError, graphToolResult, object, type McpModel, type JsonObject, type ModelRef } from '@tensorviz/graph-host/mcpGraph';
import type { CanvasAgentRoute } from './canvasAgent';
import { explanationAbortable, nodeExplanationSubmission, type CanvasNodeExplanationOptions } from './canvasNodeExplanation';
import { CANVAS_AGENT_PROMPTS } from './canvasAgentPrompts';
import type { HostToolset } from './mcpHost';

export interface CanvasModelExplanationOptions extends Pick<CanvasNodeExplanationOptions, 'route' | 'read' | 'runtime' | 'tools'> {
  prepareRoute?(signal: AbortSignal): Promise<CanvasAgentRoute>;
  current(request: LocalBrowserModelExplanationRequest, revision?: string): boolean;
  reference(request: LocalBrowserModelExplanationRequest): ModelRef;
  refusal?(request: LocalBrowserModelExplanationRequest): { ref: ModelRef; detail: string } | undefined;
  prepare(request: LocalBrowserModelExplanationRequest, nodeId: string, signal: AbortSignal): Promise<{ context: unknown; source: { file: string; startLine: number } }>;
  publish(viewId: string, target: LocalBrowserTargetIdentity, state: ModelExplanationState): void;
  publishNode(viewId: string, target: LocalBrowserTargetIdentity, state: NodeExplanationState): void;
  documentation?(request: LocalBrowserModelExplanationRequest, nodeId: string, signal: AbortSignal): Promise<NodeSourceDocumentation | undefined>;
  changed(): void;
  /** Bounded batch deadline, injectable by deterministic host tests. */
  timeoutMs?: number;
}
interface Run {
  request: LocalBrowserModelExplanationRequest; ref: ModelRef; route: CanvasAgentRoute;
  controller: AbortController; preparingRoute: boolean; state: ModelExplanationState; documentation: Map<string, NodeSourceDocumentation>;
}
interface Batch {
  run: Run; nodeIds: Set<string>; evidence: Map<string, NodeExplanationCitation>; reads: number; bytes: number;
  results: Map<string, NodeExplanation>; exposed: Set<string>; submitted: boolean; submissionAttempted: boolean;
}
const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const targetKey = (target: LocalBrowserTargetIdentity) => JSON.stringify(target);
const routeKey = (route: CanvasAgentRoute) => JSON.stringify([route.provider, route.resolvedModel, route.settings]);
const batchSubmission = { nodes: z.array(z.object({ nodeId: z.string().min(1).max(1024), ...nodeExplanationSubmission }).strict()).min(1).max(6) };
const MAX_BYTES = 48 * 1024;
const BATCH_SIZE = 6;
const FAILURE_DETAILS = {
  source_unavailable: 'Some nodes have no readable source mapping.',
  context_limit: 'Some context exceeded the explanation limit.',
  source_changed: 'Some source evidence changed during explanation.',
  provider_failed: 'The agent could not finish a batch.',
  no_submission: 'The agent returned no verified explanation for some nodes.',
  invalid_submission: 'Some generated explanations failed source validation.',
} as const;
type FailureReason = keyof typeof FAILURE_DETAILS;

/** Keeps the model pass on exactly the same serializer facts used by MCP and chat. */
export function modelExplanationNodeContext(model: McpModel, inspection: JsonObject, nodeId: string) {
  const span = object(model.projection.spanmap[nodeId]), start = span.start;
  if (typeof span.file !== 'string' || !Array.isArray(start) || !Number.isSafeInteger(start[0]) || start[0] < 1) throw new Error('No source mapping.');
  const semantic = object(object(model.projection.document).semantic), all = Array.isArray(semantic.nodes) ? semantic.nodes.map(object) : [];
  const original = all.find(node => node.node_id === nodeId), params = object(original?.params);
  const edges = Array.isArray(inspection.edges) ? inspection.edges.slice(0, 16).map(object) : [];
  const neighborIds = new Set(edges.flatMap(edge => [edge.sourceNodeId, edge.targetNodeId])); neighborIds.delete(nodeId);
  const node = Array.isArray(inspection.nodes) ? object(inspection.nodes[0]) : {};
  const parameters = JSON.stringify(node.parameters ?? {}).length <= 1024 ? node.parameters : undefined;
  return { source: { file: span.file, startLine: start[0] }, context: {
    node: { nodeId, kind: node.kind, operation: node.operation, inputPorts: node.inputPorts, outputPorts: node.outputPorts,
      shapes: node.shapes, source: node.source, parameters, parametersOmitted: parameters === undefined,
      callable: params._class, call: typeof params._raw_call === 'string' && params._raw_call.length <= 1024 ? params._raw_call : undefined },
    edges, edgeCount: inspection.edgeCount, connectionsLimited: Number(inspection.edgeCount) > edges.length,
    neighbors: all.filter(node => neighborIds.has(node.node_id)).map(node => ({ nodeId: node.node_id, kind: node.kind, operation: node.op })),
  } };
}

/** Explicit graph enrichment, with presentation-only authority and revision-scoped results. */
export class CanvasModelExplanation {
  private active?: Run;
  private disposed = false;
  private running: Promise<void> = Promise.resolve();
  private readonly cache = new Map<string, NodeExplanation>();
  constructor(private readonly options: CanvasModelExplanationOptions) {}
  get busy(): boolean { return !!this.active; }
  cached(selection: LocalBrowserViewSelectionMessage, ref: ModelRef, route: CanvasAgentRoute): NodeExplanation | undefined {
    return this.cache.get(this.key(selection.target, ref, selection.nodeIds[0], route));
  }
  private key(target: LocalBrowserTargetIdentity, ref: ModelRef, nodeId: string, route: CanvasAgentRoute): string {
    return JSON.stringify([target.workspaceFolder, target.sourceDigest, ref, nodeId, routeKey(route), CANVAS_AGENT_PROMPTS.model_explanation.version]);
  }
  private current(run: Run): boolean {
    const route = this.options.route();
    return !this.disposed && this.active === run && !run.controller.signal.aborted && !route.busy
      && (run.preparingRoute ? route.provider === run.route.provider : route.ready && routeKey(route) === routeKey(run.route))
      && this.options.current(run.request, run.ref.revision);
  }
  private check(batch: Batch): void { if (!this.current(batch.run)) throw new GraphToolError('STALE_TARGET', 'The model, source or agent changed.'); }
  private publish(run: Run): void { this.options.publish(run.request.viewId, run.request.target, structuredClone(run.state)); }
  private stop(detail: string, status: 'cancelled' | 'unavailable' = 'cancelled'): void {
    const run = this.active; if (!run) return;
    run.state = { ...run.state, status, detail }; run.controller.abort(); this.active = undefined; this.publish(run); this.options.changed();
  }
  invalidate(): void { this.stop('The graph changed. Explain the current model to continue.'); }
  disconnect(viewId: string): void { if (this.active?.request.viewId === viewId) this.stop('The graph view disconnected.'); }
  refresh(): void { if (this.active && !this.current(this.active)) this.stop('The model or agent changed. Explain again when the agent is available.', 'unavailable'); }
  dispose(): void { this.stop('The graph closed.'); this.disposed = true; this.cache.clear(); }
  request(request: LocalBrowserModelExplanationRequest): void {
    if (this.disposed) return;
    if (request.action === 'cancel') {
      const run = this.active;
      if (run && run.request.requestId === request.requestId && run.request.viewId === request.viewId && targetKey(run.request.target) === targetKey(request.target)) this.stop('Explanation stopped. Completed nodes remain available.');
      return;
    }
    if (this.active?.request.requestId === request.requestId && this.active.request.viewId === request.viewId && targetKey(this.active.request.target) === targetKey(request.target)) return;
    if (!request.nodeIds.length || request.nodeIds.length > 128 || new Set(request.nodeIds).size !== request.nodeIds.length) return;
    if (!this.options.current(request)) { this.refuse(request); return; }
    this.stop('A new explanation started.');
    let ref: ModelRef;
    try { ref = this.options.reference(request); } catch { this.refuse(request); return; }
    const route = this.options.route();
    const run: Run = { request: structuredClone(request), ref, route, controller: new AbortController(), preparingRoute: !!this.options.prepareRoute,
      documentation: new Map(), state: { requestId: request.requestId, revision: ref.revision, status: 'loading', completed: 0, total: request.nodeIds.length,
        ...(!route.ready && this.options.prepareRoute ? { detail: 'Connecting to agent…' } : {}) } };
    if ((!route.ready && !this.options.prepareRoute) || route.busy) {
      run.state.status = 'unavailable'; run.state.detail = route.busy ? 'The agent is busy. Explain when its current task finishes.' : route.detail ?? 'Connect an agent in chat, then explain this model.'; this.publish(run); return;
    }
    this.active = run; this.publish(run); this.options.changed();
    const previous = this.running;
    this.running = (async () => { await previous; if (this.current(run)) await this.start(run); else if (this.active === run) this.refresh(); })().catch(() => {});
  }
  private async start(run: Run): Promise<void> {
    if (this.options.prepareRoute) {
      const timeout = setTimeout(() => { if (this.active === run) this.stop('The configured agent did not finish connecting. Check its sign-in and try again.', 'unavailable'); }, 45000);
      try {
        const route = await explanationAbortable(this.options.prepareRoute(run.controller.signal), run.controller.signal);
        if (!this.current(run)) { if (this.active === run) this.refresh(); return; }
        run.route = route; run.preparingRoute = false;
        if (!route.ready || route.busy) { this.stop(route.detail ?? 'Check the configured agent connection, then explain again.', 'unavailable'); return; }
        run.state = { ...run.state, detail: 'Reading model components…' }; this.publish(run);
      } catch {
        if (this.active === run) this.stop('The configured agent could not connect. Check its sign-in and try again.', 'unavailable');
        return;
      } finally { clearTimeout(timeout); }
    }
    if (this.current(run)) await this.explain(run);
    else if (this.active === run) this.refresh();
  }
  private refuse(request: LocalBrowserModelExplanationRequest): void {
    const refusal = this.options.refusal?.(request);
    if (refusal) this.options.publish(request.viewId, request.target, { requestId: request.requestId, revision: refusal.ref.revision,
      status: 'unavailable', completed: 0, total: request.nodeIds.length, detail: refusal.detail });
  }
  private charge(batch: Batch, value: unknown): void {
    batch.bytes += Buffer.byteLength(JSON.stringify(value));
    if (batch.bytes > MAX_BYTES) throw new GraphToolError('SOURCE_LIMIT', 'The explanation context budget is exhausted.');
  }
  private async source(batch: Batch, file: string) {
    this.check(batch); const source = await explanationAbortable(this.options.read(batch.run.request.target, file), batch.run.controller.signal); this.check(batch);
    if (source.file !== file || hash(source.source) !== source.digest) throw new GraphToolError('STALE_TARGET', 'The source evidence changed.');
    return source;
  }
  private async excerpt(batch: Batch, file: string, startLine: number, count: number) {
    const source = await this.source(batch, file), lines = source.source.split('\n');
    if (startLine < 1 || startLine > lines.length) throw new GraphToolError('INVALID_QUERY', 'Choose a current source line.');
    const selected: string[] = []; let bytes = 0;
    for (const line of lines.slice(startLine - 1, startLine - 1 + count)) { bytes += Buffer.byteLength(line) + 1; if (bytes > 4096) break; selected.push(line); }
    if (!selected.length) throw new GraphToolError('SOURCE_LIMIT', 'The source line exceeds the preview limit.');
    const endLine = startLine + selected.length - 1;
    let citation = [...batch.evidence.values()].find(item => item.file === file && item.startLine === startLine && item.endLine === endLine && item.sourceDigest === source.digest);
    if (!citation) { citation = { id: `evidence-${batch.evidence.size + 1}`, file, sourceDigest: source.digest, startLine, endLine }; batch.evidence.set(citation.id, citation); }
    return { citation, source: selected.join('\n') };
  }
  private async documentation(batch: Batch, nodeId: string): Promise<NodeSourceDocumentation | undefined> {
    if (!this.options.documentation) return;
    try {
      const value = await explanationAbortable(this.options.documentation(batch.run.request, nodeId, batch.run.controller.signal), batch.run.controller.signal);
      this.check(batch); if (!value || !isNodeSourceDocumentation(value)) return;
      if (value.status === 'resolved') {
        const source = await this.source(batch, value.citation.file);
        if (source.digest !== value.citation.sourceDigest || value.citation.endLine > source.source.split('\n').length) return;
      }
      batch.run.documentation.set(nodeId, value); return value;
    } catch { this.check(batch); return; }
  }
  private async valid(batch: Batch, explanation: NodeExplanation): Promise<boolean> {
    if (!isNodeExplanation(explanation)) return false;
    const sources = new Map<string, Awaited<ReturnType<CanvasModelExplanation['source']>>>();
    for (const citation of [...explanation.citations, ...batch.evidence.values()]) {
      let source = sources.get(citation.file); if (!source) { source = await this.source(batch, citation.file); sources.set(citation.file, source); }
      if (source.digest !== citation.sourceDigest || citation.endLine > source.source.split('\n').length) return false;
    }
    return true;
  }
  private tools(batch: Batch, owner: string): HostToolset {
    const tools: HostToolset = new Map(), annotations = { readOnlyHint: true, destructiveHint: false, openWorldHint: false };
    const query = this.options.tools(owner).get('query_source_context');
    if (query) tools.set('query_source_context', { config: { ...query.config, inputSchema: { ...query.config.inputSchema,
      limit: z.number().int().min(1).max(8).default(8), maxEntities: z.number().int().min(1).max(16).default(16), depth: z.number().int().min(0).max(1).default(1) } },
      handler: args => graphToolResult(async () => {
        this.check(batch);
        if (++batch.reads > 4 || args.modelId !== batch.run.ref.modelId || args.revision !== batch.run.ref.revision || args.assumeValue !== undefined || args.captureId !== undefined || args.sourceId !== undefined || (args.limit ?? 8) > 8 || (args.maxEntities ?? 16) > 16 || (args.depth ?? 1) > 1) throw new GraphToolError('INVALID_QUERY', 'Use a small source-only query in the attached revision.');
        const response = await explanationAbortable(query.handler(args, { signal: batch.run.controller.signal }), batch.run.controller.signal); this.check(batch);
        if (response.isError) throw new GraphToolError('SOURCE_UNAVAILABLE', 'The source definition is unresolved.');
        const page = response.structuredContent as Record<string, unknown> | undefined;
        const citations: NodeExplanationCitation[] = [];
        for (const item of Array.isArray(page?.items) ? page.items : []) {
          const anchor = item?.anchor;
          if (!anchor || typeof item.excerpt !== 'string' || !Number.isSafeInteger(anchor.startLine) || !Number.isSafeInteger(anchor.endLine)) continue;
          const source = await this.source(batch, anchor.file);
          if (anchor.digest !== source.digest || anchor.startLine < 1 || anchor.endLine < anchor.startLine || anchor.endLine > source.source.split('\n').length) throw new GraphToolError('STALE_TARGET', 'Source evidence changed.');
          const citation = { id: `evidence-${batch.evidence.size + 1}`, file: anchor.file, sourceDigest: source.digest, startLine: anchor.startLine, endLine: anchor.endLine };
          batch.evidence.set(citation.id, citation); citations.push(citation);
        }
        const result = { ...page, citations }; this.charge(batch, result); for (const citation of citations) batch.exposed.add(citation.id); return result;
      }) });
    tools.set('read_model_source', { config: { description: 'Read a small window of graph-recorded source without executing it. Cite its returned evidence ID.', annotations,
      inputSchema: { file: z.string().min(1).max(512), startLine: z.number().int().min(1).optional(), maxLines: z.number().int().min(1).max(80).optional() } },
      handler: args => graphToolResult(async () => { this.check(batch); if (++batch.reads > 4) throw new GraphToolError('SOURCE_LIMIT', 'The source read limit is reached.');
        const result = await this.excerpt(batch, args.file, args.startLine ?? 1, args.maxLines ?? 60); this.charge(batch, result); batch.exposed.add(result.citation.id); return result; }) });
    tools.set('submit_model_explanations', { config: { description: 'Submit source-grounded presentation annotations for the requested node IDs. Does not change model facts, topology, files or permissions.', annotations,
      inputSchema: batchSubmission },
      handler: args => graphToolResult(async () => {
        batch.submissionAttempted = true; this.check(batch); if (batch.submitted) throw new GraphToolError('ALREADY_SUBMITTED', 'This batch was already submitted.');
        const parsed = z.object(batchSubmission).strict().parse(args);
        const results = new Map<string, NodeExplanation>();
        for (const entry of parsed.nodes) {
          const { nodeId, citationIds, ...generated } = entry;
          if (!batch.nodeIds.has(nodeId) || results.has(nodeId) || new Set(citationIds).size !== citationIds.length || citationIds.some(id => !batch.exposed.has(id))) throw new GraphToolError('INVALID_EXPLANATION', 'Use distinct requested nodes and evidence IDs actually supplied.');
          const explanation = { ...generated, citations: citationIds.map(id => batch.evidence.get(id)!) };
          if (!await this.valid(batch, explanation)) throw new GraphToolError('INVALID_CITATION', 'Source evidence is stale or invalid.');
          results.set(nodeId, explanation);
        }
        this.check(batch); batch.results = results; batch.submitted = true; return { submitted: results.size };
      }) });
    return tools;
  }
  private node(run: Run, nodeId: string, explanation: NodeExplanation): void {
    this.options.publishNode(run.request.viewId, run.request.target, { requestId: run.request.requestId, nodeId, revision: run.ref.revision, status: 'ready', explanation: structuredClone(explanation), ...(run.documentation.has(nodeId) ? { documentation: structuredClone(run.documentation.get(nodeId)!) } : {}) });
    run.state.completed++; this.publish(run);
  }
  private async explain(run: Run): Promise<void> {
    const failures = new Set<FailureReason>();
    const guard = setInterval(() => { if (!this.current(run)) this.refresh(); }, 250);
    try {
      for (let offset = 0; offset < run.request.nodeIds.length && this.current(run); offset += BATCH_SIZE) {
        const requested = run.request.nodeIds.slice(offset, offset + BATCH_SIZE);
        if (offset > 0) {
          run.state = { ...run.state, detail: `Reading components ${offset + 1}–${offset + requested.length} of ${run.request.nodeIds.length}…` }; this.publish(run);
        }
        const batch: Batch = { run, nodeIds: new Set(), evidence: new Map(), reads: 0, bytes: 0, results: new Map(), exposed: new Set(), submitted: false, submissionAttempted: false };
        let runtime: ReturnType<CanvasModelExplanationOptions['runtime']> | undefined;
        const timer = setTimeout(() => { if (this.active === run) this.stop('Explanation timed out. Completed nodes remain available; retry to continue.', 'unavailable'); }, this.options.timeoutMs ?? 60000);
        let stage: 'source' | 'provider' | 'validation' = 'source';
        try {
          const nodes: { nodeId: string; facts: unknown; evidenceId: string; documentationEvidenceId?: string; documentationStatus?: string }[] = [];
          const sourceEvidence: { citation: NodeExplanationCitation; source: string }[] = [];
          let sourceDocumentation: { citation: NodeExplanationCitation; kind: string; qualifiedName: string; signature: string; docstring?: string; truncated: boolean }[] = [];
          for (const nodeId of requested) {
            this.check(batch);
            const cached = this.cache.get(this.key(run.request.target, run.ref, nodeId, run.route));
            if (cached && await this.valid(batch, cached)) { this.node(run, nodeId, cached); continue; }
            try {
              const prepared = await explanationAbortable(this.options.prepare(run.request, nodeId, run.controller.signal), run.controller.signal); this.check(batch);
              if (Buffer.byteLength(JSON.stringify(prepared.context)) > 4096) { failures.add('context_limit'); continue; }
              const start = Math.max(1, prepared.source.startLine - 8);
              let evidence = sourceEvidence.find(item => item.citation.file === prepared.source.file
                && item.citation.startLine <= prepared.source.startLine && item.citation.endLine >= prepared.source.startLine);
              if (!evidence) { evidence = await this.excerpt(batch, prepared.source.file, start, 60); sourceEvidence.push(evidence); }
              const evidenceId = evidence.citation.id;
              const documentation = await this.documentation(batch, nodeId);
              let documentationEvidenceId: string | undefined;
              if (documentation?.status === 'resolved') {
                const anchor = documentation.citation;
                let authored = sourceDocumentation.find(item => item.citation.file === anchor.file && item.citation.sourceDigest === anchor.sourceDigest
                  && item.citation.startLine === anchor.startLine && item.citation.endLine === anchor.endLine && item.qualifiedName === documentation.qualifiedName);
                if (!authored) {
                  const citation = { ...anchor, id: `evidence-${batch.evidence.size + 1}` }; batch.evidence.set(citation.id, citation);
                  authored = { citation, kind: documentation.kind, qualifiedName: documentation.qualifiedName, signature: documentation.signature.slice(0, 1024),
                    ...(documentation.docstring ? { docstring: documentation.docstring.slice(0, 768) } : {}),
                    truncated: documentation.truncated || documentation.signature.length > 1024 || (documentation.docstring?.length ?? 0) > 768 };
                  sourceDocumentation.push(authored);
                }
                documentationEvidenceId = authored.citation.id;
              }
              nodes.push({ nodeId, facts: prepared.context, evidenceId, documentationEvidenceId,
                ...(documentation ? { documentationStatus: documentation.status === 'resolved' ? 'resolved' : documentation.reason } : {}) }); batch.nodeIds.add(nodeId);
            } catch (error) { this.check(batch); failures.add(error instanceof GraphToolError && error.code === 'SOURCE_LIMIT' ? 'context_limit' : 'source_unavailable'); }
          }
          if (!nodes.length) continue;
          let context = { model: run.ref, nodes, sourceEvidence, sourceDocumentation };
          // Literal docs are optional context; never crowd out the current graph and source windows.
          if (Buffer.byteLength(JSON.stringify(context)) > MAX_BYTES) {
            sourceDocumentation = sourceDocumentation.map(item => ({ ...item, signature: item.signature.slice(0, 512), docstring: item.docstring?.slice(0, 256), truncated: true }));
            context = { ...context, sourceDocumentation };
          }
          if (Buffer.byteLength(JSON.stringify(context)) > MAX_BYTES) {
            sourceDocumentation = []; for (const node of nodes) { delete node.documentationEvidenceId; if (node.documentationStatus === 'resolved') node.documentationStatus = 'context_limit'; }
            context = { ...context, sourceDocumentation };
          }
          this.charge(batch, context);
          for (const item of [...sourceEvidence, ...sourceDocumentation]) batch.exposed.add(item.citation.id);
          runtime = this.options.runtime(owner => this.tools(batch, owner));
          stage = 'provider';
          await explanationAbortable(runtime.run({ provider: run.route.provider, settings: run.route.settings, intent: 'model_explanation', cwd: run.request.target.workspaceFolder,
            conversationId: hash(JSON.stringify([run.ref, run.request.requestId, offset])), context: JSON.stringify(context),
            messages: [{ id: `${run.request.requestId}-${offset}`, role: 'user', text: 'Explain these model components and the flow between them. Submit the structured node explanations.' }] }, () => {}, run.controller.signal), run.controller.signal);
          this.check(batch); stage = 'validation';
          if (!batch.submitted || batch.results.size < batch.nodeIds.size) failures.add(batch.submissionAttempted && !batch.submitted ? 'invalid_submission' : 'no_submission');
          for (const [nodeId, explanation] of batch.results) {
            if (!await this.valid(batch, explanation)) { failures.add('source_changed'); continue; }
            const key = this.key(run.request.target, run.ref, nodeId, run.route); this.cache.delete(key); this.cache.set(key, structuredClone(explanation));
            while (this.cache.size > 256) this.cache.delete(this.cache.keys().next().value!);
            this.node(run, nodeId, explanation);
          }
        } catch (error) {
          if (!this.current(run)) break;
          failures.add(error instanceof GraphToolError && error.code === 'SOURCE_LIMIT' ? 'context_limit' : stage === 'provider' ? 'provider_failed' : stage === 'validation' ? 'source_changed' : 'source_unavailable');
        }
        finally { clearTimeout(timer); runtime?.clear(); }
      }
      if (this.current(run)) {
        const partial = run.state.completed < run.state.total;
        run.state = { ...run.state, status: partial ? 'unavailable' : 'ready', ...(partial ? { detail: `${run.state.completed} of ${run.state.total} nodes explained. ${[...failures].slice(0, 2).map(reason => FAILURE_DETAILS[reason]).join(' ') || FAILURE_DETAILS.no_submission} Retry to continue.` } : { detail: undefined }) };
        this.active = undefined; this.publish(run); this.options.changed();
      }
    } finally { clearInterval(guard); if (this.active === run && !this.current(run)) this.stop('The model or agent changed. Completed nodes remain available.', 'unavailable'); }
  }
}
