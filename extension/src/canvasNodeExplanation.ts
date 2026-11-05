import { createHash, randomUUID } from 'node:crypto';
import type { LocalBrowserTargetIdentity, LocalBrowserViewSelectionMessage, LocalBrowserNodeExplanationRequest } from '@tensorviz/contracts/protocol';
import { isNodeExplanation, type NodeExplanation, type NodeExplanationCitation, type NodeExplanationState } from '@tensorviz/contracts/nodeExplanation';
import { GraphToolError, graphToolResult, type ModelRef } from '@tensorviz/graph-host/mcpGraph';
import { z } from 'zod';
import type { CanvasAgentRoute, AgentRunInput, AgentRunEvent } from './canvasAgent';
import type { HostToolset } from './mcpHost';
import { CANVAS_AGENT_PROMPTS } from './canvasAgentPrompts';

export interface NodeExplanationSource { file: string; source: string; digest: string; }
export interface CanvasNodeExplanationOptions {
  route(): CanvasAgentRoute;
  reference(selection: LocalBrowserViewSelectionMessage): ModelRef;
  prepare(selection: LocalBrowserViewSelectionMessage, signal: AbortSignal): Promise<{ ref: ModelRef; context: unknown; source: { file: string; startLine: number } }>;
  current(selection: LocalBrowserViewSelectionMessage, revision?: string): boolean;
  read(target: LocalBrowserTargetIdentity, file: string): Promise<NodeExplanationSource>;
  tools(owner: string): HostToolset;
  runtime(tools: (owner: string) => HostToolset): { run(input: AgentRunInput, emit: (event: AgentRunEvent) => void, signal: AbortSignal): Promise<void>; clear(): void };
  publish(viewId: string, target: LocalBrowserTargetIdentity, state: NodeExplanationState): void;
  reveal(target: LocalBrowserTargetIdentity, citation: NodeExplanationCitation, current: () => boolean): Promise<void>;
  debounceMs?: number;
}
interface Work {
  selection: LocalBrowserViewSelectionMessage;
  controller: AbortController;
  state: NodeExplanationState;
  ref?: ModelRef;
  key?: string;
  evidence: Map<string, NodeExplanationCitation>;
  exposed: Set<string>;
  reads: number;
  bytes: number;
  result?: NodeExplanation;
}
const digest = (source: string) => createHash('sha256').update(source).digest('hex');
const string = (max: number) => z.string().trim().min(1).max(max).regex(/^[^\u0000-\u001f\u007f]+$/);
const submission = {
  title: string(80), purpose: string(320), inputs: z.array(string(160)).max(6), outputs: z.array(string(160)).max(4),
  steps: z.array(string(320)).max(6), uncertainties: z.array(string(240)).max(4), citationIds: z.array(string(128)).min(1).max(8),
};
const routeKey = (route: CanvasAgentRoute) => JSON.stringify([route.provider, route.resolvedModel, route.settings, route.ready, route.busy]);
const targetKey = (target: LocalBrowserTargetIdentity) => [target.workspaceFolder, target.file, target.cls, target.sourceDigest, target.documentVersion, target.backendRevision];
const identity = (selection: LocalBrowserViewSelectionMessage) => JSON.stringify([selection.viewId, targetKey(selection.target), selection.nodeIds, selection.edgeIds, selection.tooLarge]);

/** Source-only background work owns neither the user chat nor execution authority. */
export class CanvasNodeExplanation {
  private selected?: LocalBrowserViewSelectionMessage;
  private active?: Work;
  private lastRoute = '';
  private timer?: NodeJS.Timeout;
  private running: Promise<void> = Promise.resolve();
  private disposed = false;
  private readonly cache = new Map<string, NodeExplanation>();
  constructor(private readonly options: CanvasNodeExplanationOptions) {}

  select(selection: LocalBrowserViewSelectionMessage): void {
    if (this.disposed) return;
    const same = this.selected && identity(this.selected) === identity(selection);
    if (same && this.active && this.current(this.active) && this.lastRoute === routeKey(this.options.route())) return;
    this.cancel();
    this.selected = undefined;
    if (selection.tooLarge || selection.nodeIds.length !== 1 || selection.edgeIds.length || !this.options.current(selection)) return;
    this.selected = structuredClone(selection);
    this.schedule();
  }

  refresh(): void {
    if (!this.selected || this.disposed) return;
    if (!this.options.current(this.selected, this.active?.ref?.revision)) { this.invalidate(); return; }
    const next = routeKey(this.options.route());
    if (next !== this.lastRoute) { this.cancel(); this.schedule(); }
  }

  disconnect(viewId: string): void { if (this.selected?.viewId === viewId) this.invalidate(); }
  invalidate(): void { this.cancel(); this.selected = undefined; }
  dispose(): void { this.disposed = true; this.invalidate(); this.cache.clear(); }
  private cancel(): void { clearTimeout(this.timer); this.active?.controller.abort(); this.active = undefined; }
  private current(work: Work): boolean {
    return !this.disposed && this.active === work && !work.controller.signal.aborted
      && !!this.selected && identity(this.selected) === identity(work.selection)
      && this.lastRoute === routeKey(this.options.route()) && this.options.current(work.selection, work.ref?.revision);
  }
  private check(work: Work): void {
    if (!this.current(work)) throw new GraphToolError('STALE_TARGET', 'The selected node, source, graph or agent changed.');
  }
  private publish(work: Work): void {
    if (this.current(work)) this.options.publish(work.selection.viewId, work.selection.target, structuredClone(work.state));
  }
  private schedule(): void {
    const selection = this.selected!;
    const route = this.options.route(); this.lastRoute = routeKey(route);
    let ref: ModelRef;
    try { ref = this.options.reference(selection); } catch { this.invalidate(); return; }
    const work: Work = { selection, controller: new AbortController(), evidence: new Map(), exposed: new Set(), reads: 0, bytes: 0,
      ref, state: { requestId: randomUUID(), nodeId: selection.nodeIds[0], revision: ref.revision, status: 'queued' } };
    this.active = work;
    if (!route.ready || route.busy) {
      work.state.status = 'unavailable'; work.state.detail = route.busy ? 'Available when the agent finishes its current task.' : 'Connect an agent in chat for a source explanation.';
      this.publish(work); return;
    }
    this.publish(work);
    this.timer = setTimeout(() => {
      const previous = this.running;
      this.running = (async () => { await previous; if (this.current(work)) await this.explain(work, route); })().catch(() => {});
    }, this.options.debounceMs ?? 400);
  }

  private async source(work: Work, file: string): Promise<NodeExplanationSource> {
    this.check(work);
    const source = await this.options.read(work.selection.target, file);
    this.check(work);
    if (source.file !== file || digest(source.source) !== source.digest) throw new GraphToolError('STALE_TARGET', 'The source evidence changed.');
    return source;
  }
  private evidence(work: Work, source: NodeExplanationSource, startLine: number, endLine: number): NodeExplanationCitation {
    const existing = [...work.evidence.values()].find(item => item.file === source.file && item.sourceDigest === source.digest && item.startLine === startLine && item.endLine === endLine);
    if (existing) return existing;
    if (work.evidence.size >= 24) throw new GraphToolError('SOURCE_LIMIT', 'The explanation has enough source evidence; submit a bounded result.');
    const citation = { id: `evidence-${work.evidence.size + 1}`, file: source.file, sourceDigest: source.digest, startLine, endLine };
    work.evidence.set(citation.id, citation); return citation;
  }
  private charge(work: Work, value: unknown): void {
    work.bytes += Buffer.byteLength(JSON.stringify(value));
    if (work.bytes > 48 * 1024) throw new GraphToolError('SOURCE_LIMIT', 'The source explanation read budget is exhausted.');
  }
  private async excerpt(work: Work, file: string, startLine: number, maxLines = 100) {
    const source = await this.source(work, file), lines = source.source.split('\n');
    if (startLine > lines.length) throw new GraphToolError('INVALID_QUERY', 'Choose a line in the current source file.');
    const excerpt: string[] = []; let bytes = 0;
    for (let index = startLine - 1; index < Math.min(lines.length, startLine - 1 + maxLines); index++) {
      bytes += Buffer.byteLength(lines[index]) + 1;
      if (bytes > 12 * 1024) break;
      excerpt.push(lines[index]);
    }
    if (!excerpt.length) throw new GraphToolError('SOURCE_LIMIT', 'The selected source line exceeds the preview limit.');
    const citation = this.evidence(work, source, startLine, startLine + excerpt.length - 1);
    const result = { citation, source: excerpt.join('\n'), totalLines: lines.length, truncated: citation.endLine < lines.length };
    this.charge(work, result);
    work.exposed.add(citation.id);
    return result;
  }
  private tools(work: Work, owner: string): HostToolset {
    const tools: HostToolset = new Map();
    const read = { readOnlyHint: true, destructiveHint: false, openWorldHint: false };
    const query = this.options.tools(owner).get('query_source_context');
    if (query) tools.set('query_source_context', { config: { ...query.config,
      description: 'Follow bounded source definitions and references in this attached graph revision. Use its exact modelId/revision and a file plus symbol to resolve a helper beyond the callsite. This is source evidence, never execution. Cite only the returned citation IDs.',
      inputSchema: { ...query.config.inputSchema,
      limit: z.number().int().min(1).max(16).default(16), maxEntities: z.number().int().min(1).max(32).default(32), depth: z.number().int().min(0).max(2).default(2) } }, handler: (args, extra) => graphToolResult(async () => {
      this.check(work);
      if (++work.reads > 4 || args.modelId !== work.ref?.modelId || args.revision !== work.ref?.revision || args.assumeValue !== undefined || args.captureId !== undefined || args.sourceId !== undefined
        || (args.limit ?? 16) > 16 || (args.maxEntities ?? 32) > 32 || (args.depth ?? 2) > 2) {
        throw new GraphToolError('INVALID_QUERY', 'Use the attached model revision and a small source-only query (limit<=16, maxEntities<=32, depth<=2); no hypothetical edits.');
      }
      const result = await query.handler({ ...args, limit: args.limit ?? 16, maxEntities: args.maxEntities ?? 32, depth: args.depth ?? 2 }, { ...extra, signal: work.controller.signal });
      this.check(work);
      if (result.isError) throw new GraphToolError('SOURCE_UNAVAILABLE', 'The source query could not resolve this node; retain that uncertainty.');
      const page = result.structuredContent as Record<string, unknown> | undefined;
      const citations: NodeExplanationCitation[] = [];
      for (const item of Array.isArray(page?.items) ? page.items : []) {
        const anchor = item?.anchor;
        if (!anchor || typeof item.excerpt !== 'string' || !Number.isSafeInteger(anchor.startLine) || !Number.isSafeInteger(anchor.endLine)) continue;
        const source = await this.source(work, anchor.file);
        if (source.digest !== anchor.digest || anchor.startLine < 1 || anchor.endLine < anchor.startLine || anchor.endLine > source.source.split('\n').length) throw new GraphToolError('STALE_TARGET', 'Source query evidence is no longer current.');
        citations.push(this.evidence(work, source, anchor.startLine, anchor.endLine));
      }
      const value = { ...page, citations };
      this.charge(work, value);
      for (const citation of citations) work.exposed.add(citation.id);
      return value;
    }) });
    tools.set('read_model_source', { config: { description: 'Read a bounded line window from this graph-recorded workspace Python source. Supply file, optional startLine and maxLines<=120. Never executes source. Cite the returned citation ID.',
      inputSchema: { file: z.string().min(1).max(512), startLine: z.number().int().min(1).optional(), maxLines: z.number().int().min(1).max(120).optional() }, annotations: read },
      handler: args => graphToolResult(async () => { this.check(work); if (++work.reads > 4) throw new GraphToolError('SOURCE_LIMIT', 'The source read limit is reached.'); return this.excerpt(work, args.file, args.startLine ?? 1, args.maxLines ?? 100); }) });
    tools.set('submit_node_explanation', { config: { description: 'Submit the single source-grounded explanation for this selected node, citing only evidence IDs actually read. Records presentation only; cannot execute, change source, select another node or grant permissions.',
      inputSchema: submission, annotations: read }, handler: args => graphToolResult(async () => {
      this.check(work);
      if (work.result) throw new GraphToolError('ALREADY_SUBMITTED', 'The explanation is already submitted.');
      const parsed = z.object(submission).strict().parse(args);
      if (new Set(parsed.citationIds).size !== parsed.citationIds.length || parsed.citationIds.some(id => !work.exposed.has(id))) throw new GraphToolError('INVALID_CITATION', 'Cite only distinct source evidence IDs returned for this request.');
      const { citationIds, ...generated } = parsed;
      const result = { ...generated, citations: citationIds.map(id => work.evidence.get(id)!) };
      if (!isNodeExplanation(result) || !await this.validate(work, result)) throw new GraphToolError('INVALID_EXPLANATION', 'The explanation or its source evidence is no longer valid.');
      this.check(work); work.result = structuredClone(result); return { submitted: true };
    }) });
    return tools;
  }
  private async validate(work: Work, explanation: NodeExplanation): Promise<boolean> {
    for (const citation of explanation.citations) {
      const source = await this.source(work, citation.file);
      if (source.digest !== citation.sourceDigest || citation.endLine > source.source.split('\n').length) return false;
    }
    return true;
  }
  private async explain(work: Work, route: CanvasAgentRoute): Promise<void> {
    let runtime: ReturnType<CanvasNodeExplanationOptions['runtime']> | undefined;
    const timeout = setTimeout(() => {
      if (this.current(work)) {
        work.state = { ...work.state, status: 'unavailable', detail: 'The source explanation timed out. Retry when the agent is available.' }; this.publish(work);
      }
      work.controller.abort();
    }, 45000);
    const guard = setInterval(() => { if (!this.current(work)) work.controller.abort(); }, 250);
    try {
      this.check(work);
      const prepared = await this.options.prepare(work.selection, work.controller.signal); this.check(work);
      if (prepared.ref.modelId !== work.ref?.modelId || prepared.ref.revision !== work.ref.revision) throw new Error('The graph revision changed during preparation.');
      this.check(work);
      const key = work.key = JSON.stringify([work.selection.target.workspaceFolder, prepared.ref, work.selection.target.sourceDigest,
        work.state.nodeId, route.provider, route.resolvedModel, route.settings, CANVAS_AGENT_PROMPTS.node_explanation.version]);
      const cached = this.cache.get(key);
      if (cached && await this.validate(work, cached)) {
        this.cache.delete(key); this.cache.set(key, cached); work.state = { ...work.state, status: 'ready', explanation: cached }; this.publish(work); return;
      }
      work.state.status = 'loading'; this.publish(work);
      const evidence = await this.excerpt(work, prepared.source.file, Math.max(1, prepared.source.startLine - 20), 120);
      const context = { facts: prepared.context, sourceEvidence: evidence };
      if (Buffer.byteLength(JSON.stringify(context)) > 48 * 1024) throw new GraphToolError('SOURCE_LIMIT', 'This selection has more context than a focused explanation can read.');
      runtime = this.options.runtime(owner => this.tools(work, owner));
      await runtime.run({ provider: route.provider, settings: route.settings, intent: 'node_explanation', cwd: work.selection.target.workspaceFolder,
        conversationId: digest(key), context: JSON.stringify(context), messages: [{ id: randomUUID(), role: 'user', text: 'Explain this selected model node from the supplied source evidence. Submit the structured explanation.' }] }, () => {}, work.controller.signal);
      this.check(work);
      if (!work.result || !await this.validate(work, work.result)) throw new Error('No verified source explanation was returned.');
      this.cache.set(key, structuredClone(work.result)); while (this.cache.size > 64) this.cache.delete(this.cache.keys().next().value!);
      work.state = { ...work.state, status: 'ready', explanation: work.result }; this.publish(work);
    } catch {
      if (this.current(work)) { work.state = { ...work.state, status: 'unavailable', detail: 'A source explanation is unavailable. You can still inspect the node facts or retry.' }; this.publish(work); }
    } finally { clearTimeout(timeout); clearInterval(guard); runtime?.clear(); }
  }

  async request(message: LocalBrowserNodeExplanationRequest): Promise<void> {
    const work = this.active;
    if (!work || message.viewId !== work.selection.viewId || message.nodeId !== work.state.nodeId
      || message.requestId !== work.state.requestId || JSON.stringify(targetKey(message.target)) !== JSON.stringify(targetKey(work.selection.target))
      || !this.selected || identity(this.selected) !== identity(work.selection) || !this.options.current(work.selection, work.ref?.revision)
      || this.lastRoute !== routeKey(this.options.route())) return;
    if (message.action === 'retry') { if (work.key) this.cache.delete(work.key); this.cancel(); this.schedule(); return; }
    const explanation = work.state.explanation;
    if (work.state.status !== 'ready' || message.requestId !== work.state.requestId || !explanation) return;
    const citation = explanation.citations.find(item => item.id === message.citationId);
    if (citation && await this.validate(work, { ...explanation, citations: [citation] })) {
      this.check(work); await this.options.reveal(work.selection.target, citation, () => this.current(work));
    }
  }
}
