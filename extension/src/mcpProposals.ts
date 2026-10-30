import { randomUUID } from 'node:crypto';
import { candidateObservationQuerySchema, type CandidateObservationQuery } from './candidateHistory';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { GraphToolError, McpGraphService, graphToolResult, modelRefSchema, type ModelRef } from '@tensorviz/graph-host/mcpGraph';
import { agentSourceSetSchema, sourceEditNodeIds, type AgentSourceSetEdit, type LocalBrowserSourceEdits, type SourceProposalOutcome, type SourceProposalReview } from './localBrowserSourceEdits';
import { ACTIVATION_NAMES, isConstructorArgumentPath, isLocalBrowserSourceEdit, type LocalBrowserSourceEdit, type LocalBrowserTargetIdentity, type SourceEditValue } from '@tensorviz/contracts/protocol';
import { captureQuerySchema } from '@tensorviz/graph-host/captureInspection';
import { checkpointQuerySchema } from '@tensorviz/graph-host/checkpointEvidence';
import type { CaptureInspectionQuery } from '@tensorviz/contracts/protocol';

type DeterministicEdit = LocalBrowserSourceEdit;
const name = z.string().regex(/^[A-Za-z_][A-Za-z0-9_]{0,63}$/);
const nodeId = z.string().min(1).max(256);
const scalar = z.union([z.string().max(1024), z.number().finite(), z.boolean(), z.null()]);
const sourceNumber = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('int'), value: z.string().max(4096).regex(/^(?:0|-?[1-9][0-9]*)(?![\s\S])/) }).strict(),
  z.object({ kind: z.literal('float'), value: z.union([z.number().finite(), z.literal('-0')]) }).strict(),
]);
let literal: z.ZodType<SourceEditValue> = scalar;
for (let depth = 0; depth < 8; depth++) literal = z.union([scalar, z.array(literal).max(64)]);
export const mcpEditSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('insert_unary_op'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.literal('operation'), semanticId: z.string().min(1).max(128) }).strict(),
    argumentPath: z.tuple([z.literal('args'), z.union([z.literal(0), z.literal(1)])]),
    expected: z.object({ kind: z.literal('value'), id: z.string().min(1).max(128) }).strict(),
    value: z.enum(ACTIVATION_NAMES), scope: z.literal('source_expression') }).strict(),
  z.object({ kind: z.literal('delete_unary_op'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.literal('operation'), semanticId: z.string().min(1).max(128) }).strict(),
    argumentPath: z.tuple([z.literal('args'), z.literal(0)]),
    expected: z.object({ kind: z.literal('value'), id: z.string().min(1).max(128) }).strict(),
    value: z.object({ kind: z.literal('value'), id: z.string().min(1).max(128) }).strict(),
    scope: z.literal('source_expression') }).strict(),
  z.object({ kind: z.literal('reconnect_argument'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.literal('operation'), semanticId: z.string().min(1).max(128) }).strict(),
    argumentPath: z.tuple([z.literal('args'), z.union([z.literal(0), z.literal(1)])]),
    expected: z.object({ kind: z.literal('value'), id: z.string().min(1).max(128) }).strict(),
    value: z.object({ kind: z.literal('value'), id: z.string().min(1).max(128) }).strict(),
    scope: z.literal('source_expression') }).strict(),
  z.object({ kind: z.literal('set_constructor_literal'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.literal('invocation'), semanticId: z.string().min(1).max(128) }).strict(),
    sourceArgumentPath: z.union([z.tuple([z.literal('args'), z.number().int().min(0).max(255)]),
      z.tuple([z.literal('kwargs'), z.string()])]).refine(isConstructorArgumentPath),
    expected: sourceNumber, value: sourceNumber, scope: z.literal('source_expression') }).strict(),
  z.object({ kind: z.literal('set_configuration_literal'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.literal('invocation'), semanticId: z.string().min(1).max(128) }).strict(),
    sourceArgumentPath: z.union([z.tuple([z.literal('args'), z.number().int().min(0).max(255)]),
      z.tuple([z.literal('kwargs'), z.string()])]).refine(isConstructorArgumentPath),
    expected: sourceNumber, value: sourceNumber, scope: z.literal('configuration_definition') }).strict(),
  z.object({ kind: z.literal('replace_activation'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.enum(['operation', 'invocation']), semanticId: z.string().min(1).max(128) }).strict(),
    expected: z.enum(ACTIVATION_NAMES), value: z.enum(ACTIVATION_NAMES), scope: z.literal('source_expression') }).strict(),
  z.object({ kind: z.literal('set_literal'), nodeId, capabilityId: z.string().min(1).max(128),
    target: z.object({ version: z.literal(1), captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
      kind: z.literal('operation'), semanticId: z.string().min(1).max(128) }).strict(),
    argumentPath: z.tuple([z.literal('args'), z.literal(1)]), expected: sourceNumber, value: sourceNumber,
    scope: z.literal('source_expression') }).strict(),
  z.object({ kind: z.literal('set_param'), nodeId, param: name, value: literal }).strict(),
  z.object({ kind: z.literal('add_layer'), afterNodeId: nodeId, nodeId: name, layerType: name,
    params: z.record(name, literal).refine(value => Object.keys(value).length <= 32, 'At most 32 parameters.') }).strict(),
  z.object({ kind: z.literal('remove_layer'), nodeId }).strict(),
  z.object({ kind: z.literal('group_layers'), nodeIds: z.array(nodeId).min(2).max(100), name }).strict(),
]).superRefine((edit, ctx) => {
  if (!isLocalBrowserSourceEdit(edit) || JSON.stringify(edit).length > 16_384) {
    ctx.addIssue({ code: 'custom', message: 'Use distinct nodes and bounded literal parameters (at most 16 KiB per edit).' });
  }
});
interface Proposal {
  proposalId: string; requestId: string; owner: string; ref: ModelRef; target: LocalBrowserTargetIdentity; edit: DeterministicEdit | AgentSourceSetEdit;
  status: 'preparing' | 'review' | SourceProposalOutcome['status']; detail: string;
  createdAt: number; finishedAt?: number; review?: SourceProposalReview; controller: AbortController;
  candidateRequests: Map<string, string>;
  completion?: Promise<void>;
  decided?: boolean;
}
type ProposalSource = Pick<LocalBrowserSourceEdits, 'propose' | 'runCandidate' | 'cancelCandidate'
  | 'getCandidate' | 'listCandidates' | 'inspectCandidate' | 'inspectCandidateCheckpoint' | 'inspectCandidateObservations' | 'forgetCandidates' | 'inspectAppliedSource'>
  & Partial<Pick<LocalBrowserSourceEdits, 'readModelSource' | 'querySourceContext' | 'proposeSourceSet' | 'checkSourceSet'>>;
const isCodeEdit = (edit: Proposal['edit']): edit is AgentSourceSetEdit => edit.kind === 'replace_source_set';
const pendingSources = (item: Proposal) => item.edit.kind === 'replace_source_set' ? item.edit.sources : [];

/** An external client can propose and observe. Only native VS Code review can apply. */
export class McpProposals {
  private readonly proposals = new Map<string, Proposal>();
  private revision?: { item: Proposal; signal: AbortSignal };
  constructor(private readonly source: ProposalSource) {}
  endRevision(signal: AbortSignal): void { if (this.revision?.signal === signal) this.revision = undefined; }

  start(owner: string, graph: McpGraphService, ref: ModelRef, edit: DeterministicEdit): Record<string, unknown> {
    const { model, nodes } = graph.require(ref);
    const parsed = mcpEditSchema.parse(edit);
    graph.select(nodes, sourceEditNodeIds(parsed));
    return this.create(owner, ref, model.target, parsed);
  }
  async readSource(graph: McpGraphService, ref: ModelRef, owner?: string, file?: string) {
    const { model } = graph.require(ref);
    if (!this.source.readModelSource) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Model source reading is unavailable.');
    const source = await this.source.readModelSource(model.target, file);
    const revision = this.revision;
    const pending = revision && pendingSources(revision.item).find(item => item.file === source.file);
    return { ...source, ...(pending && revision && !revision.signal.aborted && revision.item.owner === owner
      && revision.item.ref.modelId === ref.modelId && revision.item.ref.revision === ref.revision
      && !revision.item.decided && revision.item.finishedAt === undefined
      ? { pendingDraft: { source: pending.source, baseDigest: pending.beforeDigest } } : {}) };
  }
  async querySourceContext(graph: McpGraphService, ref: ModelRef, query: Parameters<LocalBrowserSourceEdits['querySourceContext']>[1]) {
    const { model } = graph.require(ref);
    if (!this.source.querySourceContext) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Source dependency queries are unavailable.');
    const result = await this.source.querySourceContext(model.target, query);
    graph.require(ref);
    return result;
  }
  /** Called only by a validated human follow-up, never exposed as an agent tool. */
  beginRevision(requestId: string, signal: AbortSignal) {
    const item = [...this.proposals.values()].find(item => item.requestId === requestId && item.finishedAt === undefined);
    if (!item || item.decided || item.status !== 'review' || item.controller.signal.aborted || signal.aborted || !isCodeEdit(item.edit)) {
      throw new GraphToolError('UNKNOWN_PROPOSAL', 'This draft is no longer available. Review the current canvas before sending again.');
    }
    const revision = this.revision = { item, signal };
    signal.addEventListener('abort', () => { if (this.revision === revision) this.revision = undefined; }, { once: true });
    let remaining = 32 * 1024;
    const files = item.edit.sources.map(({ file, beforeDigest, source }) => {
      const bytes = Buffer.byteLength(JSON.stringify(source));
      const sourceTruncated = bytes > remaining;
      if (!sourceTruncated) remaining -= bytes;
      return { file, baseDigest: beforeDigest, sourceTruncated, ...(!sourceTruncated ? { source } : {}) };
    });
    return { draftId: requestId, files, capture: item.edit.capture };
  }
  async startSourceSet(owner: string, graph: McpGraphService, ref: ModelRef, edit: AgentSourceSetEdit,
    revertFiles: string[] = []): Promise<Record<string, unknown>> {
    const { model } = graph.require(ref);
    if (!this.source.proposeSourceSet) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Source-set proposals are unavailable.');
    const parsed = agentSourceSetSchema.parse(edit);
    if (parsed.capture === undefined && this.revision?.item.owner === owner && this.revision.item.edit.kind === 'replace_source_set') {
      parsed.capture = this.revision.item.edit.capture;
    }
    await this.replaceRevision(owner, graph, ref, model.target, parsed, revertFiles);
    graph.require(ref);
    return this.create(owner, ref, model.target, parsed);
  }
  private async replaceRevision(owner: string, graph: McpGraphService, ref: ModelRef, target: LocalBrowserTargetIdentity,
    parsed: AgentSourceSetEdit, revertFiles: string[] = []): Promise<void> {
    const revision = this.revision;
    if (revertFiles.length && (!revision || revision.item.owner !== owner)) throw new GraphToolError('UNKNOWN_PROPOSAL', 'Revert files only within your human-selected pending draft.');
    if (revision && revision.item.owner === owner) {
      const { item, signal } = revision;
      const valid = () => this.revision === revision && !signal.aborted && !item.controller.signal.aborted
        && this.proposals.get(item.proposalId) === item && !item.decided && item.finishedAt === undefined && item.status === 'review';
      if (!valid() || item.target.sourceDigest !== target.sourceDigest
        || item.ref.modelId !== ref.modelId || item.ref.revision !== ref.revision) {
        throw new GraphToolError('STALE_TARGET', 'The draft changed or closed. Send another message from the current canvas.');
      }
      const old = pendingSources(item), next = parsed.sources.map(item => item.file);
      if (new Set(revertFiles).size !== revertFiles.length || revertFiles.some(file => !old.some(item => item.file === file) || next.includes(file))
        || old.some(item => !next.includes(item.file) && !revertFiles.includes(item.file))) {
        throw new GraphToolError('INCOMPLETE_DRAFT', 'Preserve every pending file, or explicitly name its removal in revertFiles.');
      }
      if (!this.source.checkSourceSet) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Source-set checking is unavailable.');
      await this.source.checkSourceSet(target, parsed, signal);
      // Removing a pending edit does not permit a concurrently changed base file.
      for (const file of revertFiles) {
        if (!this.source.readModelSource) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Source reading is unavailable.');
        const live = await this.source.readModelSource(target, file);
        if (live.digest !== old.find(item => item.file === file)!.beforeDigest) throw new GraphToolError('STALE_TARGET', 'A reverted draft file changed on disk or in the editor.');
      }
      if (!valid()) throw new GraphToolError('CANCELLED', 'The draft was accepted, discarded or cancelled while revising.');
      graph.require(ref);
      item.controller.abort();
      await item.completion; // Wait for native review cleanup and the shared writer reservation.
      this.revision = undefined;
      if (signal.aborted || item.status === 'applied') throw new GraphToolError('CANCELLED', 'The draft revision is no longer current.');
      graph.require(ref);
    }
  }
  private create(owner: string, ref: ModelRef, target: LocalBrowserTargetIdentity, edit: Proposal['edit']): Record<string, unknown> {
    if ([...this.proposals.values()].some(item => item.finishedAt === undefined)) throw new GraphToolError('BUSY', 'Finish or discard the open native source review before proposing another edit.');
    for (const [id, item] of this.proposals) {
      if (item.finishedAt !== undefined && (Date.now() - item.finishedAt > 30 * 60_000 || this.proposals.size >= 32)) this.remove(id, item);
    }
    const proposalId = randomUUID();
    const proposal: Proposal = { proposalId, requestId: `mcp-${proposalId}`, owner, ref: { modelId: ref.modelId, revision: ref.revision }, target: { ...target }, edit,
      status: 'preparing', detail: 'Preparing a Python proposal for native VS Code review.',
      createdAt: Date.now(), controller: new AbortController(), candidateRequests: new Map() };
    this.proposals.set(proposal.proposalId, proposal);
    proposal.completion = this.execute(proposal);
    return this.snapshot(proposal);
  }
  get(owner: string, proposalId: string): Record<string, unknown> {
    return this.snapshot(this.require(owner, proposalId));
  }
  runCandidate(owner: string, proposalId: string, requestId: string): Record<string, unknown> {
    const item = this.require(owner, proposalId), sourceId = `mcp-${proposalId}`;
    const prior = item.candidateRequests.get(requestId);
    // A lost response can be retried after review closes. Eviction must never
    // turn the retry into another Python execution.
    if (prior) return { proposalId, candidate: this.source.getCandidate(sourceId, prior) };
    if (item.status !== 'review' || item.controller.signal.aborted) {
      throw new GraphToolError('CANDIDATE_UNAVAILABLE', 'Run a candidate only while its native source review is open.');
    }
    if (item.candidateRequests.size >= 128) throw new GraphToolError('REQUEST_LIMIT', 'This proposal has reached its candidate request limit.');
    const candidate = this.source.runCandidate(sourceId);
    item.candidateRequests.set(requestId, candidate.runId);
    return { proposalId, candidate };
  }
  cancelCandidate(owner: string, proposalId: string, runId: string): Record<string, unknown> {
    this.require(owner, proposalId);
    return { proposalId, candidate: this.source.cancelCandidate(`mcp-${proposalId}`, runId) };
  }
  listCandidates(owner: string, proposalId: string): Record<string, unknown> {
    this.require(owner, proposalId);
    return { proposalId, candidates: this.source.listCandidates(`mcp-${proposalId}`) };
  }
  inspectCandidate(owner: string, proposalId: string, runId: string, query?: CaptureInspectionQuery,
    checkpoint?: { offset?: number; limit?: number }, observations?: CandidateObservationQuery): Record<string, unknown> {
    this.require(owner, proposalId);
    if ([query, checkpoint, observations].filter(Boolean).length > 1) throw new GraphToolError('INVALID_QUERY', 'Inspect one capture, checkpoint or observation page per request.');
    return { proposalId, candidate: this.source.getCandidate(`mcp-${proposalId}`, runId),
      ...(query ? { page: this.source.inspectCandidate(`mcp-${proposalId}`, runId, query) } : {}),
      ...(checkpoint ? { checkpoint: this.source.inspectCandidateCheckpoint(`mcp-${proposalId}`, runId, checkpoint) } : {}),
      ...(observations ? { observations: this.source.inspectCandidateObservations(`mcp-${proposalId}`, runId, observations) } : {}) };
  }
  async inspectAppliedSource(owner: string, proposalId: string, signal?: AbortSignal): Promise<Record<string, unknown>> {
    if (signal?.aborted) throw new GraphToolError('CANCELLED', 'Applied-source inspection was cancelled.');
    const item = this.require(owner, proposalId);
    if (item.status !== 'applied') throw new GraphToolError('APPLIED_SOURCE_UNAVAILABLE', 'Apply the reviewed source in VS Code before checking it.');
    const applied = await this.source.inspectAppliedSource(`mcp-${proposalId}`, signal);
    if (signal?.aborted) throw new GraphToolError('CANCELLED', 'Applied-source inspection was cancelled.');
    if (this.require(owner, proposalId) !== item || item.controller.signal.aborted) {
      throw new GraphToolError('UNKNOWN_PROPOSAL', 'This proposal is unavailable in the current MCP session.');
    }
    return { proposalId, applied };
  }
  private require(owner: string, proposalId: string): Proposal {
    const item = this.proposals.get(proposalId);
    if (!item || item.owner !== owner || (item.finishedAt !== undefined && Date.now() - item.finishedAt > 30 * 60_000)) {
      if (item?.owner === owner) this.remove(proposalId, item);
      throw new GraphToolError('UNKNOWN_PROPOSAL', 'This proposal is unavailable in the current MCP session.');
    }
    return item;
  }
  /** Match a live host-owned request exactly; renderer-supplied prefixes confer no ownership. */
  ownsRequest(requestId: string | undefined): boolean {
    return [...this.proposals.values()].some(item => item.finishedAt === undefined && item.requestId === requestId);
  }
  cancelOwner(owner: string): void {
    for (const [id, item] of this.proposals) if (item.owner === owner) this.remove(id, item);
  }
  clear(): void {
    for (const [id, item] of this.proposals) this.remove(id, item);
  }
  private remove(id: string, item: Proposal): void {
    if (this.revision?.item === item) this.revision = undefined;
    item.controller.abort();
    this.source.forgetCandidates(`mcp-${id}`);
    this.proposals.delete(id);
  }
  private snapshot(item: Proposal): Record<string, unknown> {
    return { proposalId: item.proposalId, ...item.ref, target: item.target,
      edit: item.edit.kind === 'replace_source_set' ? { kind: item.edit.kind, files: item.edit.sources.map(({ file, beforeDigest }) => ({ file, beforeDigest })), capture: item.edit.capture } : item.edit,
      status: item.status, detail: item.detail, review: item.review ?? null,
      createdAt: new Date(item.createdAt).toISOString(), finishedAt: item.finishedAt === undefined ? null : new Date(item.finishedAt).toISOString() };
  }
  private async execute(item: Proposal): Promise<void> {
    await Promise.resolve();
    try {
      const onReview = (review: SourceProposalReview) => {
          if (item.controller.signal.aborted) return;
          item.review = review; item.status = 'review';
          item.detail = isCodeEdit(item.edit)
            ? 'The code diff and draft graph are ready. Accept or Discard on the canvas, or send another message to revise the pending draft. Nothing has been applied.'
            : 'Review the complete Python diff in VS Code. Choose Apply source change or Discard proposal there.';
        };
      const outcome = item.edit.kind === 'replace_source_set'
        ? await this.source.proposeSourceSet!({ requestId: item.requestId, target: item.target, edit: item.edit }, item.controller.signal, onReview,
          () => { item.decided = true; })
        : await this.source.propose({ requestId: item.requestId, target: item.target, edit: item.edit },
          item.controller.signal, onReview);
      // Once Apply has actually succeeded, later cancellation must not claim no write.
      item.status = outcome.status; item.detail = outcome.detail;
    } catch {
      item.status = 'refused'; item.detail = 'The proposal could not complete. Inspect the VS Code buffer before retrying.';
    } finally { item.finishedAt = Date.now(); }
  }
}

/** Only the canvas-owned agent gets this reviewed code fallback, never arbitrary paths or Apply. */
export function registerCanvasCodeTools(server: McpServer, graph: McpGraphService, proposals: McpProposals, owner: string): void {
  server.registerTool('read_model_source', {
    description: 'Read the complete current workspace model file and its digest. Optional file selects a workspace Python dependency recorded by this graph, using an exact relative path from source imports or graph source references. Unknown, changed or external dependencies are refused. A source read never applies an edit. During a human-authorized draft revision, pendingDraft contains this file’s complete unaccepted source; retain its earlier changes unless asked otherwise and use the live digest as beforeDigest. Use when complete current source and its digest are not already attached, including truncated or missing draft files. No model execution.',
    inputSchema: { ...modelRefSchema, file: z.string().min(1).max(1024).optional() }, annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, args => graphToolResult(() => proposals.readSource(graph, args, owner, args.file)));
  server.registerTool('propose_source_set', {
    description: 'Prepare one reviewed source proposal for one or more existing graph-recorded workspace Python files, including a single imported source or configuration file. Each sources entry selects its exact file. Use complete current attached source (target.file and target.sourceDigest), or attached pendingDraft.files source and baseDigest during a draft revision. Read missing or truncated changed files with read_model_source first. Provide each exact beforeDigest and complete replacement source. All files open in pinned native diffs. Human Apply makes one undoable editor edit, saves only the reviewed files and starts the normal model opening; save failure or intervening edits stop execution. Maximum 16 unique files, 256 KiB each and 1 MiB total. Optional capture pins a retained displayed capture so the human can Run this exact candidate with that capture’s retained setup. Apply instead opens the model with its current saved setup; explicit validation checkpoints or input overrides are not automatically written to that profile. Preparation never applies, saves or runs code. During a human-authorized draft revision include every still-pending file; use revertFiles to explicitly remove a pending change while retaining the others. Use Discard in Canvas to remove the entire draft. get_proposal confirms the complete file manifest and review state; then finish your reply.',
    inputSchema: { ...modelRefSchema, sources: agentSourceSetSchema.shape.sources,
      capture: agentSourceSetSchema.shape.capture, revertFiles: z.array(agentSourceSetSchema.shape.sources.element.shape.file).max(16).optional() },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  }, args => graphToolResult(() => proposals.startSourceSet(owner, graph, args,
    { kind: 'replace_source_set', sources: args.sources, capture: args.capture }, args.revertFiles)));
}

export function registerMcpProposalTools(server: McpServer, graph: McpGraphService, proposals: McpProposals, owner: string): void {
  server.registerTool('propose_edit', {
    description: 'Propose one deterministic source edit. Static graphs support parameter/add/remove/group. Captured operations support set_literal, replace_activation, reconnect_argument, insert_unary_op or delete_unary_op; captured module invocations support replace_activation, set_constructor_literal or set_configuration_literal through their verified constructor. These require the exact sourceEditCapability returned by inspect_nodes. Copy its id, target, expected value and scope; set_literal, reconnect_argument, insert_unary_op and delete_unary_op require the exact argumentPath; set_constructor_literal and set_configuration_literal require sourceArgumentPath. Numeric edits take a typed numeric value. reconnect_argument takes a ValueRef from the capability choices, never a source name or code. The argumentPath selects one operand use even when both use the same tensor; only listed values are proven available in source order. Availability does not prove tensor shape compatibility. insert_unary_op wraps exactly one listed named operand in the chosen activation (relu, gelu, silu, sigmoid or tanh), using nonmutating defaults. delete_unary_op bypasses the proven functional call to its original input: copy expected from the capability output and value from capability.input.value. Both use source_expression scope; do not submit names, options, source text or import requests. The capability declares any required import for native review. Configuration edits change the verified definition, preserve constructor references, and require configuration_definition scope affecting all references, including unobserved uses. Other recipes require source_expression scope. Constructor paths address authored arguments, not forward ports; choose one of the returned capabilities. Shape and checkpoint compatibility may change. replace_activation takes one of the capability’s compatibleActivations (relu, gelu, silu, sigmoid or tanh); preserve its authored options. Explicit inplace options restrict replacement to relu/silu, including input/alias mutation when true. Explicit GELU approximation permits only a no-op until the option is changed in source. Expression scope affects every execution, including unobserved uses. Returns immediately; review the full Python diff and explicitly Apply/Discard in VS Code. No automatic apply, free-form code, shell or arbitrary path. Preparation checks source syntax only; it does not run the model. Human Accept/Apply saves the accepted file and starts the model with its saved setup.',
    inputSchema: { ...modelRefSchema, edit: mcpEditSchema },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  }, (args, extra) => graph.toolResult('propose_edit', () => {
    if (extra.signal.aborted) throw new GraphToolError('CANCELLED', 'The request was cancelled.');
    return proposals.start(owner, graph, args, args.edit);
  }));
  server.registerTool('get_proposal', {
    description: 'Read your proposal preparation, native review metadata, and applied/cancelled/refused outcome. review.candidate separates the latest explicit candidate run’s runtime outcome, freshness, capture coverage, actual input/state equivalence, cleanup and unchecked semantic postconditions. Use list_proposal_candidates and inspect_proposal_candidate for retained records. This tool does not start execution. Results remain readable after an applied edit changes the graph. At most 32 completed proposals are retained per window for up to 30 minutes.',
    inputSchema: { proposalId: z.string().uuid() }, annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  }, args => graphToolResult(() => proposals.get(owner, args.proposalId)));
  const proposalId = z.string().uuid(), runId = z.string().uuid();
  const read = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
  server.registerTool('inspect_applied_proposal', {
    description: 'Inspect a confirmed native Apply without saving files or running the model. Distinguishes exact reviewed unsaved/saved source, restored original bytes and other edits. For captured source-set proposals, checks every receipt.files entry and returns sourceSetComparison for the exact applied set; generic changes have no typed recipe postconditions or task-correctness claim. Checks the latest finished post-Apply capture when available; the acceptance flow owns automatic saving and capture, and this read never starts either. Source state, current setup freshness, recorded semantic effects, input/state equality and partial dependency scope remain independent. A restored file does not prove a native Undo action. Rechecks tracked files and available saved execution setup; never rewrites historical candidate results. Results describe the instant checkedAt, not continuing certification. Evidence expires with its owner or retention limits.',
    inputSchema: { proposalId }, annotations: read,
  }, (args, extra) => graphToolResult(() => proposals.inspectAppliedSource(owner, args.proposalId, extra.signal)));
  server.registerTool('run_proposal_candidate', {
    description: 'Explicitly execute the exact reviewed source in a private copied workspace using this proposal’s retained Python setup. Requires an open native source diff and a resolved replay recipe; executes trusted workspace Python, which may have external side effects. Returns immediately with candidate.runId. Supply a new UUID requestId for an intentional run; retry the same requestId to read its existing run without reexecuting, even after review closes. Concurrent requests share a pending run. Maximum 128 request IDs per proposal. Never applies source or replaces the live graph. This retained setup may differ from the current saved setup used after Apply; do not transfer candidate outcomes to Apply without checking both setups. Runtime success does not prove complete capture, input/state equality or semantic postconditions. Poll inspect_proposal_candidate for completion; explicitly Apply/Discard in VS Code.',
    inputSchema: { proposalId, requestId: z.string().uuid() },
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: true },
  }, (args, extra) => graphToolResult(() => {
    if (extra.signal.aborted) throw new GraphToolError('CANCELLED', 'The request was cancelled.');
    return proposals.runCandidate(owner, args.proposalId, args.requestId);
  }));
  server.registerTool('cancel_proposal_candidate', {
    description: 'Request cancellation of your proposal’s exact candidate run. A completed/older run cannot cancel a newer run. Poll inspect_proposal_candidate until cleanup completes. Does not discard or apply the source proposal.',
    inputSchema: { proposalId, runId },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, args => graphToolResult(() => proposals.cancelCandidate(owner, args.proposalId, args.runId)));
  server.registerTool('list_proposal_candidates', {
    description: 'List retained candidate summaries for your proposal, including native Run actions. Records survive Apply/Discard and source changes as historical evidence, not continuing live-source certification. Up to 32 runs per host, 30 minutes after completion, 16 MiB per capture or observation pair and 32 MiB total; payload eviction preserves summary reasons. Disconnect/revocation removes your records. Never executes Python.',
    inputSchema: { proposalId }, annotations: read,
  }, args => graphToolResult(() => proposals.listCandidates(owner, args.proposalId)));
  server.registerTool('inspect_proposal_candidate', {
    description: 'Read your candidate’s bounded summary, or page its immutable canonical capture with query. Copy captureId/sourceId from candidate.capture and start with path=[]; use nextOffset and child paths exactly as inspect_capture. Alternatively supply checkpoint:{offset:0,limit:50} to inspect historical pre-load registered/stored metadata and the actual load outcome, including loads that failed before a forward capture. Supply observations:{section:"outputs"} for final-output metadata and numerical comparison with the exact retained baseline. Missing baseline cannot certify equivalence; no additional execution occurs. Follow nextOffset; use only one page kind per request. The checkpoint report is bound to the reviewed model source and copied checkpoint digest. Successful loading does not establish forward correctness. Source changes do not prevent historical inspection. Freshness is the last review observation, not certification of the current environment or edited buffer. An unavailable payload never causes reexecution. This tool cannot focus private candidate records in the live graph.',
    inputSchema: { proposalId, runId, query: captureQuerySchema.optional(), checkpoint: checkpointQuerySchema.optional(), observations: candidateObservationQuerySchema.optional() }, annotations: read,
  }, args => graphToolResult(() => proposals.inspectCandidate(owner, args.proposalId, args.runId, args.query, args.checkpoint, args.observations)));
}
