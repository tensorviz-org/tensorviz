import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { GraphToolError, McpGraphService, graphToolResult, modelRefSchema, type ModelRef } from '@tensorviz/graph-host/mcpGraph';
import { AGENT_SOURCE_LIMIT, sourceEditNodeIds, type AgentCodeEdit, type LocalBrowserSourceEdits, type SourceProposalOutcome, type SourceProposalReview } from './localBrowserSourceEdits';
import { ACTIVATION_NAMES, isConstructorArgumentPath, isLocalBrowserSourceEdit, type LocalBrowserSourceEdit, type LocalBrowserTargetIdentity, type SourceEditValue } from '@tensorviz/contracts/protocol';
import { captureQuerySchema } from '@tensorviz/graph-host/captureInspection';
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
const agentCodeSchema = z.object({ kind: z.literal('replace_model_source'), beforeDigest: z.string().regex(/^[a-f0-9]{64}$/),
  source: z.string().min(1).max(AGENT_SOURCE_LIMIT).refine(source => Buffer.byteLength(source, 'utf8') <= AGENT_SOURCE_LIMIT) }).strict();
interface Proposal {
  proposalId: string; requestId: string; owner: string; ref: ModelRef; target: LocalBrowserTargetIdentity; edit: DeterministicEdit | AgentCodeEdit;
  status: 'preparing' | 'review' | SourceProposalOutcome['status']; detail: string;
  createdAt: number; finishedAt?: number; review?: SourceProposalReview; controller: AbortController;
  candidateRequests: Map<string, string>;
  completion?: Promise<void>;
  decided?: boolean;
}
type ProposalSource = Pick<LocalBrowserSourceEdits, 'propose' | 'runCandidate' | 'cancelCandidate'
  | 'getCandidate' | 'listCandidates' | 'inspectCandidate' | 'forgetCandidates' | 'inspectAppliedSource'>
  & Partial<Pick<LocalBrowserSourceEdits, 'readModelSource' | 'proposeCode' | 'checkCode'>>;

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
  async readSource(graph: McpGraphService, ref: ModelRef, owner?: string) {
    const { model } = graph.require(ref);
    if (!this.source.readModelSource) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Model source reading is unavailable.');
    const source = await this.source.readModelSource(model.target);
    const revision = this.revision;
    return { ...source, ...(revision && !revision.signal.aborted && revision.item.owner === owner
      && revision.item.ref.modelId === ref.modelId && revision.item.ref.revision === ref.revision
      && !revision.item.decided && revision.item.finishedAt === undefined && revision.item.edit.kind === 'replace_model_source'
      ? { pendingDraft: { source: revision.item.edit.source, baseDigest: revision.item.edit.beforeDigest } } : {}) };
  }
  /** Called only by a validated human follow-up, never exposed as an agent tool. */
  beginRevision(requestId: string, signal: AbortSignal) {
    const item = [...this.proposals.values()].find(item => item.requestId === requestId && item.finishedAt === undefined);
    if (!item || item.decided || item.status !== 'review' || item.controller.signal.aborted || signal.aborted || item.edit.kind !== 'replace_model_source') {
      throw new GraphToolError('UNKNOWN_PROPOSAL', 'This draft is no longer available. Review the current canvas before sending again.');
    }
    const revision = this.revision = { item, signal };
    signal.addEventListener('abort', () => { if (this.revision === revision) this.revision = undefined; }, { once: true });
    return { draftId: requestId, source: item.edit.source, baseDigest: item.edit.beforeDigest };
  }
  async startCode(owner: string, graph: McpGraphService, ref: ModelRef, edit: AgentCodeEdit): Promise<Record<string, unknown>> {
    const { model } = graph.require(ref);
    if (!this.source.proposeCode) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Code proposals are unavailable.');
    const parsed = agentCodeSchema.parse(edit), revision = this.revision;
    if (revision && revision.item.owner === owner) {
      const { item, signal } = revision;
      const valid = () => this.revision === revision && !signal.aborted && !item.controller.signal.aborted
        && this.proposals.get(item.proposalId) === item && !item.decided && item.finishedAt === undefined && item.status === 'review';
      if (!valid() || !this.source.checkCode || item.target.sourceDigest !== model.target.sourceDigest
        || item.ref.modelId !== ref.modelId || item.ref.revision !== ref.revision) {
        throw new GraphToolError('STALE_TARGET', 'The draft changed or closed. Send another message from the current canvas.');
      }
      await this.source.checkCode(model.target, parsed, signal);
      if (!valid()) throw new GraphToolError('CANCELLED', 'The draft was accepted, discarded or cancelled while revising.');
      graph.require(ref);
      item.controller.abort();
      await item.completion; // Wait for native review cleanup and the shared writer reservation.
      this.revision = undefined;
      if (signal.aborted || item.status === 'applied') throw new GraphToolError('CANCELLED', 'The draft revision is no longer current.');
      graph.require(ref);
    }
    return this.create(owner, ref, model.target, parsed);
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
  inspectCandidate(owner: string, proposalId: string, runId: string, query?: CaptureInspectionQuery): Record<string, unknown> {
    this.require(owner, proposalId);
    return { proposalId, candidate: this.source.getCandidate(`mcp-${proposalId}`, runId),
      ...(query ? { page: this.source.inspectCandidate(`mcp-${proposalId}`, runId, query) } : {}) };
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
      edit: item.edit.kind === 'replace_model_source' ? { kind: item.edit.kind, beforeDigest: item.edit.beforeDigest } : item.edit,
      status: item.status, detail: item.detail, review: item.review ?? null,
      createdAt: new Date(item.createdAt).toISOString(), finishedAt: item.finishedAt === undefined ? null : new Date(item.finishedAt).toISOString() };
  }
  private async execute(item: Proposal): Promise<void> {
    await Promise.resolve();
    try {
      const onReview = (review: SourceProposalReview) => {
          if (item.controller.signal.aborted) return;
          item.review = review; item.status = 'review';
          item.detail = item.edit.kind === 'replace_model_source'
            ? 'The code diff and draft graph are ready. Accept or Discard on the canvas, or send another message to revise the pending draft. Nothing has been applied.'
            : 'Review the complete Python diff in VS Code. Choose Apply source change or Discard proposal there.';
        };
      const outcome = item.edit.kind === 'replace_model_source'
        ? await this.source.proposeCode!({ requestId: item.requestId, target: item.target, edit: item.edit }, item.controller.signal, onReview,
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
    description: 'Read the complete current workspace model file and its digest. During a human-authorized draft revision, pendingDraft contains the complete unaccepted source to continue editing; retain its earlier changes unless asked otherwise and use the live digest as beforeDigest. Use before propose_code_edit, including inherited layers without typed capabilities. No model execution.',
    inputSchema: modelRefSchema, annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, args => graphToolResult(() => proposals.readSource(graph, args, owner)));
  server.registerTool('propose_code_edit', {
    description: 'Propose normal Python code changes to the current workspace model file through native VS Code Apply/Discard review. Use for structural changes, new layers, imports, forward logic or subclass overrides unsupported by propose_edit. Read the full model source first; send the complete updated source and its original digest. Preserve unrelated code. Inherited library layers should be overridden in the workspace subclass, never edited in site-packages. Returns a proposalId; use get_proposal to confirm review opened, then finish your reply. Does not apply, save or execute code. Maximum 256 KiB.',
    inputSchema: { ...modelRefSchema, beforeDigest: agentCodeSchema.shape.beforeDigest, source: agentCodeSchema.shape.source },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  }, args => graphToolResult(() => proposals.startCode(owner, graph, args,
    { kind: 'replace_model_source', beforeDigest: args.beforeDigest, source: args.source })));
}

export function registerMcpProposalTools(server: McpServer, graph: McpGraphService, proposals: McpProposals, owner: string): void {
  server.registerTool('propose_edit', {
    description: 'Propose one deterministic source edit. Static graphs support parameter/add/remove/group. Captured operations support set_literal, replace_activation, reconnect_argument, insert_unary_op or delete_unary_op; captured module invocations support replace_activation, set_constructor_literal or set_configuration_literal through their verified constructor. These require the exact sourceEditCapability returned by inspect_nodes. Copy its id, target, expected value and scope; set_literal, reconnect_argument, insert_unary_op and delete_unary_op require the exact argumentPath; set_constructor_literal and set_configuration_literal require sourceArgumentPath. Numeric edits take a typed numeric value. reconnect_argument takes a ValueRef from the capability choices, never a source name or code. The argumentPath selects one operand use even when both use the same tensor; only listed values are proven available in source order. Availability does not prove tensor shape compatibility. insert_unary_op wraps exactly one listed named operand in the chosen activation (relu, gelu, silu, sigmoid or tanh), using nonmutating defaults. delete_unary_op bypasses the proven functional call to its original input: copy expected from the capability output and value from capability.input.value. Both use source_expression scope; do not submit names, options, source text or import requests. The capability declares any required import for native review. Configuration edits change the verified definition, preserve constructor references, and require configuration_definition scope affecting all references, including unobserved uses. Other recipes require source_expression scope. Constructor paths address authored arguments, not forward ports; choose one of the returned capabilities. Shape and checkpoint compatibility may change. replace_activation takes one of the capability’s compatibleActivations (relu, gelu, silu, sigmoid or tanh); preserve its authored options. Explicit inplace options restrict replacement to relu/silu, including input/alias mutation when true. Explicit GELU approximation permits only a no-op until the option is changed in source. Expression scope affects every execution, including unobserved uses. Returns immediately; review the full Python diff and explicitly Apply/Discard in VS Code. No automatic apply, free-form code, shell or arbitrary path. Preparation checks source syntax only; it does not run the model.',
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
    description: 'Inspect a confirmed native Apply without saving files or running the model. Distinguishes exact reviewed unsaved/saved source, restored original bytes and other edits. Checks the latest finished post-Apply capture when available; save and explicitly validate separately to create one. Source state, current setup freshness, recorded semantic effects, input/state equality and partial dependency scope remain independent. A restored file does not prove a native Undo action. Rechecks tracked files and available saved execution setup; never rewrites historical candidate results. Results describe the instant checkedAt, not continuing certification. Evidence expires with its owner or retention limits.',
    inputSchema: { proposalId }, annotations: read,
  }, (args, extra) => graphToolResult(() => proposals.inspectAppliedSource(owner, args.proposalId, extra.signal)));
  server.registerTool('run_proposal_candidate', {
    description: 'Explicitly execute the exact reviewed source in a private copied workspace using this proposal’s retained Python setup. Requires an open native source diff and a resolved replay recipe; executes trusted workspace Python, which may have external side effects. Returns immediately with candidate.runId. Supply a new UUID requestId for an intentional run; retry the same requestId to read its existing run without reexecuting, even after review closes. Concurrent requests share a pending run. Maximum 128 request IDs per proposal. Never applies source or replaces the live graph. Runtime success does not prove complete capture, input/state equality or semantic postconditions. Poll inspect_proposal_candidate for completion; explicitly Apply/Discard in VS Code.',
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
    description: 'List retained candidate summaries for your proposal, including native Run actions. Records survive Apply/Discard and source changes as historical evidence, not continuing live-source certification. Up to 32 runs per host, 30 minutes after completion, 16 MiB/capture and 32 MiB total; payload eviction preserves summary reasons. Disconnect/revocation removes your records. Never executes Python.',
    inputSchema: { proposalId }, annotations: read,
  }, args => graphToolResult(() => proposals.listCandidates(owner, args.proposalId)));
  server.registerTool('inspect_proposal_candidate', {
    description: 'Read your candidate’s bounded summary, or page its immutable canonical capture with query. Copy captureId/sourceId from candidate.capture and start with path=[]; use nextOffset and child paths exactly as inspect_capture. Source changes do not prevent historical inspection. Freshness is the last review observation, not certification of the current environment or edited buffer. An unavailable payload never causes reexecution. This tool cannot focus private candidate records in the live graph.',
    inputSchema: { proposalId, runId, query: captureQuerySchema.optional() }, annotations: read,
  }, args => graphToolResult(() => proposals.inspectCandidate(owner, args.proposalId, args.runId, args.query)));
}
