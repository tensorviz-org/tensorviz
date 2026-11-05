import { matchesHostFile } from './workspaceFiles';
import type { ExecutionObservation } from '@tensorviz/graph-host/executionObservation';
import * as path from 'node:path';
import * as fs from 'node:fs';
import * as vscode from 'vscode';
import { isDeepStrictEqual } from 'node:util';
import { z } from 'zod';
import { SourceWriter, CodeArtifact, sha256 } from './sourceChanges';
import { EditApplier, SourceEdit, toRange } from './apply';
import { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { LocalBrowserSync } from './localBrowserSync';
import {
  LocalBrowserSourceEdit, LocalBrowserSourceEditRequestMessage,
  LocalBrowserTargetIdentity, isLocalBrowserSourceEdit, isProjectDeclaration,
  LocalBrowserProjectDeclareMessage, candidatePostconditionsText,
} from '@tensorviz/contracts/protocol';
import { isNormalizedPythonPath } from './workspaceSources';
import { reviewSource, reviewSources } from './sourceReview';
import { SOURCE_SET_LIMITS } from '@tensorviz/graph-host/workspaceSourceCore';
import type { ProjectDeclarationContext } from './projectSync';
import { PROJECT_LIMITS } from './projectSources';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { CapturedEditError, isSourceEditCapability, prepareCapturedSourceEdit } from './capturedSourceEdit';
import { reviewResourceGuard, ReviewResourceError } from './sourceReviewResources';
import { CandidateReview, type CandidateReviewSnapshot } from './candidateReview';
import type { ResolvedExecutionPlan } from './executionPlan';
import type { TraceWorker } from './worker';
import { CandidateHistory, type CandidateRecord, type CandidateObservationQuery } from './candidateHistory';
import type { CaptureInspectionQuery } from '@tensorviz/contracts/protocol';
import { GraphToolError, object } from '@tensorviz/graph-host/mcpGraph';
import { AppliedSourceReceipts, type AppliedSourceReceipt } from './appliedSourceReceipts';
import { AppliedSourceInspector, type AppliedCapture } from './appliedSourceInspection';
import type { SourceDraftInput, SourceDraftHandle } from './sourceDraft';
import { sourceContextPage, sourceContextQuerySchema, type SourceContextQuery, type SourceContextResult } from '@tensorviz/graph-host/sourceContext';

const DECLARATION_TIMEOUT_MS = 10_000;
type ReviewDocuments = Map<string, { document: vscode.TextDocument; version: number; digest: string }>;

export type SourceProposalRequest = Pick<LocalBrowserSourceEditRequestMessage, 'requestId' | 'target' | 'edit'>;
export const AGENT_SOURCE_LIMIT = 256 * 1024;
export const agentSourceSetSchema = z.object({
  kind: z.literal('replace_source_set'),
  sources: z.array(z.object({
    file: z.string().min(1).max(1024).refine(isNormalizedPythonPath).refine(file => !/[\x00-\x1f\x7f]/.test(file)),
    beforeDigest: z.string().length(64).regex(/^[a-f0-9]+$/),
    source: z.string().max(AGENT_SOURCE_LIMIT).refine(source => Buffer.byteLength(source, 'utf8') <= AGENT_SOURCE_LIMIT
      && Buffer.from(source).toString('utf8') === source),
  }).strict()).min(1).max(16),
  capture: z.object({ captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128) }).strict().optional(),
}).strict().refine(edit => new Set(edit.sources.map(item => item.file)).size === edit.sources.length
  && edit.sources.reduce((sum, item) => sum + Buffer.byteLength(item.source, 'utf8'), 0) <= 1024 * 1024,
  'Use unique files and at most 1 MiB of replacement source.');
export type AgentSourceSetEdit = z.infer<typeof agentSourceSetSchema>;
export interface SourceProposalOutcome { status: 'applied' | 'cancelled' | 'refused'; detail: string }
export type SourceProposalReview = { title: string; candidate?: CandidateReviewSnapshot } & (
  { file: string; beforeDigest: string; afterDigest: string; files?: never }
  | { files: Array<{ file: string; beforeDigest: string; afterDigest: string }>; file?: never; beforeDigest?: never; afterDigest?: never });
export interface AcceptedModelChange {
  target: LocalBrowserTargetIdentity;
  file: string;
  files?: string[];
  current(): boolean;
}

type EditContext = NonNullable<ReturnType<LocalBrowserSync['sourceEditContext']>>;
export interface SourceEditsOptions {
  origin(): string | undefined;
  token(): string | undefined;
  context(target: LocalBrowserTargetIdentity): EditContext | undefined;
  capture?(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): CaptureSnapshot;
  capturePlan?(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): ResolvedExecutionPlan | undefined;
  captureObservations?(target: LocalBrowserTargetIdentity, captureId: string, sourceId: string): ExecutionObservation | undefined;
  runCandidate?: TraceWorker['runCandidate'];
  appliedCapture?(target: LocalBrowserTargetIdentity): AppliedCapture | undefined;
  checkAppliedSetup?(target: LocalBrowserTargetIdentity, live: AppliedCapture, signal: AbortSignal): Promise<boolean | undefined>;
  projectContext?(project: LocalBrowserProjectDeclareMessage['project']): ProjectDeclarationContext | undefined;
  publish(message: LocalBrowserHostPublish): unknown;
  channel: { appendLine(line: string): void };
  review?: typeof reviewSource;
  reviewSources?: typeof reviewSources;
  draft?(input: SourceDraftInput, signal: AbortSignal, current: () => boolean): SourceDraftHandle;
  accepted?(change: AcceptedModelChange): Promise<string>;
  fetchImpl?: typeof fetch;
}

export function sourceEditNodeIds(edit: LocalBrowserSourceEdit): string[] {
  if (edit.kind === 'group_layers') return edit.nodeIds;
  return [edit.kind === 'add_layer' ? edit.afterNodeId : edit.nodeId];
}

/** Translate semantic intent once, on the host, into the existing patch vocabulary. */
export function sourceEditOperations(edit: LocalBrowserSourceEdit, localIds: string[]): unknown[] {
  switch (edit.kind) {
    case 'insert_unary_op':
    case 'delete_unary_op':
    case 'reconnect_argument':
    case 'set_literal':
    case 'set_constructor_literal':
    case 'set_configuration_literal':
    case 'replace_activation': throw new Error('captured source edits require their exact recipe');
    case 'set_param': return [{ op: 'set_param', op_id: 'edit', node_id: localIds[0], param: edit.param, value: edit.value }];
    case 'remove_layer': return [{ op: 'remove_layer', op_id: 'delete', node_id: localIds[0] }];
    case 'group_layers': return [{ op: 'group_layers', op_id: 'group', node_id: edit.name, node_ids: localIds }];
    case 'add_layer': return [
      { op: 'add_layer', op_id: 'add', node_id: edit.nodeId, layer_op: edit.layerType, params: edit.params },
      { op: 'connect', op_id: 'place', source_node_id: localIds[0], target_node_id: edit.nodeId },
    ];
  }
}

/** Reconstruct the reviewed artifact and reject bad ranges/digest disagreements. */
export function sourceArtifact(document: vscode.TextDocument, rel: string, edits: SourceEdit[], digest: string): CodeArtifact {
  const before = document.getText();
  const lines = before.split('\n').map(line => line.replace(/\r$/, ''));
  const changes = edits.map(edit => {
    if (![...edit.range.start, ...edit.range.end].every(Number.isInteger)) throw new Error('invalid patch');
    for (const [line, col] of [edit.range.start, edit.range.end]) {
      if (line < 1 || line > lines.length || col < 0 || col > lines[line - 1].length) throw new Error('patch range outside source');
    }
    const range = toRange(edit.range);
    const start = document.offsetAt(range.start), end = document.offsetAt(range.end);
    if (start > end || typeof edit.newText !== 'string') throw new Error('invalid patch');
    return { edit, start, end };
  }).sort((a, b) => b.start - a.start);
  let after = before, previousStart = before.length + 1;
  for (const change of changes) {
    if (change.end > previousStart) throw new Error('overlapping patch');
    after = after.slice(0, change.start) + change.edit.newText + after.slice(change.end);
    previousStart = change.start;
  }
  if (sha256(after) !== digest || after === before) throw new Error('patch digest mismatch');
  return {
    path: rel, before, before_digest: sha256(before), after, after_digest: digest,
    edits: edits.map((edit, index) => ({
      op: 'edit_code', op_id: `edit-${index}`, path: rel, range: edit.range,
      before_digest: sha256(before), after: edit.newText,
    })),
  };
}

export class LocalBrowserSourceEdits {
  private pending?: { id: string; controller: AbortController; candidate?: CandidateReview; reviewReady?: boolean; accepted?: AcceptedModelChange };
  private readonly candidates = new CandidateHistory();
  private readonly applied = new AppliedSourceReceipts();
  private readonly appliedInspector: AppliedSourceInspector;
  private seen = new Set<string>();
  get reviewing(): boolean { return !!this.pending; }
  constructor(private readonly options: SourceEditsOptions) {
    this.appliedInspector = new AppliedSourceInspector(this.applied, { origin: options.origin, token: options.token,
      trusted: () => vscode.workspace.isTrusted, live: target => options.appliedCapture?.(target),
      checkSetup: options.checkAppliedSetup, fetchImpl: options.fetchImpl });
  }

  /** Lets a transport preserve another surface's in-flight request on disconnect. */
  get activeRequestId(): string | undefined { return this.pending?.id; }

  cancel(requestId?: string): void {
    if (!requestId || this.pending?.id === requestId) this.pending?.controller.abort();
  }

  /** Transport adapters must authorize the proposal owner before calling these. */
  runCandidate(requestId: string): CandidateRecord {
    const candidate = this.pending?.id === requestId ? this.pending.candidate : undefined;
    if (!candidate?.available || this.pending?.reviewReady === false) throw new GraphToolError('CANDIDATE_UNAVAILABLE', 'This proposal has no open native review with a retained replay recipe.');
    void candidate.run();
    return this.candidates.get(requestId, candidate.snapshot().runId!);
  }
  cancelCandidate(requestId: string, runId: string): CandidateRecord {
    this.candidates.get(requestId, runId);
    if (this.pending?.id === requestId && this.pending.candidate?.snapshot().runId === runId) this.pending.candidate.cancel();
    return this.getCandidate(requestId, runId);
  }
  listCandidates(requestId: string): CandidateRecord[] {
    if (this.pending?.id === requestId) this.pending.candidate?.snapshot();
    return this.candidates.list(requestId);
  }
  getCandidate(requestId: string, runId: string): CandidateRecord {
    if (this.pending?.id === requestId) this.pending.candidate?.snapshot();
    return this.candidates.get(requestId, runId);
  }
  inspectCandidate(requestId: string, runId: string, query: CaptureInspectionQuery) {
    return this.candidates.inspect(requestId, runId, query);
  }
  inspectCandidateObservations(requestId: string, runId: string, query: CandidateObservationQuery) {
    return this.candidates.inspectObservations(requestId, runId, query);
  }
  inspectCandidateCheckpoint(requestId: string, runId: string, query: { offset?: number; limit?: number }) {
    return this.candidates.inspectCheckpoint(requestId, runId, query);
  }
  inspectCandidateCheckpointText(requestId: string, runId: string, query: { offset: number; limit: number }) {
    return this.candidates.inspectCheckpointText(requestId, runId, query);
  }
  appliedSource(requestId: string): AppliedSourceReceipt | null { return this.applied.summary(requestId); }
  inspectAppliedSource(requestId: string, signal?: AbortSignal) { return this.appliedInspector.inspect(requestId, signal); }
  forgetCandidates(requestId: string): void {
    this.candidates.forget(requestId); this.applied.forget(requestId); this.appliedInspector.cancel(requestId);
  }
  dispose(): void { this.cancel(); this.candidates.clear(); this.applied.dispose(); this.appliedInspector.cancel(); }

  handle(message: LocalBrowserSourceEditRequestMessage): void {
    if (this.seen.has(message.requestId)) return;
    this.seen.add(message.requestId);
    if (this.seen.size > 256) this.seen.delete(this.seen.values().next().value!);
    void this.propose(message).then(outcome => this.options.publish({ type: 'sourceEditResult',
      requestId: message.requestId, target: message.target, ...outcome }));
  }

  /** Shared by the browser and MCP: one native review, with the same freshness guard. */
  async propose(message: SourceProposalRequest, signal?: AbortSignal,
    onReview?: (review: SourceProposalReview) => void): Promise<SourceProposalOutcome> {
    return this.withReview(message.requestId, abort => this.run(message, abort, onReview), signal);
  }

  /** Project intent shares the browser/MCP reservation, native diff and writer. */
  async proposeDeclaration(message: LocalBrowserProjectDeclareMessage): Promise<SourceProposalOutcome> {
    return this.withReview(message.messageId, signal => this.runDeclaration(message, signal));
  }

  /** Canvas agents can edit the owning workspace model, even when its layers are inherited. */
  async readModelSource(target: LocalBrowserTargetIdentity, file = target.file) {
    const current = this.modelFileGuard(target);
    const rootDocument = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, target.file)));
    const dependencyDigest = this.options.context(target)?.dependencies.get(file);
    if (!isNormalizedPythonPath(file) || file !== target.file && !dependencyDigest) {
      throw new GraphToolError('SOURCE_UNAVAILABLE', 'Read only the model file or a Python dependency recorded by this graph.');
    }
    const root = fs.realpathSync(target.workspaceFolder), absolute = path.join(root, file);
    const identity = fs.lstatSync(absolute);
    const regular = () => {
      try {
        const now = fs.lstatSync(absolute);
        return now.isFile() && now.dev === identity.dev && now.ino === identity.ino && fs.realpathSync(absolute) === absolute;
      } catch { return false; }
    };
    if (!regular()) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Graph source must be a regular workspace Python file without symbolic links.');
    const document = file === target.file ? rootDocument : await vscode.workspace.openTextDocument(vscode.Uri.file(absolute));
    const source = document.getText();
    if (!current() || !regular() || rootDocument.version !== target.documentVersion || sha256(rootDocument.getText()) !== target.sourceDigest
      || file !== target.file && (sha256(source) !== dependencyDigest || this.options.context(target)?.dependencies.get(file) !== dependencyDigest)) {
      throw new GraphToolError('STALE_TARGET', 'The model or dependency source changed. Refresh the graph and read its source again.');
    }
    if (Buffer.byteLength(source, 'utf8') > AGENT_SOURCE_LIMIT) throw new GraphToolError('SOURCE_LIMIT', 'This Python file exceeds the 256 KiB Canvas source limit.');
    const dependencies = [...(this.options.context(target)?.dependencies.keys() ?? [])].filter(name => name !== target.file && isNormalizedPythonPath(name)).sort();
    return { file, source, digest: sha256(source), documentVersion: document.version,
      ...(file === target.file ? { dependencies: dependencies.slice(0, 64), dependenciesTruncated: dependencies.length > 64 } : {}) };
  }

  /** Current base buffers only. The existing reader owns file/trust authority;
   * the parser receives text and never opens or imports project files. */
  async querySourceContext(target: LocalBrowserTargetIdentity, requested: SourceContextQuery): Promise<SourceContextResult> {
    const parsed = sourceContextQuerySchema.safeParse(requested);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use a bounded source symbol or exact index anchors.');
    const query = parsed.data, current = this.modelFileGuard(target);
    const dependencies = new Map(this.options.context(target)!.dependencies);
    if (query.file !== target.file && !dependencies.has(query.file)) throw new GraphToolError('SOURCE_UNAVAILABLE', 'Select the model or a graph-recorded workspace Python dependency.');
    const available = [...new Set([target.file, ...dependencies.keys()])].filter(isNormalizedPythonPath);
    const names = [...new Set([target.file, query.file, ...available.sort()])].slice(0, 16);
    const read = await Promise.all(names.map(file => this.readModelSource(target, file)));
    let bytes = 0;
    const sources = read.filter(source => {
      const size = Buffer.byteLength(source.source);
      if (bytes + size > 1024 * 1024) return false;
      bytes += size; return true;
    });
    const origin = this.options.origin(), token = this.options.token();
    if (!origin || !token) throw new GraphToolError('SOURCE_UNAVAILABLE', 'The local parser is unavailable.');
    const stillCurrent = () => current() && origin === this.options.origin() && token === this.options.token()
      && isDeepStrictEqual(this.options.context(target)?.dependencies, dependencies)
      && sources.every(source => {
        const absolute = path.join(target.workspaceFolder, source.file);
        const document = vscode.workspace.textDocuments.find(item => matchesHostFile(item.uri, absolute));
        try { return !!document && document.version === source.documentVersion && sha256(document.getText()) === source.digest
          && fs.lstatSync(absolute).isFile() && fs.realpathSync(absolute) === path.join(fs.realpathSync(target.workspaceFolder), source.file); }
        catch { return false; }
      });
    const check = () => { if (!stillCurrent()) throw new GraphToolError('STALE_TARGET', 'Source or dependencies changed. Refresh the graph and repeat the query.'); };
    check();
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 10_000);
    try {
      const response = await (this.options.fetchImpl ?? fetch)(`${origin}/api/ide/source-context`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': token }, redirect: 'error',
        body: JSON.stringify({ sources: sources.map(({ file, source, digest }) => ({ file, source, digest })), query }), signal: controller.signal,
      });
      const reader = response.body?.getReader();
      if (!reader) throw new GraphToolError('SOURCE_CONTEXT_INVALID', 'The parser returned no source context.');
      const chunks: Uint8Array[] = []; let size = 0;
      try {
        for (;;) {
          const next = await reader.read(); if (next.done) break;
          size += next.value.byteLength;
          if (size > 64 * 1024) throw new GraphToolError('SOURCE_LIMIT', 'The source context exceeded its response limit.');
          chunks.push(next.value);
        }
      } finally { await reader.cancel(); }
      const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks)));
      check();
      if (!response.ok) {
        const reason = object(object(value).detail).reason;
        const messages = new Map([
          ['source_index_changed', 'The supplied source index changed. Start a new symbol query before following old anchors.'],
          ['unknown_source_symbol', 'No matching source symbol exists in the selected file. Inspect its source and select an exact name or line.'],
          ['unknown_source_anchor', 'This source anchor is not in the current index. Start a new symbol query.'],
          ['source_index_limit', 'These files exceed the lexical index limit. A smaller graph source inventory is required.'],
          ['source_result_limit', 'The source result exceeds its page limit. Reduce the query scope.'],
          ['configuration_change_unavailable', 'Select one direct unambiguous integer assignment by symbol and line. Dynamic, conditional and computed definitions are not supported.'],
        ]);
        if (response.status === 422 && typeof reason === 'string' && messages.has(reason)) {
          throw new GraphToolError(reason.toUpperCase(), messages.get(reason)!);
        }
        throw new GraphToolError('SOURCE_CONTEXT_UNAVAILABLE', 'The parser refused this source query. Check the selector, source syntax and supported file limits.');
      }
      const page = sourceContextPage(value, sources, query);
      return { ...page, collection: { availableFiles: available.length, omittedFiles: available.length - sources.length,
        byteLimit: 1024 * 1024, fileLimit: 16 } };
    } catch (error) {
      if (error instanceof GraphToolError) throw error;
      throw new GraphToolError('SOURCE_CONTEXT_UNAVAILABLE', 'The local source query failed or timed out. No source was changed or executed.');
    } finally { clearTimeout(timer); }
  }

  private modelFileGuard(target: LocalBrowserTargetIdentity, relative = target.file): () => boolean {
    if (!vscode.workspace.isTrusted || !this.options.context(target) || !isNormalizedPythonPath(relative)) {
      throw new GraphToolError('STALE_TARGET', 'Source changes need a trusted workspace and a current model.');
    }
    const root = fs.realpathSync(target.workspaceFolder), file = path.join(root, relative);
    const identity = fs.lstatSync(file);
    const current = () => {
      try {
        const now = fs.lstatSync(file);
        return vscode.workspace.isTrusted && !!this.options.context(target) && fs.realpathSync(target.workspaceFolder) === root
          && fs.realpathSync(file) === file && now.isFile() && now.dev === identity.dev && now.ino === identity.ino;
      } catch { return false; }
    };
    if (!current()) throw new GraphToolError('SOURCE_UNAVAILABLE', 'The model must be a regular Python file inside this workspace, without symbolic links.');
    return current;
  }

  private async checkPythonSource(file: string, source: string, signal: AbortSignal, checkCompilation = false): Promise<void> {
    const origin = this.options.origin(), token = this.options.token();
    if (!origin || !token) throw new GraphToolError('SOURCE_UNAVAILABLE', 'The syntax-check service is unavailable.');
    const response = await (this.options.fetchImpl ?? fetch)(`${origin}/api/ide/buffer-classes`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': token },
      body: JSON.stringify({ file, source, ...(checkCompilation ? { check_compilation: true } : {}) }), signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => undefined) as { detail?: { error?: string; message?: string; line?: number; column?: number } } | undefined;
      const error = body?.detail;
      if (response.status === 400 && error?.error === 'syntax_error' && typeof error.message === 'string') {
        throw new GraphToolError('SYNTAX_ERROR', `Python syntax error at line ${error.line ?? '?'}:${(error.column ?? 0) + 1}: ${error.message}. Correct it and propose again.`);
      }
      throw new GraphToolError('SOURCE_UNAVAILABLE', 'The syntax-check service is unavailable. No source change was applied; reconnect and retry.');
    }
  }

  /** Validate the entire replacement before a human-selected pending review closes. */
  async checkSourceSet(target: LocalBrowserTargetIdentity, value: AgentSourceSetEdit, signal: AbortSignal): Promise<void> {
    const edit = agentSourceSetSchema.parse(value);
    const guards = [this.modelFileGuard(target), ...edit.sources.map(item => this.modelFileGuard(target, item.file))];
    const dependencies = new Map(this.options.context(target)!.dependencies);
    const origin = this.options.origin(), token = this.options.token();
    let captured: CaptureSnapshot | undefined, plan: ResolvedExecutionPlan | undefined;
    if (edit.capture) {
      if (!this.options.capture) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'The selected capture is unavailable.');
      captured = this.options.capture(target, edit.capture.captureId, edit.capture.sourceId);
      plan = this.options.capturePlan?.(target, captured.id, captured.source.id);
      const files = captured.source.files;
      if (edit.sources.some(item => !files.some(file => file.role === 'source'
        && file.path === item.file && file.digest === item.beforeDigest))) {
        throw new GraphToolError('STALE_TARGET', 'Every proposed file must match the selected capture before candidate review.');
      }
    }
    for (const item of edit.sources) {
      if (signal.aborted) throw new GraphToolError('CANCELLED', 'Draft revision cancelled.');
      const live = await this.readModelSource(target, item.file);
      if (live.digest !== item.beforeDigest) throw new GraphToolError('STALE_TARGET', 'Read every current file before proposing its replacement.');
      if (item.source === live.source) throw new GraphToolError('UNCHANGED_SOURCE', 'Omit unchanged files from a source-set proposal.');
    }
    await Promise.all(edit.sources.map(item => this.checkPythonSource(item.file, item.source, signal, true)));
    for (const item of edit.sources) {
      const live = await this.readModelSource(target, item.file);
      if (signal.aborted || live.digest !== item.beforeDigest) throw new GraphToolError('STALE_TARGET', 'Source changed during proposal preparation.');
    }
    if (!guards.every(guard => guard()) || !isDeepStrictEqual(this.options.context(target)?.dependencies, dependencies)
      || origin !== this.options.origin() || token !== this.options.token()
      || captured && (!isDeepStrictEqual(this.options.capture?.(target, captured.id, captured.source.id), captured)
        || !isDeepStrictEqual(this.options.capturePlan?.(target, captured.id, captured.source.id), plan))) {
      throw new GraphToolError('STALE_TARGET', 'The source or capture context changed during proposal preparation.');
    }
  }

  async proposeSourceSet(message: { requestId: string; target: LocalBrowserTargetIdentity; edit: AgentSourceSetEdit }, signal?: AbortSignal,
    onReview?: (review: SourceProposalReview) => void, onDecision?: () => void): Promise<SourceProposalOutcome> {
    return this.withReview(message.requestId, async abort => {
      const { target } = message, edit = agentSourceSetSchema.parse(message.edit);
      if (this.pending?.id === message.requestId) this.pending.reviewReady = false;
      const modelCurrent = this.modelFileGuard(target);
      await this.readModelSource(target);
      const graphDependencies = new Map(this.options.context(target)!.dependencies);
      const dependencies = new Map(graphDependencies), resources: Array<{ path: string; digest: string }> = [];
      let captured: CaptureSnapshot | undefined, retainedPlan: ResolvedExecutionPlan | undefined;
      if (edit.capture) {
        if (!this.options.capture) throw new GraphToolError('CAPTURE_UNAVAILABLE', 'The selected capture is unavailable.');
        captured = this.options.capture(target, edit.capture.captureId, edit.capture.sourceId);
        retainedPlan = this.options.capturePlan?.(target, captured.id, captured.source.id);
        for (const file of captured.source.files) {
          if (file.role !== 'source') { resources.push(file); continue; }
          if (dependencies.has(file.path) && dependencies.get(file.path) !== file.digest) throw new GraphToolError('STALE_TARGET', 'Captured source differs from the current graph.');
          dependencies.set(file.path, file.digest);
        }
      }
      if (dependencies.size > SOURCE_SET_LIMITS.maxSources) throw new GraphToolError('SOURCE_LIMIT', 'The review source inventory exceeds the projection limit.');
      const resourcesCurrent = await reviewResourceGuard(target.workspaceFolder, resources, abort,
        absolute => vscode.workspace.textDocuments.some(document => matchesHostFile(document.uri, absolute) && document.isDirty));
      const documents: ReviewDocuments = new Map(), guards: Array<() => boolean> = [modelCurrent];
      let bytes = 0;
      for (const [file, digest] of dependencies) {
        guards.push(this.modelFileGuard(target, file));
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, file)));
        const source = document.getText(), size = Buffer.byteLength(source, 'utf8'); bytes += size;
        if (size > SOURCE_SET_LIMITS.maxSourceBytes || bytes > SOURCE_SET_LIMITS.maxTotalBytes) throw new GraphToolError('SOURCE_LIMIT', 'The review source inventory exceeds the projection byte limit.');
        if (sha256(source) !== digest) throw new GraphToolError('STALE_TARGET', 'Source changed after this graph was projected.');
        documents.set(file, { document, version: document.version, digest });
      }
      const origin = this.options.origin(), token = this.options.token();
      if (!origin || !token) throw new GraphToolError('SOURCE_UNAVAILABLE', 'The syntax-check service is unavailable.');
      const captureCurrent = () => {
        if (!captured) return true;
        try { return isDeepStrictEqual(this.options.capture?.(target, captured.id, captured.source.id), captured)
          && isDeepStrictEqual(this.options.capturePlan?.(target, captured.id, captured.source.id), retainedPlan); } catch { return false; }
      };
      const stillCurrent = () => !abort.aborted && guards.every(guard => guard()) && resourcesCurrent() && captureCurrent()
        && isDeepStrictEqual(this.options.context(target)?.dependencies, graphDependencies)
        && this.options.origin() === origin && this.options.token() === token
        && [...documents.values()].every(item => item.document.version === item.version && sha256(item.document.getText()) === item.digest);
      if (!stillCurrent()) throw new GraphToolError('STALE_TARGET', 'The review context changed.');
      await this.checkSourceSet(target, edit, abort);
      if (!stillCurrent()) throw new GraphToolError('STALE_TARGET', 'The review context changed.');
      const artifacts = edit.sources.map(item => {
        const snapshot = documents.get(item.file);
        if (!snapshot || snapshot.digest !== item.beforeDigest) throw new GraphToolError('STALE_TARGET', 'The source set no longer matches its reviewed base.');
        const lines = snapshot.document.getText().split('\n');
        return sourceArtifact(snapshot.document, item.file, [{ range: { start: [1, 0],
          end: [lines.length, lines.at(-1)!.replace(/\r$/, '').length] }, newText: item.source }], sha256(item.source));
      });
      const candidateInput = captured && retainedPlan && this.options.runCandidate ? structuredClone({ version: 2 as const,
        plan: retainedPlan, capture: captured, reviewedSources: edit.sources }) : undefined;
      const candidate = candidateInput ? new CandidateReview(candidateInput.capture, abort, stillCurrent, signal =>
        this.options.runCandidate!({ workspace: target.workspaceFolder, file: path.join(target.workspaceFolder, target.file), cls: target.cls,
          candidate: candidateInput, current: async () => stillCurrent() }, signal), { history: this.candidates, requestId: message.requestId,
          baseline: this.options.captureObservations?.(target, candidateInput.capture.id, candidateInput.capture.source.id) }) : undefined;
      if (this.pending?.id === message.requestId) this.pending.candidate = candidate;
      const title = `tensorViz agent source set — ${artifacts.length} file${artifacts.length === 1 ? '' : 's'}; syntax checked${candidate ? '' : ', runtime not run'}`;
      const reservation = captured ? this.applied.reserve(message.requestId, target, { version: 2, base: captured,
        files: artifacts.map(item => ({ file: item.path, beforeDigest: item.before_digest, afterDigest: item.after_digest })) }) : undefined;
      if (captured && !reservation) {
        await candidate?.finish();
        return this.outcome('refused', 'This source request is no longer available. Prepare a new proposal.');
      }
      try {
        const outcome = await this.reviewAndWrite(target.workspaceFolder, title,
          artifacts, abort, stillCurrent, documents, onReview, candidate, (reveal, decide) => {
            if (this.pending?.id === message.requestId) this.pending.reviewReady = true;
          }, undefined, { id: message.requestId, target, canRevise: true }, () => {
            if (this.pending?.id === message.requestId) this.pending.reviewReady = false;
            onDecision?.();
          });
        if (outcome.status === 'applied' && reservation) this.applied.confirm(message.requestId, reservation);
        const runtime = candidate?.snapshot().runtime ?? 'not_run';
        return outcome.status === 'applied' ? this.outcome('applied', outcome.detail + (runtime === 'not_run'
          ? ' Source syntax checked; runtime validation has not run.'
          : ` Copied candidate runtime: ${runtime.replace('_', ' ')}. Semantic postconditions: not checked.${this.options.accepted ? '' : ' Save and Validate the live source.'}`)) : outcome;
      } finally {
        if (reservation) this.applied.discard(message.requestId, reservation);
        if (candidate) try {
          onReview?.({ title, files: artifacts.map(item => ({ file: item.path, beforeDigest: item.before_digest,
            afterDigest: item.after_digest })), candidate: candidate.snapshot() });
        } catch { this.options.channel.appendLine('[source] Could not publish the final candidate summary; the source outcome is unchanged.'); }
      }
    }, signal);
  }

  private async withReview(id: string, run: (signal: AbortSignal) => Promise<SourceProposalOutcome>,
    signal?: AbortSignal): Promise<SourceProposalOutcome> {
    if (signal?.aborted) return this.outcome('cancelled', 'Proposal cancelled. No source change was applied.');
    if (this.pending) return this.outcome('refused', 'Another source review is open. Finish or discard it first.');
    const controller = new AbortController();
    const abort = () => controller.abort();
    const pending: NonNullable<LocalBrowserSourceEdits['pending']> = { id, controller };
    this.pending = pending;
    signal?.addEventListener('abort', abort, { once: true });
    // Bound an abandoned native review as well as preparation; client/session
    // disposal normally cancels sooner. A timed-out proposal can never apply later.
    const timeout = setTimeout(abort, 30 * 60_000);
    let outcome: SourceProposalOutcome;
    try { outcome = await run(controller.signal); }
    catch (error) {
      outcome = this.outcome(controller.signal.aborted ? 'cancelled' : 'refused', controller.signal.aborted
        ? 'Proposal cancelled. No source change was applied.'
        : error instanceof CapturedEditError || error instanceof ReviewResourceError || error instanceof GraphToolError ? error.message
        : 'The source proposal could not be prepared or applied. Refresh the graph and try again.');
    } finally {
      clearTimeout(timeout); signal?.removeEventListener('abort', abort);
      if (this.pending?.controller === controller) this.pending = undefined;
    }
    // Release the review before opening: retargeting cancels pending proposals.
    // A failed save/run must never turn a confirmed source write into a refusal.
    if (outcome.status === 'applied' && pending.accepted && this.options.accepted) {
      let detail: string;
      try { detail = await this.options.accepted(pending.accepted); }
      catch {
        detail = 'The change was applied, but automatic capture could not start. Inspect the source, then use Run model.';
        void vscode.window.showWarningMessage(detail);
      }
      outcome = this.outcome('applied', detail + ' Use native Undo to revert.' + (outcome.detail.includes('Copied candidate runtime:')
        ? ' ' + outcome.detail.slice(outcome.detail.indexOf('Copied candidate runtime:')) : ''));
    }
    return outcome;
  }

  private outcome(status: SourceProposalOutcome['status'], detail: string): SourceProposalOutcome {
    return { status, detail: detail.slice(0, 512) };
  }

  private async run(message: SourceProposalRequest, signal: AbortSignal,
    onReview?: (review: SourceProposalReview) => void, onShown?: () => void,
    onApplied?: (uri: vscode.Uri, artifact: CodeArtifact) => void): Promise<SourceProposalOutcome> {
    const { target, edit } = message;
    const context = this.options.context(target);
    if (!vscode.workspace.isTrusted || !context || !isLocalBrowserSourceEdit(edit)) {
      return this.outcome('refused', 'Source changes need a trusted workspace and a current graph. Wait for refresh and try again.');
    }
    // A captured location proves navigation, not a legacy declaration recipe.
    // Never let its span reach the codemod's line-based declaration fallback.
    const semantic = (context.projection.document as { semantic?: { metadata?: Record<string, unknown>; nodes?: Array<{ node_id: string; metadata?: Record<string, unknown> }> } } | undefined)?.semantic;
    const capturedEdit = 'capabilityId' in edit;
    const canonical = !!(semantic?.metadata?.capture || semantic?.nodes?.some(node => node.metadata?.captureRef));
    if (canonical && !capturedEdit) {
      return this.outcome('refused', 'Captured records need an explicit source-edit capability. A source location alone cannot authorize a layer edit.');
    }
    let captured: CaptureSnapshot | undefined;
    const dependencies = new Map(context.dependencies);
    const resources: Array<{ path: string; digest: string }> = [];
    if (capturedEdit) {
      if (!canonical || !this.options.capture) return this.outcome('refused', 'This graph has no retained source-edit capability.');
      captured = this.options.capture(target, edit.target.captureId, edit.target.sourceId);
      for (const file of captured.source.files) {
        if (file.role === 'native') {
          if (dependencies.has(file.path)) return this.outcome('refused', 'This resource has a conflicting source role. Validate the saved setup again.');
          resources.push(file);
          continue;
        }
        if (dependencies.has(file.path) && dependencies.get(file.path) !== file.digest) {
          return this.outcome('refused', 'Captured source differs from the current projection. Validate the current saved source again.');
        }
        dependencies.set(file.path, file.digest);
      }
    }
    const origin = this.options.origin(), token = this.options.token();
    if (!origin || !token) throw new Error('backend unavailable');
    const backend = { origin: () => origin, headers: () => ({ 'X-tensorViz-Token': token }) };
    const resourcesCurrent = await reviewResourceGuard(target.workspaceFolder, resources, signal,
      absolute => vscode.workspace.textDocuments.some(document => matchesHostFile(document.uri, absolute) && document.isDirty));
    const documents: ReviewDocuments = new Map();
    for (const [file, digest] of dependencies) {
      if (!isNormalizedPythonPath(file) && !(captured && file === 'tensorviz.toml')) throw new Error('invalid dependency path');
      const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, file)));
      if (sha256(document.getText()) !== digest) {
        return this.outcome('refused', 'Source changed after this graph was projected. Wait for refresh and try again.');
      }
      documents.set(file, { document, version: document.version, digest });
    }
    const captureCurrent = () => {
      if (!captured) return true;
      try { return !!this.options.capture?.(target, captured.id, captured.source.id); } catch { return false; }
    };
    const stillCurrent = () => !signal.aborted && vscode.workspace.isTrusted && !!this.options.context(target) &&
      this.options.origin() === origin && this.options.token() === token && captureCurrent() && resourcesCurrent() &&
      [...documents.values()].every(item => item.document.version === item.version && sha256(item.document.getText()) === item.digest);
    if (!stillCurrent()) throw new Error('stale source');

    if (capturedEdit && captured) {
      const metadata = semantic?.nodes?.find(node => node.node_id === edit.nodeId)?.metadata;
      const capabilities = metadata?.sourceEditCapabilities;
      const capability = Array.isArray(capabilities) ? capabilities.find(cap => cap?.id === edit.capabilityId) : undefined;
      if (!isSourceEditCapability(capability) || capability.command !== edit.kind || !isDeepStrictEqual(metadata?.captureRef, edit.target) ||
        capability.scope.kind !== edit.scope || !isDeepStrictEqual(capability.target, edit.target) || !isDeepStrictEqual(capability.expected, edit.expected) ||
        ('argumentPath' in edit && (!('argumentPath' in capability) || !isDeepStrictEqual(capability.argumentPath, edit.argumentPath))) ||
        ((edit.kind === 'set_constructor_literal' || edit.kind === 'set_configuration_literal') &&
          ((capability.command !== 'set_constructor_literal' && capability.command !== 'set_configuration_literal') || !isDeepStrictEqual(capability.sourceArgumentPath, edit.sourceArgumentPath)))) {
        return this.outcome('refused', 'That exact edit capability is unavailable for this selected node. Refresh the graph and inspect its supported edits.');
      }
      const { nodeId: _node, ...intent } = edit;
      const sources = Object.fromEntries(captured.source.files.filter(file => file.role === 'source')
        .map(file => [file.path, documents.get(file.path)!.document.getText()]));
      const plan = await prepareCapturedSourceEdit(captured, sources, intent, capability,
        { origin, token, signal, fetchImpl: this.options.fetchImpl });
      if (!stillCurrent()) throw new Error('stale source');
      if (!plan.edits.length) return this.outcome('refused', 'The requested source value is unchanged. No source edit or runtime validation was performed.');
      const snapshot = documents.get(plan.file);
      if (!snapshot || snapshot.digest !== plan.expected_digest) throw new Error('unknown source dependency');
      const artifact = sourceArtifact(snapshot.document, plan.file, plan.edits, plan.source_digest);
      const retainedPlan = this.options.runCandidate && this.options.capturePlan?.(target, captured.id, captured.source.id);
      const candidateInput = retainedPlan ? structuredClone({ version: 1 as const, plan: retainedPlan, capture: captured,
        intent, reviewedFile: artifact.path, reviewedSource: artifact.after }) : undefined;
      const candidate = candidateInput ? new CandidateReview(captured, signal, stillCurrent, abort =>
        this.options.runCandidate!({ workspace: target.workspaceFolder, file: path.join(target.workspaceFolder, target.file),
          cls: target.cls, candidate: candidateInput, current: async () => stillCurrent() }, abort),
        { history: this.candidates, requestId: message.requestId,
          baseline: this.options.captureObservations?.(target, candidateInput.capture.id, candidateInput.capture.source.id) }) : undefined;
      if (this.pending?.id === message.requestId) this.pending.candidate = candidate;
      const scope = capability.scope;
      const impact = scope.kind === 'configuration_definition'
        ? `all references (${scope.knownSourceReferences} known source references, ${scope.observedConstructorSites} observed constructor sites)`
        : 'all executions';
      const options = capability.command === 'replace_activation' && capability.options?.inplace != null
        ? `; inplace=${capability.options.inplace ? 'True' : 'False'} preserved` : '';
      const title = `tensorViz ${edit.kind === 'insert_unary_op' ? `activation insertion on operand ${Number(edit.argumentPath[1]) + 1}` : edit.kind === 'delete_unary_op' ? 'activation bypass' : edit.kind === 'reconnect_argument' ? `operand ${Number(edit.argumentPath[1]) + 1} reconnection` : edit.kind === 'set_literal' ? 'literal' : edit.kind === 'set_constructor_literal' ? 'constructor argument' : edit.kind === 'set_configuration_literal' ? 'configuration definition' : 'activation'} — ${plan.file} — ${impact} (${scope.observedOperations} captured operation${scope.observedOperations === 1 ? '' : 's'}, ${scope.observedModuleInstances} instance${scope.observedModuleInstances === 1 ? '' : 's'})${options}; runtime not run`;
      const reviewTitle = candidate ? title.replace('; runtime not run', '; static checks passed') : title;
      const reservation = this.applied.reserve(message.requestId, target, { version: 1, base: captured,
        sources, intent, reviewedFile: artifact.path, reviewedSource: artifact.after });
      if (!reservation) {
        await candidate?.finish();
        return this.outcome('refused', 'This source request is no longer available. Prepare a new proposal.');
      }
      let outcome: SourceProposalOutcome;
      try {
        outcome = await this.reviewAndWrite(target.workspaceFolder, reviewTitle,
          artifact, signal, stillCurrent, documents, onReview, candidate, onShown, onApplied,
          { id: message.requestId, target, canRevise: false });
        // Apply can finish after an owner disconnects. Preserve the actual write
        // outcome while allowing only the original still-retained reservation to
        // confirm evidence; never create a new receipt after asynchronous work.
        if (outcome.status === 'applied') this.applied.confirm(message.requestId, reservation);
      } finally {
        this.applied.discard(message.requestId, reservation);
        // A completed Apply changes the original source context. Publish its
        // final candidate attribution without letting an observer undo the receipt.
        if (candidate) try {
          onReview?.({ title: reviewTitle, file: artifact.path, beforeDigest: artifact.before_digest,
            afterDigest: artifact.after_digest, candidate: candidate.snapshot() });
        } catch {
          this.options.channel.appendLine('[source] Could not publish the final candidate summary; the source outcome is unchanged.');
        }
      }
      const summary = candidate?.snapshot(), runtime = summary?.runtime ?? 'not_run';
      const semantics = summary?.postconditions ? candidatePostconditionsText(summary) : 'not checked.';
      return outcome.status === 'applied' ? this.outcome('applied', outcome.detail + (runtime === 'not_run'
        ? ' Source syntax checked; runtime validation has not run.'
        : ` Copied candidate runtime: ${runtime.replace('_', ' ')}. Semantic postconditions: ${semantics}${this.options.accepted ? '' : ' Save and Validate the live source.'}`)) : outcome;
    }

    const ids = sourceEditNodeIds(edit);
    const spans = ids.map(id => context.projection.spanmap[id] as { file?: string; start?: number[]; end?: number[] } | undefined);
    const rel = spans[0]?.file;
    const snapshot = rel && documents.get(rel);
    if (!rel || !snapshot || spans.some(span => span?.file !== rel || !span.start || !span.end)) {
      return this.outcome('refused', 'Select declared layers in the same source file. Synthetic boundaries cannot be edited.');
    }
    const response = await (this.options.fetchImpl ?? fetch)(`${origin}/api/ide/buffer-classes`, {
      method: 'POST', headers: { ...backend.headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: rel, source: snapshot.document.getText() }), signal,
    });
    if (!response.ok) throw new Error('classes unavailable');
    const body = await response.json() as { classes: Array<{ name: string; line: number; end_line: number }> };
    const classes = body.classes.filter(cls => ids.length
      ? spans.every(span => span!.start![0] >= cls.line && span!.end![0] <= cls.end_line)
      : cls.name === target.cls)
      .sort((a, b) => (a.end_line - a.line) - (b.end_line - b.line));
    if (!classes.length) {
      return this.outcome('refused', 'The selection crosses class definitions. Select layers within one block.');
    }
    const localIds = ids.map(id => id.split('.').pop()!);
    if (new Set(localIds).size !== localIds.length) {
      return this.outcome('refused', 'Those instances share a source declaration. Select one instance of each layer.');
    }
    const shared = Object.entries(context.projection.spanmap).filter(([id, value]) => {
      const span = value as { file?: string; start?: number[]; end?: number[] };
      return !ids.includes(id) && spans.some(selected => selected?.file === span.file &&
        JSON.stringify(selected?.start) === JSON.stringify(span.start) && JSON.stringify(selected?.end) === JSON.stringify(span.end));
    }).length;
    const sharedTitle = shared ? ` — also affects ${shared} other projected instance${shared === 1 ? '' : 's'}` : '';
    const applier = new EditApplier(backend, this.options.channel as vscode.OutputChannel, undefined, this.options.fetchImpl);
    const planned = await applier.plan({ uri: snapshot.document.uri, cls: classes[0].name, rel,
      sourceDigest: snapshot.digest,
      spanmap: Object.fromEntries(localIds.map((id, i) => [id, spans[i]])),
    }, sourceEditOperations(edit, localIds));
    if (!planned || planned.refused.length || !planned.edits.length) {
      return this.outcome('refused', planned?.refused[0]?.message ?? 'This operation produced no source changes.');
    }
    if (!stillCurrent()) throw new Error('stale source');
    const artifact = sourceArtifact(snapshot.document, rel, planned.edits, planned.source_digest);
    const title = (edit.kind === 'group_layers'
      ? `Group as ${edit.name} — parameter paths and checkpoint keys will change`
      : `tensorViz source change — ${rel}`) + sharedTitle;
    return this.reviewAndWrite(target.workspaceFolder, title, artifact, signal, stillCurrent, documents, onReview, undefined, onShown, onApplied,
      { id: message.requestId, target, canRevise: false });
  }

  private async runDeclaration(message: LocalBrowserProjectDeclareMessage, signal: AbortSignal): Promise<SourceProposalOutcome> {
    const context = this.options.projectContext?.(message.project);
    if (!vscode.workspace.isTrusted || !context) {
      return this.outcome('refused', 'Declarations need a trusted workspace and the current project. Reopen the project graph and try again.');
    }
    const { declaration, args } = message;
    if (!isProjectDeclaration(declaration, args) ||
      context.labels.length > 4096 || context.labels.some(label => typeof label !== 'string' || label.length > 512) ||
      (declaration === 'edge' && args.slice(0, 2).some(label => !context.labels.includes(label)))) {
      return this.outcome('refused', 'That declaration does not name valid targets in the current experiment. Refresh the project and try again.');
    }
    const origin = this.options.origin(), token = this.options.token();
    if (!origin || !token) throw new Error('backend unavailable');
    const documents: ReviewDocuments = new Map();
    for (const [file, digest] of context.dependencies) {
      signal.throwIfAborted();
      if (!isNormalizedPythonPath(file)) throw new Error('invalid declaration dependency');
      const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(context.project.workspaceFolder, file)));
      if (sha256(document.getText()) !== digest) throw new Error('stale project source');
      documents.set(file, { document, version: document.version, digest });
    }
    const snapshot = documents.get(context.entrypoint);
    if (!snapshot) throw new Error('entrypoint unavailable');
    const stillCurrent = () => {
      const current = this.options.projectContext?.(context.project);
      return !signal.aborted && vscode.workspace.isTrusted && this.options.origin() === origin && this.options.token() === token &&
        current?.entrypoint === context.entrypoint &&
        [...documents.values()].every(item => item.document.version === item.version && sha256(item.document.getText()) === item.digest);
    };
    if (!stillCurrent()) throw new Error('stale project');
    const source = snapshot.document.getText();
    if (Buffer.byteLength(source, 'utf8') > PROJECT_LIMITS.maxBodyBytes) throw new Error('source too large');

    // Race decoding as well as fetch. A cancelled or timed-out adapter that
    // ignores AbortSignal may finish later, but cannot open a review or apply.
    const controller = new AbortController();
    let rejectAborted!: (error: Error) => void;
    const cancelled = new Promise<never>((_resolve, reject) => { rejectAborted = reject; });
    const abort = () => { controller.abort(); rejectAborted(new Error('declaration request cancelled')); };
    signal.addEventListener('abort', abort, { once: true });
    const timeout = setTimeout(abort, DECLARATION_TIMEOUT_MS);
    let patch: { status?: string; file?: string; expected_digest?: string; contents?: string } | null;
    try {
      const request = async () => {
        const response = await (this.options.fetchImpl ?? fetch)(`${origin}/api/ide/project/declare`, {
          method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': token },
          body: JSON.stringify({ file: context.entrypoint, source, source_digest: snapshot.digest,
            kind: declaration, args, known_labels: context.labels }), signal: controller.signal,
        });
        if (!response.ok) throw new Error('declaration backend refused');
        return response.json();
      };
      patch = await Promise.race([request(), cancelled]);
    } finally {
      clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
    }
    if (!stillCurrent()) throw new Error('stale project source');
    if (patch?.status !== 'ok' || patch.file !== context.entrypoint || patch.expected_digest !== snapshot.digest ||
      typeof patch.contents !== 'string' || Buffer.byteLength(patch.contents, 'utf8') > PROJECT_LIMITS.maxBodyBytes + 16_384) {
      throw new Error('invalid declaration response');
    }
    const end = snapshot.document.positionAt(source.length);
    const artifact = sourceArtifact(snapshot.document, context.entrypoint, [{
      range: { start: [1, 0], end: [end.line + 1, end.character] }, newText: patch.contents,
    }], sha256(patch.contents));
    // The project declaration caller already displays refusal warnings.
    return this.reviewAndWrite(context.project.workspaceFolder, `tensorViz project declaration — ${context.entrypoint}`,
      artifact, signal, stillCurrent, documents, undefined, undefined, undefined, undefined, undefined, undefined, false);
  }

  private async reviewAndWrite(folder: string, title: string, artifact: CodeArtifact | CodeArtifact[], signal: AbortSignal,
    stillCurrent: () => boolean, documents: ReviewDocuments,
    onReview?: (review: SourceProposalReview) => void, candidate?: CandidateReview, onShown?: (reveal: () => Promise<void>, decide: (approved: boolean) => void) => void,
    onApplied?: (uri: vscode.Uri, artifact: CodeArtifact) => void,
    draftInput?: Pick<SourceDraftInput, 'id' | 'target' | 'canRevise'>, onDecision?: () => void, notifyRefusal = true): Promise<SourceProposalOutcome> {
    if (draftInput && this.options.accepted) title += ' — Apply saves and runs the model';
    const artifacts = Array.isArray(artifact) ? artifact : [artifact];
    const files = artifacts.map(item => ({ file: item.path, beforeDigest: item.before_digest, afterDigest: item.after_digest }));
    const review: SourceProposalReview = Array.isArray(artifact) ? { title, files } : { title, ...files[0] };
    const changed = candidate?.onDidChange(value => onReview?.({ ...review, candidate: value }));
    let approved = false, applied = false;
    const refuse = (detail: string) => {
      this.options.channel.appendLine(`[source] ${detail}`);
      if (notifyRefusal) void vscode.window.showWarningMessage(detail);
      return this.outcome(signal.aborted ? 'cancelled' : 'refused', detail);
    };
    let draft: SourceDraftHandle | undefined;
    try {
      if (!stillCurrent()) throw new Error('stale source');
      if (draftInput) draft = this.options.draft?.({ ...draftInput, artifacts, sources: { truncated: false,
        entries: [...documents].filter(([file]) => isNormalizedPythonPath(file))
          .map(([file, snapshot]) => ({ file, source: snapshot.document.getText(), digest: snapshot.digest })) } }, signal, stillCurrent);
      const publishReview = () => onReview?.({ ...review, ...(candidate ? { candidate: candidate.snapshot() } : {}) });
      const shown = (reveal: () => Promise<void>, decide: (approved: boolean) => void) => {
        onShown?.(reveal, decide); draft?.show(reveal, decide);
      };
      if (!Array.isArray(artifact)) publishReview();
      approved = Array.isArray(artifact)
        ? await (this.options.reviewSources ?? reviewSources)(title, artifact, signal, candidate, (reveal, decide) => {
          shown(reveal, decide); publishReview();
        }, onDecision)
        : await (this.options.review ?? reviewSource)(title, artifact, signal, candidate, shown, onDecision);
      onDecision?.();
    } catch (error) {
      draft?.dispose();
      await candidate?.finish(); changed?.dispose();
      throw error;
    }
    try {
      // Wait for private execution cleanup while retaining the review and its guards.
      // A refused candidate run does not prevent applying the reviewed source.
      await candidate?.settle();
      if (candidate?.snapshot().cleanup === 'failed') {
        return refuse('Candidate cleanup failed. Private temporary files may remain. No source change was applied.');
      }
      if (!approved) {
        return this.outcome('cancelled', 'Proposal discarded. No source change was applied.');
      }
      if (!stillCurrent()) return refuse('The review context changed. No source change was applied. Refresh the graph and review the proposal again.');
      // Native diff previews can close source documents. A closed TextDocument
      // remains readable but no longer follows disk/reopened-buffer changes.
      // Rebind only after proving the same reviewed bytes, retaining the ordinary
      // version check for documents that stayed open throughout review.
      for (const snapshot of documents.values()) {
        if (!snapshot.document.isClosed) continue;
        const document = await vscode.workspace.openTextDocument(snapshot.document.uri);
        if (sha256(document.getText()) !== snapshot.digest) return refuse('The source changed after review. No source change was applied.');
        snapshot.document = document;
        snapshot.version = document.version;
      }
      const currentOpenSources = () => stillCurrent() && [...documents.values()].every(item => !item.document.isClosed);
      const writer = new SourceWriter(vscode.Uri.file(folder), this.options.channel);
      let result;
      try { result = await writer.write(artifacts, currentOpenSources); }
      catch {
        return refuse('The editor did not confirm the source change. Inspect the buffer before retrying; use VS Code Undo if needed.');
      }
      if (result.outcome !== 'applied') return refuse(result.reason ?? 'The editor refused the source change. Inspect the buffer before retrying.');
      applied = true;
      for (const item of artifacts) onApplied?.(vscode.Uri.file(path.join(folder, item.path)), item);
      if (draftInput && this.pending?.id === draftInput.id && this.options.accepted) {
        const origin = this.options.origin(), token = this.options.token();
        const replacements = new Map(artifacts.map(item => [item.path, item.after_digest]));
        const accepted = [...documents].map(([file, snapshot]) => ({ document: snapshot.document,
          version: snapshot.document.version, digest: replacements.get(file) ?? snapshot.digest }));
        this.pending.accepted = { target: draftInput.target, file: artifacts[0].path,
          ...(artifacts.length > 1 ? { files: artifacts.map(item => item.path) } : {}),
          current: () => !signal.aborted && vscode.workspace.isTrusted && this.options.origin() === origin && this.options.token() === token
            && accepted.every(item => !item.document.isClosed && item.document.version === item.version && sha256(item.document.getText()) === item.digest) };
      }
      return this.outcome('applied',
        `Applied to ${artifacts.length === 1 ? 'the VS Code buffer' : `all ${artifacts.length} unsaved VS Code buffers in one edit`}. The graph refreshes from source; use VS Code Undo to revert.`);
    } catch {
      return refuse('The editor did not confirm the source change. Inspect the buffer before retrying; use VS Code Undo if needed.');
    } finally {
      if (!approved || applied) draft?.dispose();
      await candidate?.finish(); changed?.dispose();
    }
  }
}
