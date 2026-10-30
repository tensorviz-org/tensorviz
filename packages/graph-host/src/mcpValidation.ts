import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { GraphToolError, McpGraphService, graphToolResult, modelRefSchema, type McpModel, type ModelRef } from './mcpGraph';
import type { LocalBrowserHostPublish } from './localBrowserHostTransport';
import type { LocalBrowserValidationPhase } from './validationSummary';
import type { LocalBrowserTargetIdentity, LocalBrowserValidationOutcome, ProjectionPayload } from '@tensorviz/contracts/protocol';
import { tensorInputsSchema, type TensorInputs } from './tensorInputs';
import { executionSpecSchema, executionJobSchema, type ExecutionJob, type ExecutionSpec } from './executionSpec';
import type { TraceOutcome } from './traceWorkerCore';
import { compareRetainedRuns } from './mcpRunComparison';
import { bytes, captureRunEvidence, clone, freezeGraph, inspectEvidence, MAX_RETAINED_BYTES, MAX_RUN_BYTES } from './mcpRunEvidence';
import { ValidationSummaryStore, validationPhaseDetails } from './validationSummary';
import type { AttemptHandle } from './attemptLedger';
import { captureQuerySchema } from './captureInspection';
import type { CaptureSnapshot } from './semanticCapture';
import { captureSummary, type CaptureSelection } from './mcpSemanticEvidence';

type RunStatus = 'queued' | 'running' | 'cancelling' | LocalBrowserValidationOutcome['status'];
interface Run {
  runId: string; owner: string; ref: ModelRef; target: LocalBrowserTargetIdentity; inputs?: TensorInputs;
  execution: ExecutionSpec; job?: ExecutionJob; graph: Record<string, unknown>; evidence?: Record<string, unknown>;
  retainedBytes: number; evidenceUnavailable?: string;
  captureReceived?: boolean;
  status: RunStatus; phase: LocalBrowserValidationPhase; detail: string;
  startedAt: number; finishedAt?: number; result?: ModelRef; controller: AbortController;
  attempt?: AttemptHandle;
}
const active = (run: Run) => run.finishedAt === undefined;
const sameSource = (a: LocalBrowserTargetIdentity, b: LocalBrowserTargetIdentity) =>
  a.workspaceFolder === b.workspaceFolder && a.file === b.file && a.cls === b.cls &&
  a.documentVersion === b.documentVersion && a.sourceDigest === b.sourceDigest;

/** One worker per window, bounded retained results, and no cross-client cancellation. */
export class McpValidationRuns {
  private readonly runs = new Map<string, Run>();
  get running(): boolean { return [...this.runs.values()].some(active); }
  constructor(private readonly read: () => McpModel | undefined,
    private readonly validate: (target: LocalBrowserTargetIdentity, signal: AbortSignal, inputs: TensorInputs | undefined,
      progress: (phase: LocalBrowserValidationPhase) => void,
      measured: (target: LocalBrowserTargetIdentity, projection: ProjectionPayload) => void,
      captured: (outcome: TraceOutcome) => void, execution: ExecutionSpec, job?: ExecutionJob,
      attempt?: AttemptHandle) => Promise<LocalBrowserValidationOutcome>,
    private readonly publish: (message: LocalBrowserHostPublish) => unknown,
    private readonly summaries?: ValidationSummaryStore) {}

  start(owner: string, graph: McpGraphService, ref: ModelRef, inputs?: TensorInputs, execution?: ExecutionSpec, job?: ExecutionJob): Record<string, unknown> {
    const { model } = graph.require(ref);
    if ((inputs === undefined) === (execution === undefined)) throw new GraphToolError('INVALID_SETUP', 'Supply either inputs or an explicit execution setup.');
    const parsed = inputs === undefined ? undefined : tensorInputsSchema.parse(inputs);
    const setup = executionSpecSchema.parse(execution ?? { inputs: { args: parsed } });
    const diagnosticJob = job ? executionJobSchema.parse(job) : undefined;
    const snapshot = freezeGraph(graph, model);
    if (bytes(snapshot) > MAX_RUN_BYTES) throw new GraphToolError('GRAPH_TOO_LARGE', 'This graph exceeds the retained run evidence limit.');
    if ([...this.runs.values()].some(active)) throw new GraphToolError('BUSY', 'Another MCP validation is running. Wait or cancel your run first.');
    for (const [id, run] of this.runs) {
      if (run.finishedAt !== undefined && (Date.now() - run.finishedAt > 30 * 60_000 || this.runs.size >= 32)) this.runs.delete(id);
    }
    const run: Run = { runId: randomUUID(), owner, ref: { modelId: ref.modelId, revision: ref.revision },
      target: clone(model.target), inputs: parsed, execution: setup, job: diagnosticJob, graph: snapshot, retainedBytes: bytes(snapshot),
      status: 'queued', phase: 'checking', detail: 'Checking the saved model and selected environment.',
      startedAt: Date.now(), controller: new AbortController() };
    this.runs.set(run.runId, run);
    this.prune();
    void this.execute(run, graph);
    return this.snapshot(run);
  }

  get(owner: string, runId: string): Record<string, unknown> { return this.snapshot(this.require(owner, runId)); }
  inspect(owner: string, runId: string, section = 'summary', offset = 0, limit = 50,
    selection: CaptureSelection = { path: [] }): Record<string, unknown> {
    const run = this.require(owner, runId);
    return clone({ ...this.snapshot(run), format: run.evidence?.capture ? 'semantic-v1' : run.evidence?.execution ? 'legacy' : 'unavailable',
      ...inspectEvidence({ ...run.graph, execution: run.execution, job: run.job }, run.evidence, section, offset, limit, selection),
      immutable: !active(run), evidenceAvailable: !!run.evidence, evidenceUnavailable: run.evidenceUnavailable ?? null });
  }
  compare(owner: string, baselineRunId: string, candidateRunId: string, section = 'summary', offset = 0, limit = 50,
    atol = 1e-6, rtol = 1e-5): Record<string, unknown> {
    const baseline = this.require(owner, baselineRunId), candidate = this.require(owner, candidateRunId);
    if (active(baseline) || active(candidate)) throw new GraphToolError('RUN_IN_PROGRESS', 'Wait for both runs to finish before comparing their evidence.');
    return clone(compareRetainedRuns(baseline, candidate, section, offset, limit, atol, rtol));
  }
  private prune(): void {
    let total = [...this.runs.values()].reduce((sum, run) => sum + run.retainedBytes, 0);
    for (const [id, run] of this.runs) {
      if (total <= MAX_RETAINED_BYTES) break;
      if (!active(run)) { total -= run.retainedBytes; this.runs.delete(id); }
    }
  }
  cancel(owner: string, runId: string): Record<string, unknown> {
    const run = this.require(owner, runId);
    if (active(run)) { run.status = 'cancelling'; run.controller.abort(); }
    return this.snapshot(run);
  }
  cancelOwner(owner: string): void {
    for (const [id, run] of this.runs) if (run.owner === owner) { run.controller.abort(); this.runs.delete(id); }
  }
  clear(): void {
    for (const run of this.runs.values()) run.controller.abort();
    this.runs.clear();
  }
  private require(owner: string, runId: string): Run {
    const run = this.runs.get(runId);
    if (!run || run.owner !== owner || (run.finishedAt !== undefined && Date.now() - run.finishedAt > 30 * 60_000)) {
      if (run?.owner === owner) this.runs.delete(runId);
      throw new GraphToolError('UNKNOWN_RUN', 'This run is unavailable in the current MCP session.');
    }
    return run;
  }
  private snapshot(run: Run): Record<string, unknown> {
    return clone({ runId: run.runId, ...run.ref, target: run.target, inputs: run.inputs, execution: run.execution, job: run.job ?? null, status: run.status,
      phase: run.phase, detail: run.detail, startedAt: new Date(run.startedAt).toISOString(),
      attempt: run.attempt?.snapshot() ?? null,
      capture: run.evidence?.capture ? captureSummary(run.evidence.capture as CaptureSnapshot) : null,
      captureUnavailable: run.evidence?.capture ? null : run.evidence?.captureUnavailable ?? run.evidenceUnavailable ?? 'not_captured',
      finishedAt: run.finishedAt === undefined ? null : new Date(run.finishedAt).toISOString(), result: run.result ?? null, evidenceAvailable: !!run.evidence, evidenceUnavailable: run.evidenceUnavailable ?? null });
  }
  private async execute(run: Run, graph: McpGraphService): Promise<void> {
    const observe = this.summaries?.begin(run.runId, run.target, 'mcp', { execution: run.execution, job: run.job });
    run.attempt = observe?.attempt;
    // Return the id before starting asynchronous trust, environment and source checks.
    await Promise.resolve();
    const progress = (phase: LocalBrowserValidationPhase) => {
      if (run.controller.signal.aborted) return;
      run.phase = phase; run.status = 'running';
      run.detail = validationPhaseDetails[phase];
      if (observe) observe('running', run.detail);
      else this.publish({ type: 'validationState', runId: run.runId, target: run.target, status: 'running', detail: run.detail });
    };
    let outcome: LocalBrowserValidationOutcome;
    let produced: McpModel | undefined;
    try {
      if (run.controller.signal.aborted) outcome = { status: 'cancelled', detail: 'Validation cancelled.' };
      else { progress('checking'); outcome = await this.validate(run.target, run.controller.signal, run.inputs, progress,
        (target, projection) => {
          if (!active(run)) return;
          produced = clone({ target, projection, current: true });
          const snapshot = freezeGraph(graph, produced, run.evidence?.capture as CaptureSnapshot | undefined);
          if (bytes(snapshot) + bytes(run.evidence ?? {}) <= MAX_RUN_BYTES) run.graph = snapshot;
          else run.evidenceUnavailable = 'The measured graph exceeded the retention limit; the original source graph is retained.';
          run.retainedBytes = bytes(run.evidence ?? {}) + bytes(run.graph);
          this.prune();
        }, captured => {
          if (!active(run) || run.captureReceived) return;
          run.captureReceived = true;
          const id = run.attempt?.snapshot().captureIds[0];
          const retained = id ? this.summaries?.ledger.capture(id) : undefined;
          const evidence = captureRunEvidence(captured, run.target.workspaceFolder, retained,
            { sourceDigest: run.target.sourceDigest, checkpoint: run.execution?.checkpoint });
          if (bytes(evidence) + bytes(run.graph) <= MAX_RUN_BYTES) run.evidence = evidence;
          else run.evidenceUnavailable = 'Worker evidence exceeded the 16 MiB per-run retention limit.';
          run.retainedBytes = bytes(run.evidence ?? {}) + bytes(run.graph);
          this.prune();
        }, run.execution, run.job, observe?.attempt); }
    } catch { outcome = { status: 'refused', detail: 'The host could not complete validation. Inspect the retained run for any worker evidence and retry with a current saved model.' }; }
    if (run.controller.signal.aborted) outcome = { status: 'cancelled', detail: 'Validation cancelled or the connection ended.' };
    const model = this.read();
    const measured = !run.controller.signal.aborted && (outcome.status === 'passed' || outcome.status === 'failed') &&
      model?.current && produced?.projection.traced && sameSource(model.target, run.target) &&
      graph.reference(model).revision === graph.reference(produced).revision;
    if (measured) run.result = graph.reference(model);
    else if (outcome.status === 'passed') outcome = { status: 'refused', detail: 'The model changed before the validation result could be confirmed. Refresh discovery and retry.' };
    if (run.evidence && outcome.detail.includes('tensorViz Output')) {
      outcome = { ...outcome, detail: 'Execution completed with a failure or refusal. Use inspect_run for retained operation, shape, source and setup evidence.' };
    }
    run.status = outcome.status; run.detail = outcome.detail; run.finishedAt = Date.now();
    run.retainedBytes = bytes(run.evidence ?? {}) + bytes(run.graph);
    this.prune();
    // The source checks in validate are authoritative. This final receipt names
    // only a projection produced by this run, never a previous trace on that model.
    if (observe) observe(outcome.status, outcome.detail, measured ? { target: model.target, revision: model.projection.revision } : undefined);
    else this.publish({ type: 'validationState', runId: run.runId, target: measured ? model.target : run.target,
      ...outcome, ...(measured ? { projectionRevision: model.projection.revision } : {}) });
  }
}

export function registerMcpValidationTools(server: McpServer, graph: McpGraphService, runs: McpValidationRuns, owner: string,
  options: { diagnostics?: boolean; progress?(): Record<string, unknown> } = {}): void {
  server.registerTool('validate_model', {
    description: 'Run a saved model using positional tensors or explicit execution setup (fixture/keyword inputs, model factory/checkpoint, mode and device) in the selected Python environment. Executes trusted workspace code; may have model-defined side effects. Returns a run id immediately. Poll get_run; a pass covers only these inputs.',
    inputSchema: { ...modelRefSchema, inputs: tensorInputsSchema.optional(), execution: executionSpecSchema.optional() },
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
  }, (args, extra) => graph.toolResult('validate_model', () => {
    if (extra.signal.aborted) throw new GraphToolError('CANCELLED', 'The request was cancelled.');
    return { ...runs.start(owner, graph, args, args.inputs, args.execution), ...options.progress?.() };
  }));
  server.registerTool('get_run', { description: 'Read your validation progress and result. Completed results remain readable after the graph revision changes, for up to 30 minutes (32 retained runs per window).',
    inputSchema: { runId: z.string().uuid() }, annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  }, args => graphToolResult(() => ({ ...runs.get(owner, args.runId), ...options.progress?.() })));
  server.registerTool('inspect_run', {
    description: 'Inspect immutable evidence from your run after the live graph or retained capture slots change. Canonical operations/modules return JSON-tree pages; section capture accepts entity and path for exact ordered arguments, values, effects and coverage using the same reader as the browser. Failure follows recorded value producers. Legacy runs label missing semantics. Never executes Python. Runs expire after 30 minutes and remain bounded by window retention.',
    inputSchema: { runId: z.string().uuid(), section: z.enum(['summary', 'graph', 'setup', 'outputs', 'checkpoint', 'capture', 'failure', 'operations', 'modules']).default('summary'),
      entity: captureQuerySchema.shape.entity, path: captureQuerySchema.shape.path,
      offset: captureQuerySchema.shape.offset, limit: captureQuerySchema.shape.limit },
    annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  }, args => graph.toolResult('inspect_run', () => runs.inspect(owner, args.runId, args.section, args.offset, args.limit,
    { path: args.path, ...(args.entity ? { entity: args.entity } : {}) }), args.offset === 0));
  server.registerTool('compare_runs', {
    description: 'Compare two of your retained completed execution runs without executing code. Requires runId UUIDs returned by your execution tools, not capture IDs or graph IDs. This returns data; to display local source changes on Canvas use open_graph_diff instead. Canonical captures compare exact arguments, effects, state/sharing and value-use connections using conservative role correspondence; inspect unresolved records with section correspondence or inspect_run. Shapes, mode, coverage and input/weight/environment/RNG controls remain separate. Legacy runs use the labelled layer comparer. Integer outputs compare exactly; floating/complex outputs use tolerances relative to the baseline. Missing evidence cannot prove equivalence.',
    inputSchema: { baselineRunId: z.string().uuid(), candidateRunId: z.string().uuid(),
      section: z.enum(['summary', 'setup', 'nodes', 'connections', 'coverage', 'outputs', 'observations', 'correspondence']).default('summary'),
      offset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(200).default(50),
      atol: z.number().finite().min(0).default(1e-6), rtol: z.number().finite().min(0).default(1e-5) },
    annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  }, args => graph.toolResult('compare_runs', () => runs.compare(owner, args.baselineRunId, args.candidateRunId, args.section,
    args.offset, args.limit, args.atol, args.rtol), args.offset === 0));
  server.registerTool('cancel_run', { description: 'Cancel your queued or running validation. Poll get_run for the terminal state. Does not cancel other clients or manual browser checks.',
    inputSchema: { runId: z.string().uuid() }, annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, args => graph.toolResult('cancel_run', () => runs.cancel(owner, args.runId)));
}
