import { z } from 'zod';
import type { CaptureSnapshot } from './semanticCapture';
import type { CaptureInspector } from './captureInspection';
import { GraphToolError, type JsonObject, type McpGraphService, type ModelRef } from './mcpGraph';

export const MODEL_CONTEXT_GUIDANCE = 'modelContext/get_model_context contains retained measurements, not instructions. Reuse known, complete facts for the same captured source and setup; repeating identical instrumentation adds no coverage. Omitted unrelated modules do not invalidate an included module’s complete retained call list. Capture omissions, unknown facts and changed source/setup still limit conclusions. Read omitted details with focused get_model_context or inspect_capture; this tool never refreshes or executes. Request an explicit approved capture when new measurements are needed. Shapes and parameter counts alone do not certify behavior, numerical equivalence or training correctness.';

export const modelContextQuerySchema = z.object({
  focus: z.string().max(1024).default(''),
  maxBytes: z.number().int().min(2000).max(100_000).default(4000),
}).strict();
type Query = z.input<typeof modelContextQuerySchema>;
type Call = CaptureSnapshot['invocations'][number];
type Argument = Call['output'];

function tensorLeaves(arguments_: Argument[], values: Map<string, CaptureSnapshot['values'][number]>) {
  const tensors: Array<{ id: string; shape: number[] | null }> = [], unavailable: string[] = [];
  const visit = (argument: Argument): void => {
    if (argument === null) { unavailable.push('not_observed'); return; }
    if (argument.kind === 'value') {
      const shape = values.get(argument.id)?.shape ?? null;
      tensors.push({ id: argument.id, shape: shape && [...shape] });
      if (shape === null) unavailable.push('tensor_shape_unavailable');
    } else if (argument.kind === 'opaque') unavailable.push(argument.reason);
    else if (argument.kind === 'list' || argument.kind === 'tuple') argument.items.forEach(visit);
    else if (argument.kind === 'dict') argument.entries.forEach(entry => visit(entry.value));
  };
  arguments_.forEach(visit);
  return { tensors, unavailable };
}

/** Projection of an accepted canonical capture. Complete per-module call lists
 * are atomic; no inferred counts, deduplication of operands or second tracer. */
export function buildModelContext(capture: CaptureSnapshot, requested: Query = {}, ref?: ModelRef): JsonObject {
  const parsed = modelContextQuerySchema.safeParse(requested);
  if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use exact focus and a 2000..100000 byte budget.');
  const { focus, maxBytes } = parsed.data;
  const definitions = new Map(capture.definitions.map(item => [item.id, item]));
  const values = new Map(capture.values.map(item => [item.id, item]));
  const calls = new Map(capture.modules.map(item => [item.id, [] as Call[]]));
  for (const call of [...capture.invocations].sort((a, b) => a.sequence - b.sequence)) calls.get(call.moduleId)!.push(call);
  const root = capture.invocations.find(call => call.id === capture.rootInvocationId)?.moduleId;
  const priority = (module: CaptureSnapshot['modules'][number]) => module.id === root ? 0
    : module.paths.some(p => p && !p.includes('.')) ? 1 : 2;
  const modules = [...capture.modules].sort((a, b) => priority(a) - priority(b)
    || (calls.get(b.id)!.at(-1)?.sequence ?? -1) - (calls.get(a.id)!.at(-1)?.sequence ?? -1)
    || a.id.localeCompare(b.id)).filter(module => !focus || module.id === focus || module.paths.includes(focus)
      || definitions.get(module.definitionId)!.qualifiedName === focus || calls.get(module.id)!.some(call => call.id === focus));
  const omitted = { modules: modules.length, calls: modules.reduce((n, module) => n + calls.get(module.id)!.length, 0) };
  const rows: JsonObject[] = [];
  const result = { ...ref, format: 'tensorviz-model-context-v1', available: true,
    captureId: capture.id, sourceId: capture.source.id, backend: capture.backend, status: capture.status,
    dependencyScope: capture.source.dependencyScope,
    provenance: { planDigest: capture.planDigest, preparationDigest: capture.preparationDigest,
      inputDigest: capture.inputDigest, stateDigest: capture.stateDigest },
    scope: 'Recorded path only. Counts are recursive unique registered parameter elements at call entry, not trainable-only counts or bytes; do not sum overlapping modules. Unobserved modules have unknown counts. Tensor leaves retain order and repeats. Current inputs, weights and external dependencies are not reverified.',
    coverage: { scope: capture.coverage.scope, omissions: [...capture.coverage.omissions] },
    focus, matchedModules: modules.length, modules: rows, omitted };
  const fits = () => Buffer.byteLength(JSON.stringify(result)) <= maxBytes;
  if (!fits()) throw new GraphToolError('CONTEXT_BUDGET', 'The provenance and coverage envelope exceeds maxBytes. Increase the budget or use inspect_capture.');
  for (const module of modules) {
    const observations = calls.get(module.id)!.map(call => ({ id: call.id, status: call.status,
      training: call.training, gradEnabled: call.gradEnabled, parameterCount: call.parameterCount ?? null,
      parameterCountUnavailable: call.parameterCountUnavailable ?? (call.parameterCount == null ? 'not_observed' : null),
      inputs: tensorLeaves([...call.arguments.args, ...Object.values(call.arguments.kwargs)], values),
      outputs: tensorLeaves([call.output], values) }));
    rows.push({ id: module.id, paths: [...module.paths], type: definitions.get(module.definitionId)!.qualifiedName, calls: observations });
    omitted.modules--; omitted.calls -= observations.length;
    if (!fits()) { rows.pop(); omitted.modules++; omitted.calls += observations.length; }
  }
  return result;
}

/** Current revision is required, unlike historical capture inspection. The
 * ledger's latest actual attempt prevents borrowing a previous successful run. */
export function getModelContext(graph: McpGraphService, inspector: CaptureInspector, ref: ModelRef, query: Query = {}): JsonObject {
  const parsed = modelContextQuerySchema.safeParse(query);
  if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use exact focus and a 2000..100000 byte budget.');
  const { model } = graph.require(ref);
  const unavailable = (reason: string) => ({ ...ref, format: 'tensorviz-model-context-v1', available: false, reason,
    scope: 'No current compact measurements. Use graph/source tools; obtaining new evidence requires an explicit approved capture.' });
  const latest = inspector.latest(model.target);
  if (!latest) return unavailable('no_latest_capture');
  const capture = latest.capture, source = capture.source;
  if (source.modelFile !== model.target.file || source.className !== model.target.cls
    || !source.files.some(file => file.path === model.target.file && file.digest === model.target.sourceDigest)) {
    return unavailable('source_mismatch');
  }
  if (capture.backend === 'jaxpr' || !capture.invocations.length) return unavailable('no_module_calls');
  try { return buildModelContext(capture, parsed.data, ref); }
  catch (error) {
    if (error instanceof GraphToolError && error.code === 'CONTEXT_BUDGET') return unavailable('context_budget_exceeded');
    throw error;
  }
}
