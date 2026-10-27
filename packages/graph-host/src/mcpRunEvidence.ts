import * as path from 'node:path';
import * as fs from 'node:fs';
import { McpGraphService, object, type JsonObject, type McpModel } from './mcpGraph';
import type { TraceOutcome } from './traceWorkerCore';
import type { AttemptLedger } from './attemptLedger';
import type { CaptureSnapshot } from './semanticCapture';
import { inspectSemanticEvidence, type CaptureSelection } from './mcpSemanticEvidence';

export const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
export const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value));
export const MAX_RUN_BYTES = 16 * 1024 * 1024;
export const MAX_RETAINED_BYTES = 64 * 1024 * 1024;

export function freezeGraph(graph: McpGraphService, model: McpModel, capture?: CaptureSnapshot): JsonObject {
  const semantic = object(object(model.projection.document).semantic);
  const nodes = Array.isArray(semantic.nodes) ? semantic.nodes.map(object) : [];
  const edges = Array.isArray(semantic.edges) ? semantic.edges.map(object) : [];
  let canonicalWorkspace = model.target.workspaceFolder;
  try { canonicalWorkspace = fs.realpathSync(canonicalWorkspace); } catch { /* Synthetic/test or removed workspace. */ }
  const identity = object(object(semantic.metadata).capture);
  const calls = new Map(capture && capture.id === identity.id ? capture.invocations.map(call => [call.id, call]) : []);
  const modules = new Map(capture?.modules.map(module => [module.id, module]));
  const definitions = new Map(capture?.definitions.map(definition => [definition.id, definition]));
  return clone({ ...graph.reference(model), target: model.target, canonicalWorkspace, traced: model.projection.traced,
    capture: Object.keys(identity).length ? identity : null,
    truncated: model.projection.truncated ?? false,
    nodes: nodes.map(node => {
      const ref = object(object(node.metadata).captureRef);
      const call = ref.kind === 'invocation' && ref.captureId === capture?.id && ref.sourceId === capture?.source.id
        ? calls.get(String(ref.semanticId)) : undefined;
      const module = call && modules.get(call.moduleId);
      const definition = module && definitions.get(module.definitionId);
      return { ...graph.node(node, model.projection), modulePath: object(node.params)._module_path ?? null,
        moduleClass: object(node.params)._class ?? node.op, definition: object(node.params)._import_path ?? null,
        repeatCount: object(node.params)._repeat_count ?? 1,
        ...(module ? { modulePaths: module.paths, modulePath: module.paths.length === 1 ? module.paths[0] : null,
          moduleId: module.id, invocationId: call!.id, moduleClass: definition?.qualifiedName ?? node.op } : {}) };
    }),
    edges: edges.map(edge => graph.edge(edge)) });
}

/** Keep structured facts, not arbitrary Python exception text, stdout or source bodies. */
export function captureRunEvidence(outcome: TraceOutcome, workspace: string,
  retained?: ReturnType<AttemptLedger['capture']>): JsonObject {
  const envelope = outcome.envelope;
  const diagnostics = object(object(envelope?.result).diagnostics);
  const error = envelope?.error;
  let root = workspace;
  try { root = fs.realpathSync(workspace); } catch { /* Synthetic/test or removed workspace. */ }
  const frames = error?.frames?.filter(frame => {
    const relative = path.relative(root, frame.file);
    return !path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`);
  }).map(frame => ({ file: frame.file, line: frame.line, function: frame.name })) ?? [];
  const { importedSources, ...worker } = (envelope?.worker ?? {}) as JsonObject;
  const accepted = retained?.record.format === 'semantic' && !retained.record.unavailable
    ? object(object(retained.envelope?.result).diagnostics).semantic_capture : undefined;
  const captureUnavailable = accepted ? null : retained?.record.semanticRejection ?? retained?.record.unavailable
    ?? (diagnostics.semantic_capture != null ? 'not_validated' : 'not_captured');
  return clone({ setup: diagnostics.execution ?? (importedSources ? { importedSources } : null), execution: diagnostics.run_evidence ?? null,
    capture: accepted ?? null, captureUnavailable,
    worker: envelope?.worker ? worker : null, exitCode: outcome.exitCode ?? null,
    error: outcome.ok ? null : {
      code: outcome.cancelled ? 'CANCELLED' : outcome.timedOut ? 'TIMEOUT'
        : outcome.reason === 'The trace worker exceeded its output limit.' ? 'OUTPUT_LIMIT' : error ? 'PYTHON_ERROR' : 'WORKER_FAILED',
      type: error?.type ?? null, causeTypes: error?.cause_types ?? [], missingModule: error?.missing_module ?? null,
      // Fixed-vocabulary facts validated against Python's semantic contract.
      diagnostic: error?.diagnostic ?? null,
      frames, nodeId: error?.node_id ?? null,
      stage: outcome.exitCode === 3 ? 'setup' : outcome.exitCode === 4 ? 'forward' : 'worker',
    } });
}

function records(value: unknown): JsonObject[] { return Array.isArray(value) ? value.map(object) : []; }
function page(items: JsonObject[], offset: number, limit: number): JsonObject {
  const selected: JsonObject[] = [];
  let size = 0;
  for (const [index, item] of items.slice(offset, offset + limit).entries()) {
    const itemBytes = bytes(item);
    const bounded = itemBytes > 64 * 1024 ? { unavailable: 'record_size_limit', index: offset + index, serializedBytes: itemBytes } : item;
    const nextSize = bytes(bounded);
    if (size + nextSize > 64 * 1024) break;
    selected.push(bounded); size += nextSize;
  }
  return { items: selected, count: items.length,
    nextOffset: offset + selected.length < items.length ? offset + selected.length : null };
}

function treeEntries(value: unknown, prefix = ''): JsonObject[] {
  const tree = object(value);
  if (tree.kind === 'dict') return Object.entries(object(tree.items)).flatMap(([key, item]) => treeEntries(item, `${prefix}[${JSON.stringify(key)}]`));
  if (tree.kind === 'list' || tree.kind === 'tuple') return records(tree.items).flatMap((item, index) => treeEntries(item, `${prefix}[${index}]`));
  return Object.keys(tree).length ? [{ path: prefix, ...tree }] : [];
}

export function inspectEvidence(snapshot: JsonObject, evidence: JsonObject | undefined,
  section: string, offset: number, limit: number, selection: CaptureSelection = { path: [] }): JsonObject {
  const execution = object(evidence?.execution);
  const semantic = inspectSemanticEvidence(evidence?.capture as CaptureSnapshot | undefined, section, offset, limit,
    selection, typeof evidence?.captureUnavailable === 'string' ? evidence.captureUnavailable : undefined);
  if (semantic) return { ...semantic, error: evidence?.error ?? null,
    ...(section === 'summary' ? { graphNodeCount: records(snapshot.nodes).length, graphEdgeCount: records(snapshot.edges).length,
      executionMode: object(evidence?.setup).mode ?? null, environment: object(evidence?.setup).environment ?? evidence?.worker ?? null,
      sections: ['graph', 'setup', 'capture', 'failure', 'operations', 'modules'] } : {}) };
  if (section === 'graph') {
    return { snapshot: { modelId: snapshot.modelId, revision: snapshot.revision, target: snapshot.target,
      traced: snapshot.traced, truncated: snapshot.truncated, capture: snapshot.capture ?? null }, nodes: page(records(snapshot.nodes), offset, limit),
      edges: page(records(snapshot.edges), offset, limit) };
  }
  if (section === 'setup') {
    if (!evidence?.setup) return { setup: null, worker: evidence?.worker ?? null };
    const setup = object(evidence.setup), inputs = object(setup.inputs), state = object(setup.state);
    const plan = object(setup.resolvedPlan), source = object(plan.source);
    const recipe = object(setup.setup), checkpoint = object(recipe.checkpoint);
    const keys = (kind: string) => (Array.isArray(checkpoint[kind]) ? checkpoint[kind] as unknown[] : []).map(key => ({ kind, key }));
    return { setup: { ...setup, moduleModes: undefined, importedSources: undefined,
      resolvedPlan: setup.resolvedPlan ? { ...plan, source: { ...source, files: undefined } } : undefined,
      setup: { ...recipe, checkpoint: recipe.checkpoint ? { ...checkpoint, missingKeys: undefined, unexpectedKeys: undefined,
        missingKeyCount: keys('missingKeys').length, unexpectedKeyCount: keys('unexpectedKeys').length } : null },
      inputs: { ...inputs, description: undefined }, state: { ...state, parametersAndBuffers: undefined } }, worker: evidence.worker ?? null,
      inputEntries: page(treeEntries(inputs.description), offset, limit),
      stateEntries: page(treeEntries(state.parametersAndBuffers), offset, limit),
      moduleModes: page(Object.entries(object(setup.moduleModes)).map(([modulePath, training]) => ({ modulePath, training })), offset, limit),
      importedSources: page(records(setup.importedSources), offset, limit),
      planFiles: page(records(source.files), offset, limit),
      checkpointKeys: page([...keys('missingKeys'), ...keys('unexpectedKeys')], offset, limit) };
  }
  if (section === 'operations') return { ...page(records(execution.operations), offset, limit), truncated: execution.truncated ?? [] };
  if (section === 'modules') return { calls: page(records(execution.moduleCalls), offset, limit),
    coverage: page(records(execution.coverage), offset, limit), truncated: execution.truncated ?? [] };
  if (section === 'failure') {
    const failure = object(execution.failure);
    const operations = records(execution.operations);
    const byId = new Map(operations.map(operation => [operation.operationId, operation]));
    const upstream = new Map<unknown, JsonObject>();
    const missing = new Set<unknown>();
    const queue = records(failure.inputs).map(input => input.producerOperationId).filter(id => id !== undefined);
    while (queue.length) {
      const id = queue.pop();
      if (upstream.has(id) || missing.has(id)) continue;
      const operation = byId.get(id);
      if (!operation) { missing.add(id); continue; }
      upstream.set(id, operation);
      queue.push(...records(operation.inputs).map(input => input.producerOperationId).filter(parent => parent !== undefined));
    }
    return { error: evidence?.error ?? null, failure: execution.failure ?? null, lastObservedFailure: execution.lastObservedFailure ?? null,
      upstream: page([...upstream.values()].sort((a, b) => Number(a.index) - Number(b.index)), offset, limit),
      missingUpstreamOperationCount: missing.size, truncated: execution.truncated ?? [],
      detail: execution.failure ? 'Recorded tensor inputs and their producer operations; unknown or truncated paths are not inferred.'
        : 'No failing tensor operation was captured. Use the recorded exception frames and module coverage; the last completed operation is not necessarily the failure.' };
  }
  const setup = object(evidence?.setup);
  return { format: Object.keys(execution).length ? 'legacy' : 'unavailable', captureUnavailable: evidence?.captureUnavailable ?? 'not_captured',
    error: evidence?.error ?? null, failure: execution.failure ?? null, lastObservedFailure: execution.lastObservedFailure ?? null,
    executionMode: setup.mode ?? null, environment: setup.environment ?? evidence?.worker ?? null,
    operationCount: execution.operationCount ?? null, moduleCallCount: execution.moduleCallCount ?? null,
    graphNodeCount: records(snapshot.nodes).length, graphEdgeCount: records(snapshot.edges).length,
    reachedModuleCount: Array.isArray(execution.coverage) ? records(execution.coverage).filter(module => module.reached).length : null,
    unreachedModuleCount: Array.isArray(execution.coverage) ? records(execution.coverage).filter(module => !module.reached).length : null,
    truncated: execution.truncated ?? [], sections: ['graph', 'setup', 'failure', 'operations', 'modules'] };
}
