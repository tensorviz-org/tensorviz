/** Read-only views of the already validated capture retained by an MCP run. */
import { inspectCapture } from './captureInspection';
import { GraphToolError, type JsonObject } from './mcpGraph';
import type { CaptureSnapshot } from './semanticCapture';
import type { CaptureInspectionQuery } from '@tensorviz/contracts/protocol';

export type CaptureSelection = Pick<CaptureInspectionQuery, 'entity' | 'path'>;

export function captureSummary(capture: CaptureSnapshot): JsonObject {
  return { id: capture.id, sourceId: capture.source.id, status: capture.status,
    rootInvocationId: capture.rootInvocationId, operationCount: capture.operations.length,
    invocationCount: capture.invocations.length, valueCount: capture.values.length,
    omissions: capture.coverage.omissions };
}

function valueIds(argument: CaptureSnapshot['output']): string[] {
  if (argument === null) return [];
  if (argument.kind === 'value') return [argument.id];
  if (argument.kind === 'list' || argument.kind === 'tuple') return argument.items.flatMap(valueIds);
  if (argument.kind === 'dict') return argument.entries.flatMap(entry => valueIds(entry.value));
  return [];
}

export function inspectSemanticEvidence(capture: CaptureSnapshot | undefined, section: string,
  offset: number, limit: number, selection: CaptureSelection, unavailable?: string): JsonObject | undefined {
  if (section !== 'capture' && (selection.entity || selection.path.length)) {
    throw new GraphToolError('INVALID_QUERY', 'Use section capture to select an entity or nested path.');
  }
  if (section === 'capture' && !capture) {
    throw new GraphToolError('CAPTURE_UNAVAILABLE', unavailable
      ? `Canonical capture is unavailable: ${unavailable}.` : 'This run has no canonical capture.');
  }
  if (!capture && unavailable && unavailable !== 'not_captured'
    && ['operations', 'modules', 'failure', 'summary'].includes(section)) {
    return { format: 'unavailable', capture: null, captureUnavailable: unavailable,
      detail: 'This run did not retain an accepted canonical capture. Older recorder summaries cannot establish its missing semantics.' };
  }
  if (!capture || !['capture', 'operations', 'modules', 'failure', 'summary'].includes(section)) return undefined;
  const query = { captureId: capture.id, sourceId: capture.source.id, offset, limit };
  const read = (path: Array<string | number>) => inspectCapture(capture, { ...query, path });
  const base = { format: 'semantic-v1', capture: captureSummary(capture) };
  if (section === 'capture') return { ...base, page: inspectCapture(capture, { ...query, ...selection }) };
  // Child descriptors are exact JSON-tree pages, not lossy first-tensor summaries.
  // Their keys are absolute indices to append to the returned path.
  if (section === 'operations') return { ...base, operations: read(['operations']) };
  if (section === 'modules') return { ...base, calls: read(['invocations']), modules: read(['modules']), coverage: read(['coverage', 'modules']) };
  const root = capture.invocations.find(call => call.id === capture.rootInvocationId);
  const failed = capture.operations.filter(op => op.status === 'failed' && root?.diagnosticId != null && op.diagnosticId === root.diagnosticId);
  if (section === 'failure') {
    const byId = new Map(capture.operations.map(op => [op.id, op]));
    const byValue = new Map(capture.values.map(value => [value.id, value]));
    const terminal = new Set(failed.map(op => op.id)), seen = new Set<string>();
    const queue = failed.map(op => op.id);
    while (queue.length) {
      const id = queue.pop()!;
      if (seen.has(id)) continue;
      seen.add(id);
      const op = byId.get(id)!;
      const inputs = [...op.arguments.args, ...Object.values(op.arguments.kwargs)].flatMap(valueIds);
      for (const valueId of [...inputs, ...op.effects.map(effect => effect.beforeId)]) {
        const producer = byValue.get(valueId)?.producer?.operationId;
        if (producer && byId.has(producer) && !seen.has(producer)) queue.push(producer);
      }
    }
    const upstream = capture.operations.filter(op => seen.has(op.id) && !terminal.has(op.id));
    const references = (records: Array<{ id: string; sequence: number; diagnosticId: string | null }>, kind: 'operation' | 'invocation') => ({
      items: records.slice(offset, offset + limit).map(record => ({ id: record.id, sequence: record.sequence,
        diagnosticId: record.diagnosticId, entity: { kind, id: record.id } })), count: records.length,
      nextOffset: offset + limit < records.length ? offset + limit : null,
    });
    return { ...base, terminalDiagnosticId: root?.diagnosticId ?? null,
      matchingFailureOperations: references(failed, 'operation'),
      failedOperations: references(capture.operations.filter(op => op.status === 'failed'), 'operation'),
      failedInvocations: references(capture.invocations.filter(call => call.status === 'failed'), 'invocation'),
      upstream: references(upstream, 'operation'), upstreamBasis: 'matching_diagnostic_candidates', diagnostics: read(['diagnostics']),
      detail: 'Upstream records follow exact value producers from operations sharing the root failure diagnostic. Diagnostic equality does not prove exception causality: earlier caught failures may share those facts. Assertions may match no operation. Use section capture with an entity to inspect ordered inputs, outputs and effects.' };
  }
  return { ...base, terminalDiagnosticId: root?.diagnosticId ?? null,
    operationCount: capture.operations.length, moduleCallCount: capture.invocations.length,
    observedModulePathCount: capture.coverage.modules.filter(module => module.status === 'observed').length,
    notObservedModulePathCount: capture.coverage.modules.filter(module => module.status === 'not_observed').length,
    detail: 'Canonical operations, values, invocations, state and coverage are retained for this run. Use section capture for exact argument trees; observations cover only this execution.' };
}
