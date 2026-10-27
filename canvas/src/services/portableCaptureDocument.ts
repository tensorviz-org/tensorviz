/** Verify every rendered operand/use against the retained computation. Layout
 * and annotations can change; missing, invented or redirected computation cannot. */
import type { components } from '../types/semanticContracts.generated';
import type { GraphDocumentV2 } from '../types/graphDocument';
import { argumentLeaves, inputLeaves, requireCaptureFact as check, sameJson } from './portableCapture';
import type { Argument, CaptureSnapshot } from './portableCapture';
import { isCaptureReference, isDisplayPort, isProjectionEdge } from '../types/portableValidators.generated';

type Reference = components['schemas']['CaptureReference'];
type Port = components['schemas']['DisplayPort'];
type Edge = components['schemas']['ProjectionEdge'];
const referenceStructure = isCaptureReference as (value: unknown) => value is Reference;
const portStructure = isDisplayPort as (value: unknown) => value is Port;
const edgeStructure = isProjectionEdge as (value: unknown) => value is Edge;
export const captureReference = (value: unknown): Reference => { check(referenceStructure(value)); return value; };
export const capturePorts = (value: unknown): Port[] => {
  check(Array.isArray(value) && value.length <= 32768 && value.every(port => portStructure(port)));
  return value as Port[];
};
export const captureEdge = (value: unknown): Edge => { check(edgeStructure(value)); return value; };

export function validateCaptureDocument(document: GraphDocumentV2, capture: CaptureSnapshot): void {
  const byId = new Map(document.semantic.nodes.map(node => [node.node_id, node]));
  const ops = new Map(capture.operations.map(op => [op.id, op])), calls = new Map(capture.invocations.map(call => [call.id, call]));
  const modules = new Map(capture.modules.map(module => [module.id, module])), definitions = new Map(capture.definitions.map(def => [def.id, def]));
  const values = new Map(capture.values.map(value => [value.id, value]));
  const ports = new Map<string, Port>(), entities = new Map<string, string>(), uses = new Map<string, { value: string; kind: Edge['kind'] }>();
  const byPath = new Map<string, Port>();
  const entityKey = (kind: string, id: string) => JSON.stringify([kind, id]);
  const pathKey = (nodeId: string, path: Array<string | number>) => JSON.stringify([nodeId, path]);
  const inputs = inputLeaves(capture.inputs, 'inputs');
  const inputValues = new Set(inputs.flatMap(entry => entry.argument.kind === 'value' ? [entry.argument.id] : []));
  for (const node of document.semantic.nodes) {
    const ref = captureReference(node.metadata.captureRef);
    check(ref.captureId === capture.id && ref.sourceId === capture.source.id);
    const key = entityKey(ref.kind, ref.semanticId);
    check(!entities.has(key)); entities.set(key, node.node_id);
  }
  for (const node of document.semantic.nodes) {
    const ref = captureReference(node.metadata.captureRef), own = capturePorts(node.metadata.capturePorts);
    const op = ref.kind === 'operation' ? ops.get(ref.semanticId) : undefined;
    const call = ref.kind === 'invocation' ? calls.get(ref.semanticId) : undefined;
    const value = ref.kind === 'value' ? values.get(ref.semanticId) : undefined;
    if (ref.kind === 'operation') check(op && node.kind === 'layer' && node.op === op.target
      && (node.metadata.parentId ?? null) === (op.invocationId === null ? null : entities.get(entityKey('invocation', op.invocationId))));
    if (ref.kind === 'invocation') check(call && node.kind === 'module' && node.op === definitions.get(modules.get(call.moduleId)!.definitionId)?.qualifiedName
      && (node.metadata.parentId ?? null) === (call.parentId === null ? null : entities.get(entityKey('invocation', call.parentId))));
    if (ref.kind === 'value') check(value && value.producer === null && !inputValues.has(value.id) && node.kind === 'layer');
    if (ref.kind === 'input' || ref.kind === 'output') check(ref.semanticId === capture.id && node.kind === ref.kind && !node.metadata.parentId);
    const record = op ?? call;
    const expected: Array<{ path: Array<string | number>; argument: Argument; direction: Port['direction'] }> = record ? [
      ...inputLeaves(record.arguments, 'arguments').map(entry => ({ ...entry, direction: 'in' as const })),
      ...argumentLeaves(record.output, ['output']).map(entry => ({ ...entry, direction: 'out' as const })),
    ] : ref.kind === 'input' ? inputs.map(entry => ({ ...entry, direction: 'out' as const }))
      : ref.kind === 'output' ? argumentLeaves(capture.output, ['output']).map(entry => ({ ...entry, direction: 'in' as const })) : [];
    if (op) op.effects.forEach((effect, i) => expected.push(
      { path: ['effects', i, 'beforeId'], argument: { kind: 'value', id: effect.beforeId }, direction: 'in' },
      { path: ['effects', i, 'after'], argument: effect.after, direction: 'out' }));
    const expectedPaths = new Map(expected.map(entry => [JSON.stringify(entry.path), entry]));
    check(sameJson(node.in_ports, own.filter(port => port.direction === 'in').map(port => port.id))
      && sameJson(node.out_ports, own.filter(port => port.direction === 'out').map(port => port.id)));
    let pathsSeen = 0;
    for (const port of own) {
      check(!ports.has(port.id) && port.nodeId === node.node_id && port.memberPortId === null);
      check((port.argumentKind === 'value') === (port.valueId !== null));
      ports.set(port.id, port);
      const fact = port.valueId === null ? undefined : values.get(port.valueId);
      check(port.valueId === null ? port.shape === null && port.dtype === null && port.device === null
        : fact && sameJson(fact.shape, port.shape) && fact.dtype === port.dtype && fact.device === port.device);
      if (value) check(own.length === 1 && !port.paths.length && port.direction === 'out' && port.valueId === value.id);
      else check(port.paths.length && ((ref.kind === 'input' && port.valueId !== null) || port.paths.length === 1));
      for (const path of port.paths) {
        const pathId = pathKey(node.node_id, path), entry = expectedPaths.get(JSON.stringify(path));
        check(!byPath.has(pathId) && entry && entry.direction === port.direction && entry.argument.kind === port.argumentKind
          && (entry.argument.kind !== 'value' || entry.argument.id === port.valueId));
        byPath.set(pathId, port); pathsSeen++;
        if (port.valueId !== null && ((op && port.direction === 'in') || ref.kind === 'output')) {
          check(!uses.has(port.id)); uses.set(port.id, { value: port.valueId, kind: ref.kind === 'output' ? 'result' : path[0] === 'effects' ? 'effect' : 'argument' });
        }
      }
    }
    check(pathsSeen === expected.length && (!value || own.length === 1));
    const status = record?.status ?? (ref.kind === 'output' && capture.output === null ? 'interrupted' : 'completed');
    const completedEvidence = capture.backend === 'jaxpr' ? 'inferred' : 'measured';
    check(node.metadata.shapeEvidence === 'stale' || node.metadata.shapeEvidence === (status === 'completed' ? completedEvidence : 'unknown'));
    if (node.metadata.parameterCount !== undefined) check(call && node.metadata.parameterCount === call.parameterCount);
    if (node.metadata.outputTensor !== undefined) {
      const output = record?.output ?? (value ? { kind: 'value' as const, id: value.id } : ref.kind === 'input' && inputs.length === 1 ? inputs[0].argument : undefined);
      const fact = output?.kind === 'value' ? values.get(output.id) : undefined;
      check(fact && fact.shape !== null && fact.logicalBytes != null && sameJson(node.metadata.outputTensor,
        { shape: fact.shape, dtype: fact.dtype.replace(/^torch\./, ''), logicalBytes: fact.logicalBytes }));
    }
    const shapeOf = (argument: Argument | null | undefined) => argument?.kind === 'value' ? values.get(argument.id)?.shape : undefined;
    const arguments_ = record ? [...record.arguments.args, ...Object.values(record.arguments.kwargs)] : [];
    const inputShape = record ? arguments_.length === 1 ? shapeOf(arguments_[0]) : undefined
      : ref.kind === 'output' && own.length === 1 ? own[0].shape : undefined;
    const outputShape = record ? shapeOf(record.output) : value?.shape ?? (ref.kind === 'input' && own.length === 1 ? own[0].shape : undefined);
    if (node.metadata.inputShape !== undefined) check(inputShape != null && sameJson(node.metadata.inputShape, inputShape));
    if (node.metadata.outputShape !== undefined) check(outputShape != null && sameJson(node.metadata.outputShape, outputShape));
    node.params = { _shape_provenance: status === 'completed' ? capture.backend === 'jaxpr' ? 'inferred' : 'observed' : status === 'failed' ? 'failed' : 'unknown',
      ...(node.metadata.inputShape !== undefined ? { _in_shape: node.metadata.inputShape } : {}),
      ...(node.metadata.outputShape !== undefined ? { _out_shape: node.metadata.outputShape } : {}) };
    if (status === 'failed') node.metadata.error = 'This captured call failed. Inspect its recorded inputs and diagnostic.';
  }
  // A saved document cannot hide a captured event or add a lookalike event.
  const expectedEntities = [entityKey('input', capture.id), entityKey('output', capture.id),
    ...capture.invocations.map(call => entityKey('invocation', call.id)), ...capture.operations.map(op => entityKey('operation', op.id)),
    ...capture.values.filter(value => value.producer === null && !inputValues.has(value.id)).map(value => entityKey('value', value.id))];
  check(entities.size === expectedEntities.length && expectedEntities.every(key => entities.has(key)));
  const sources = new Map<string, string>();
  const boundarySources = new Map<string, Port>();
  for (const port of Array.from(ports.values())) {
    const ref = captureReference(byId.get(port.nodeId)!.metadata.captureRef);
    if ((ref.kind === 'input' || ref.kind === 'value') && port.valueId !== null) {
      check(!boundarySources.has(port.valueId)); boundarySources.set(port.valueId, port);
    }
  }
  for (const value of capture.values) {
    const producer = value.producer;
    const port = producer ? byPath.get(pathKey(entities.get(entityKey('operation', producer.operationId))!,
      [producer.kind === 'output' ? 'output' : 'effects', ...producer.path]))
      : boundarySources.get(value.id);
    check(port); sources.set(value.id, port.id);
  }
  const used = new Set<string>();
  for (const edge of document.semantic.edges) {
    const captured = captureEdge(edge.metadata.captureEdge), source = ports.get(edge.source_port), target = ports.get(edge.target_port);
    const use = uses.get(edge.target_port);
    check(source && target && use && !used.has(target.id) && captured.valueId === use.value && captured.kind === use.kind
      && source.nodeId === edge.source_node_id && target.nodeId === edge.target_node_id && sources.get(use.value) === source.id
      && captured.id === edge.edge_id && captured.sourcePortId === source.id && captured.targetPortId === target.id
      && captured.displaySourcePortId === source.id && captured.displayTargetPortId === target.id);
    used.add(target.id);
  }
  check(used.size === uses.size);
  // Counts are view summaries derived from actual descendants, never file claims.
  for (const node of document.semantic.nodes) node.metadata.capturedOperationCount = captureReference(node.metadata.captureRef).kind === 'operation' ? 1 : 0;
  for (const node of document.semantic.nodes) {
    if (captureReference(node.metadata.captureRef).kind !== 'operation') continue;
    let parent = node.metadata.parentId;
    while (typeof parent === 'string') { const owner = byId.get(parent)!; owner.metadata.capturedOperationCount = Number(owner.metadata.capturedOperationCount) + 1; parent = owner.metadata.parentId; }
  }
  document.semantic.metadata = { capture: { version: 1, id: capture.id, sourceId: capture.source.id, status: capture.status,
    backend: capture.backend, rootInvocationId: capture.rootInvocationId, omissions: [...capture.coverage.omissions],
    operationCount: capture.operations.length, invocationCount: capture.invocations.length } };
}
