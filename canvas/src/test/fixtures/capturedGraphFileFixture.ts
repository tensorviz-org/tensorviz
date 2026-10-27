import document from './canvas/capture-document.json';
import portable from '../../../../tests/fixtures/contracts/portable-capture.json';
import { createGraphFile, parseGraphFile, type GraphFile } from '../../services/graphFile';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { requirePortableCapture } from '../../services/portableCapture';
import { capturePorts, captureReference } from '../../services/portableCaptureDocument';
import type { GraphDocumentV2 } from '../../types/graphDocument';

export const cloneCaptureFile = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
export const capturedFile = () => createGraphFile(graphDocumentV2ToModelGraph(cloneCaptureFile(document) as GraphDocumentV2), {
  name: 'Model', status: 'traced', capture: requirePortableCapture(cloneCaptureFile(portable)),
});
export const validatedCaptureFile = (file: GraphFile) => parseGraphFile(JSON.stringify(file));
export const operationNode = (file: GraphFile, id: string) => file.document.semantic.nodes.find(node => {
  const ref = captureReference(node.metadata.captureRef); return ref.kind === 'operation' && ref.semanticId === id;
})!;

/** Every scoped identity changes, including view dictionary keys. Detached source
 * identity and actual literal contents remain the same in this small fixture. */
export function renameCaptureFile(file: GraphFile): GraphFile {
  const capture = file.capture!.capture;
  const ids = new Set([capture.id, capture.attemptId,
    ...[capture.definitions, capture.modules, capture.invocations, capture.operations, capture.values, capture.sourceBindings].flatMap(items => items.map(item => item.id)),
    ...capture.values.flatMap(value => [value.objectId, value.storageId].filter((id): id is string => id !== null)),
    ...file.document.semantic.nodes.flatMap(node => [node.node_id, ...capturePorts(node.metadata.capturePorts).map(port => port.id)]),
    ...file.document.semantic.edges.map(edge => edge.edge_id)]);
  const rename = (id: string) => ids.has(id) ? `after:${id}` : id;
  const rewrite = (value: unknown): unknown => Array.isArray(value) ? value.map(rewrite)
    : value !== null && typeof value === 'object' ? Object.fromEntries(Object.entries(value).map(([key, value]) => [rename(key), rewrite(value)]))
      : typeof value === 'string' ? rename(value) : value;
  return validatedCaptureFile(rewrite(file) as GraphFile);
}

export function bindOperation(file: GraphFile, index: number) {
  const capture = file.capture!.capture, id = `binding:${index}`;
  capture.sourceBindings.push({ id, snapshotId: capture.source.id, kind: 'call', quality: 'exact', anchors: [{
    file: capture.source.modelFile, digest: capture.source.files[0].digest, startLine: index + 10, endLine: index + 10,
    startColumn: 0, endColumn: 12, scope: 'Model.forward', tokenDigest: String(index + 1).repeat(64),
  }] });
  capture.operations[index].sourceBindingIds = [id];
}

/** Update the exact operand port and original edge alongside canonical evidence. */
export function replaceOperand(file: GraphFile, operation: number, argument: number, valueId: string) {
  const op = file.capture!.capture.operations[operation];
  op.arguments.args[argument] = { kind: 'value', id: valueId };
  const node = operationNode(file, op.id);
  const port = capturePorts(node.metadata.capturePorts).find(port => JSON.stringify(port.paths) === JSON.stringify([['arguments', 'args', argument]]))!;
  port.valueId = valueId;
  const source = file.document.semantic.nodes.flatMap(node => capturePorts(node.metadata.capturePorts))
    .find(port => port.direction === 'out' && port.valueId === valueId && file.document.semantic.nodes.find(node => node.node_id === port.nodeId)?.kind !== 'module')!;
  const edge = file.document.semantic.edges.find(edge => edge.target_port === port.id)!;
  edge.source_node_id = source.nodeId; edge.source_port = source.id;
  Object.assign(edge.metadata.captureEdge as object, { valueId, sourcePortId: source.id, displaySourcePortId: source.id });
}
