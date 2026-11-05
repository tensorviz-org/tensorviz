/** Deterministic identity shared by cards, selection and SVG exports.
 * Generated interpretations are supplied separately by React context. */
import type { ModelNodeData } from '../types';
import { customCardModel } from './customCard';
import { operationDisplayName } from './operationDisplayName';
import { tensorEvidence, tensorShapes } from './graphSemantics';
import { portLabel } from './capturePorts';
import { getLayerByType } from '../data/layerTypes';

const text = (value: unknown): string | undefined => typeof value === 'string' && value.trim() ? value.trim() : undefined;
const publicLabel = (value: unknown): string | undefined => {
  const label = text(value);
  return label && !/^#?python@/.test(label) ? label : undefined;
};

export function nodePresentation(data: ModelNodeData, nodeId: string) {
  const custom = customCardModel(data.layerType?.id, data.parameters);
  const rawCall = text(data.parameters?._raw_call);
  const result = data.parameters?._source_result_call as { name?: unknown; target?: unknown } | undefined;
  // Only simple source callees are read here. Chained calls keep their exact
  // statement below; a name is never expanded into a semantic purpose.
  const sourceCallee = rawCall?.match(/^(?:[\w.]+\s*=\s*)?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(/)?.[1];
  const operation = operationDisplayName(data);
  const capturedTarget = data.captureRef?.kind === 'operation' && operation && /^(?:aten[.:]|torch\.ops\.)/.test(data.captureLabel ?? '') ? data.captureLabel : undefined;
  const callable = text(result?.target) || (sourceCallee && (!result?.name || sourceCallee.split('.').pop() === result.name) ? sourceCallee : undefined)
    || text(result?.name) || text(data.parameters?._class)
    || (data.captureRef?.kind === 'invocation' ? text(data.layerType?.name) : undefined)
    || (data.layerType?.id === 'module' ? text(data.moduleTemplateId) : undefined) || capturedTarget || text(data.layerType?.name);
  const title = (custom?.title === 'Custom Code' ? publicLabel((data as { label?: string }).label) : undefined)
    || custom?.title || operation || publicLabel(data.layerType?.displayName)
    || publicLabel((data as { label?: string }).label) || publicLabel(data.captureLabel) || publicLabel(data.layerType?.name) || publicLabel(nodeId) || 'Python call';
  const path = publicLabel(data.captureLabel);
  const subtitle = callable && callable !== title ? callable : path && path !== title ? path : undefined;
  // Catalogue descriptions are reference information, not model-specific intent.
  const purpose = custom || !data.layerType?.id ? undefined : text(getLayerByType(data.layerType.id)?.description);
  return { title, callable, subtitle, rawCall, purpose };
}

export function nodeEvidenceLabel(data: ModelNodeData): string {
  const evidence = tensorEvidence(data);
  if (data.debugEvidence === 'previous' || evidence === 'stale') return 'Previous capture · stale';
  if (evidence === 'measured') return 'Captured · shapes recorded';
  if (evidence === 'inferred') return 'Shapes inferred';
  return data.debugEvidence === 'source' || data.sourceLocation || data.parameters?._raw_call
    ? 'From source · shapes not recorded' : data.captureRef ? 'Captured · shapes not recorded' : 'Shapes not recorded';
}

export function nodeInputOutput(data: ModelNodeData): { inputs: string[]; outputs: string[] } {
  const evidence = tensorEvidence(data);
  const shapes = tensorShapes(data);
  const values = (direction: 'in' | 'out'): string[] => {
    const ports = data.captureDisplayPorts ?? data.capturePorts;
    const recorded = ports?.filter(port => port.direction === direction && port.argumentKind === 'value');
    if (recorded?.length) return recorded.map((port, index) => {
      const shape = evidence !== 'unknown' && Array.isArray(port.shape) ? ` (${port.shape.join(', ')})` : '';
      return `${portLabel(port, index)}${shape}`;
    });
    const names = direction === 'in' ? data.inputPorts : data.outputPorts;
    const shape = evidence !== 'unknown' ? direction === 'in' ? shapes.inputShape : shapes.outputShape : undefined;
    if (names?.length) return names.map(name => `${name}${shape && names.length === 1 ? ` (${shape.join(', ')})` : ''}`);
    return shape ? [`Tensor (${shape.join(', ')})`] : [];
  };
  return { inputs: values('in'), outputs: values('out') };
}
