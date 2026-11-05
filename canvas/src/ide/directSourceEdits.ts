import { ModelNodeData } from '../types';
import { LocalBrowserSourceEdit, SourceEditValue } from './protocol.generated';

const POSITIONAL: Record<string, string[]> = {
  Linear: ['in_features', 'out_features', 'bias'], Dropout: ['p', 'inplace'],
  LayerNorm: ['normalized_shape', 'eps', 'elementwise_affine', 'bias'],
};
export const DIRECT_LAYERS = ['linear', 'dropout', 'layernorm', 'relu', 'gelu', 'silu', 'sigmoid', 'tanh'];
export const sourceEditLimitation = (data: ModelNodeData) => data.sourceEditLimitation ||
  'This node has no safe source edit. Use the agent for dynamic code or edit the source directly.';
export function hasDeclarationEdit(data: ModelNodeData): boolean {
  return !data.focusBoundary && ((!data.captureRef && !!data.sourceLocation) ||
    !!data.sourceEditCapabilities?.some(cap => cap.command === 'set_constructor_literal' &&
      cap.expression.kind === 'constructor' && cap.expression.quality === 'exact'));
}
export function constructorParameters(data: ModelNodeData): string[] {
  const names = (data.sourceEditCapabilities || []).flatMap(cap => {
    if (cap.command !== 'set_constructor_literal') return [];
    const [kind, key] = cap.sourceArgumentPath;
    const name = kind === 'kwargs' ? String(key) : POSITIONAL[cap.constructor.split('.').pop()!]?.[Number(key)];
    return name ? [name] : [];
  });
  if (hasDeclarationEdit(data)) names.push(...Object.keys(data.parameters).filter(key =>
    !key.startsWith('_') && ['number', 'boolean'].includes(typeof data.parameters[key])));
  return [...new Set(names)];
}
export function parameterSourceEdit(nodeId: string, data: ModelNodeData, param: string, value: SourceEditValue): LocalBrowserSourceEdit {
  const cap = data.sourceEditCapabilities?.find(cap => cap.command === 'set_constructor_literal' &&
    (cap.sourceArgumentPath[0] === 'kwargs' ? cap.sourceArgumentPath[1] === param :
      POSITIONAL[cap.constructor.split('.').pop()!]?.[Number(cap.sourceArgumentPath[1])] === param));
  if (cap?.command === 'set_constructor_literal') {
    if (typeof value !== 'number' || !Number.isFinite(value) || (cap.expected.kind === 'int' && !Number.isSafeInteger(value))) {
      throw new Error('Enter a finite number of the same type as the constructor argument.');
    }
    return { kind: 'set_constructor_literal', nodeId, capabilityId: cap.id, target: cap.target,
      expected: cap.expected, scope: cap.scope.kind, sourceArgumentPath: cap.sourceArgumentPath,
      value: cap.expected.kind === 'int' ? { kind: 'int', value: String(value) } : { kind: 'float', value } };
  }
  if (!hasDeclarationEdit(data)) throw new Error(sourceEditLimitation(data));
  return { kind: 'set_param', nodeId, param, value };
}
export function removeSourceEdit(nodeId: string, data: ModelNodeData): LocalBrowserSourceEdit {
  const cap = data.sourceEditCapabilities?.find(cap => cap.command === 'delete_unary_op');
  if (cap?.command === 'delete_unary_op') return { kind: cap.command, nodeId, capabilityId: cap.id,
    target: cap.target, expected: cap.expected, scope: cap.scope.kind, argumentPath: cap.argumentPath, value: cap.input.value };
  if (!hasDeclarationEdit(data)) throw new Error(sourceEditLimitation(data));
  return { kind: 'remove_layer', nodeId };
}
export function addSourceEdit(nodeId: string, data: ModelNodeData, layerType: string,
  name: string, params: Record<string, SourceEditValue>): LocalBrowserSourceEdit {
  if (!DIRECT_LAYERS.includes(layerType)) throw new Error('This layer requires the agent or a source edit.');
  if (!hasDeclarationEdit(data)) throw new Error(sourceEditLimitation(data));
  return { kind: 'add_layer', afterNodeId: nodeId, nodeId: name, layerType, params };
}
export function insertUnarySourceEdit(nodeId: string, data: ModelNodeData, capabilityId: string, value: string): LocalBrowserSourceEdit {
  const cap = data.sourceEditCapabilities?.find(cap => cap.id === capabilityId && cap.command === 'insert_unary_op');
  if (cap?.command !== 'insert_unary_op' || !['relu', 'gelu', 'silu', 'sigmoid', 'tanh'].includes(value)) {
    throw new Error(sourceEditLimitation(data));
  }
  return { kind: 'insert_unary_op', nodeId, capabilityId: cap.id, target: cap.target,
    expected: cap.expected, scope: cap.scope.kind, argumentPath: cap.argumentPath,
    value: value as 'relu' | 'gelu' | 'silu' | 'sigmoid' | 'tanh' };
}
