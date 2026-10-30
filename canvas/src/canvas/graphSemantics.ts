import { ModelEdgeData, ModelNodeData } from '../types';

/** A closed allowlist: portable files must not smuggle arbitrary source metadata. */
export function residualMetadata(value: unknown): Pick<ModelEdgeData, 'role' | 'roleEvidence'> {
  if (!value || typeof value !== 'object') return {};
  const data = value as ModelEdgeData;
  if (data.role !== 'residual') return {};
  if (!['identity-bypass', 'projection-shortcut', 'declared'].includes(data.roleEvidence || '')) return {};
  return { role: 'residual', roleEvidence: data.roleEvidence };
}

type TensorData = Pick<ModelNodeData, 'parameters' | 'inputShape' | 'outputShape' | 'shapeEvidence' | 'capturePorts'>;
const UNMEASURED = ['unreached', 'unknown', 'not_observed', 'capture_limited', 'static_only', 'unresolved', 'failed'];

/** Fixed labels for source/capture correspondence; no arbitrary runtime prose. */
export function coverageLabel(data: Pick<ModelNodeData, 'parameters'>, compact = false): string | undefined {
  const coverage = data.parameters?._capture_coverage;
  if (!coverage || typeof coverage !== 'object') return undefined;
  if (coverage.status === 'state') return Array.isArray(coverage.stateValueIds) && coverage.stateValueIds.length
    ? compact ? 'State' : 'State value observed' : compact ? 'No state' : 'State value not recorded';
  if (compact) return ({ not_observed: 'Not observed', capture_limited: 'Limited',
    static_only: 'Source only', unresolved: 'Unmapped', failed: 'Failed' } as Record<string, string>)[coverage.status];
  const labels: Record<string, string> = { not_observed: 'Not observed in this run', capture_limited: 'Capture limited',
    static_only: 'Source only', unresolved: 'Source mapping unresolved', failed: 'Call failed' };
  return labels[coverage.status];
}

/** Trace provenance owns trace shapes; cached presentation fields cannot fill gaps. */
export function tensorShapes(data: TensorData): Pick<ModelNodeData, 'inputShape' | 'outputShape'> {
  const params = data.parameters || {};
  if (['observed', 'state', 'inferred', ...UNMEASURED].includes(params._shape_provenance)) return {
    inputShape: measuredShape(params._in_shape),
    outputShape: ['observed', 'state', 'inferred'].includes(params._shape_provenance) ? measuredShape(params._out_shape) : undefined,
  };
  return { inputShape: data.inputShape, outputShape: data.outputShape };
}

export function tensorEvidence(data: TensorData): NonNullable<ModelNodeData['shapeEvidence']> {
  if (data.shapeEvidence === 'stale') return 'stale';
  const provenance = data.parameters?._shape_provenance;
  if (UNMEASURED.includes(provenance)) return 'unknown';
  if (data.shapeEvidence === 'unknown') return 'unknown';
  const shapes = tensorShapes(data);
  if (!shapes.inputShape && !shapes.outputShape && !data.capturePorts?.some(port => port.argumentKind === 'value' && port.shape !== null)) return 'unknown';
  if (provenance === 'inferred') return 'inferred';
  if (data.shapeEvidence === 'measured' || provenance === 'observed' || provenance === 'state') return 'measured';
  return 'inferred';
}

/** Unknown or multi-tensor values stay unknown, never plausible shape claims. */
export function measuredShape(value: unknown): number[] | undefined {
  const dims = typeof value === 'string' && /^\d+(?:x\d+)*$/.test(value)
    ? value.split('x').map(Number) : value;
  return Array.isArray(dims) && dims.every(dim => Number.isSafeInteger(dim) && dim >= 0) ? dims : undefined;
}

export function exactCount(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

/** Closed, path-free portable metadata; no dtype-to-byte inference. */
export function outputTensorMetadata(value: unknown): ModelNodeData['outputTensor'] {
  if (!value || typeof value !== 'object') return undefined;
  const tensor = value as NonNullable<ModelNodeData['outputTensor']>;
  const shape = measuredShape(tensor.shape);
  if (!Array.isArray(tensor.shape) || !shape || shape.length > 32 || typeof tensor.dtype !== 'string'
    || !/^[a-z][a-z0-9_]{0,63}$/.test(tensor.dtype) || exactCount(tensor.logicalBytes) === undefined) return undefined;
  return { shape: [...shape], dtype: tensor.dtype, logicalBytes: tensor.logicalBytes };
}

/** Different ranks have no safe positional axis correspondence. */
export function changedDimensions(input?: number[], output?: number[]): number[] {
  if (!input || !output || input.length !== output.length) return [];
  return output.flatMap((dimension, axis) => dimension >= 0 && input[axis] >= 0 && dimension !== input[axis] ? [axis] : []);
}
