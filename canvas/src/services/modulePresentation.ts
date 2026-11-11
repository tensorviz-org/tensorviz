import { ModelNodeData } from '../types';
import { tensorEvidence, tensorShapes, coverageLabel } from '../canvas/graphSemantics';
import { portLabel, portType } from '../canvas/capturePorts';
import { cardParams } from '../ide/cardParams';
import { boundaryTextLayout } from '../canvas/cardContentLayout';

/** Short boundary labels (In, Out, In x) share one line with their value; longer
 *  names keep the two-line row. Shared by the renderer and cardMetrics. */
export const compactBoundaryLabel = (label: string): boolean => label.length <= 6;
export const BOUNDARY_ROW_COMPACT = 26; // 22px row + 4px gap
export const BOUNDARY_ROW_FULL = 40;    // 36px row + 4px gap

/** At most four constructor arguments, two per row. A fifth or later collapses into the
 *  fourth cell as "+N more". Captured calls without source arguments draw no grid. */
export const MODULE_PARAM_CELLS = 4;
export function moduleParameters(data: ModelNodeData): ReadonlyArray<readonly [string, unknown]> {
  if (!data.captureRef || data.capturePassThrough) return [];
  // `name` is usually display metadata; keep it only when the source proved it as an argument.
  const proven = data.parameters?._source_arguments;
  const namedInSource = !!proven && typeof proven === 'object' && !Array.isArray(proven)
    && 'name' in proven && typeof proven.name === 'string';
  return cardParams(data.parameters).entries.filter(([key]) => key !== 'name' || namedInSource);
}
export function moduleParameterRows(data: ModelNodeData): number {
  return Math.ceil(Math.min(moduleParameters(data).length, MODULE_PARAM_CELLS) / 2);
}

/** The call's own interface, never the synthetic ports of collapsed children. */
export function moduleBoundaryRows(data: ModelNodeData): Array<{ id: string; label: string; value: string; path?: string }> {
  // A replaced call has not run: its ports keep the flow, not the old measurements.
  if (data.sourceReplacement) return [];
  if (data.capturePorts) {
    const inputs = data.capturePorts.filter(port => port.direction === 'in').length;
    const outputs = data.capturePorts.length - inputs;
    const ordinal = { in: 0, out: 0 };
    return data.capturePorts.map(port => {
      const direction = port.direction === 'in' ? 'In' : 'Out';
      const multiple = (port.direction === 'in' ? inputs : outputs) > 1;
      const index = ordinal[port.direction]++;
      const named = (port.displayName && !/^\d+$/.test(port.displayName)) || port.paths.some(path => path[1] === 'kwargs');
      return { id: port.id, label: multiple || named ? portLabel(port, index, direction) : direction,
        value: portType(port), path: portLabel(port, index) };
    });
  }
  // Legacy snapshots carry single-tensor shapes. Trace provenance takes priority
  // over cached presentation values, including when a trace has no output.
  const { inputShape, outputShape } = tensorShapes(data);
  return [
    ...(inputShape ? [{ id: 'in', label: 'In', value: `(${inputShape.join(', ')})` }] : []),
    ...(outputShape ? [{ id: 'out', label: 'Out', value: `(${outputShape.join(', ')})` }] : []),
  ];
}

export function moduleBoundaryLayout(data: ModelNodeData, width: number) {
  return moduleBoundaryRows(data).map(row => ({ ...row, ...boundaryTextLayout(row.label, row.value, width, compactBoundaryLabel(row.label)) }));
}

/** Captured operation counts belong in capture details, not on module tiles. */
export function moduleContents(data: ModelNodeData) {
  const count = data.moduleLayerCount || 0;
  const canExpand = data.expandable ?? count > 0;
  if (data.captureRef) {
    if (data.sourceReplacement) return { canExpand: false, label: '',
      detail: 'Not run: the replacement module’s contents are unknown until the model is validated.' };
    if (data.capturePassThrough) return { canExpand: false, label: 'Input unchanged',
      detail: 'This call returned the same tensor without recording tensor operations.' };
    const operations = data.capturedOperationCount ?? 0;
    return { canExpand, label: '',
      detail: operations === 0 ? 'This call recorded no tensor operations. Its observed interface is retained.' : '' };
  }
  const limited = data.moduleCapture?.status === 'depth-limited';
  const captured = limited || ['observed', 'unreached'].includes(data.parameters?._shape_provenance);
  const declared = !canExpand && Array.isArray(data.parameters?._modules)
    ? cardParams(data.parameters).contents || '0 members' : undefined;
  return {
    canExpand,
    label: canExpand ? `${count} ${captured ? 'captured ' : ''}layer${count === 1 ? '' : 's'}`
      : limited ? 'Depth-limited' : declared ? `Declared: ${declared}` : '',
    detail: limited
      ? `Internal trace contents not captured at depth ${data.moduleCapture!.maxDepth}.`
      : !canExpand ? 'No internal contents are available in this graph.' : '',
  };
}

/** Empty means measured without a badge; undefined allows the source-only fallback. */
export function moduleEvidenceLabel(data: ModelNodeData): string | undefined {
  if (data.sourceReplacement) return 'Not run';
  if (data.debugEvidence === 'previous') return 'Previous run';
  if (data.debugEvidence === 'source') return '';
  if (data.shapeEvidence === 'stale') return 'Stale shapes';
  const coverage = coverageLabel(data);
  if (coverage) return coverage;
  const params = data.parameters || {};
  if (params._shape_provenance === 'unreached') return 'Not observed';
  if (params._shape_provenance === 'declared') return 'Declared shapes';
  if (data.shapeEvidence === 'unknown' || params._shape_provenance === 'unknown') return undefined;
  if (data.shapeEvidence === 'measured' || params._shape_provenance === 'observed') {
    return tensorEvidence(data) === 'measured' ? '' : 'Shapes unavailable';
  }
  if (data.shapeEvidence === 'inferred') return 'Inferred shapes';
  return undefined;
}
