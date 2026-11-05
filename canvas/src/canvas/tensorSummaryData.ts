import type { ModelNodeData } from '../types';
import { tensorEvidence, tensorShapes, coverageLabel } from './graphSemantics';
import { operationTensorRows } from './operationTensorData';
import { tensorRowLayout } from './cardContentLayout';
import { nodeFailureLines } from './graphIssues';

export const tensorShapeValue = (values: number[]) => `(${values.map(value => value < 0 ? '?' : value).join(', ')})`;

/** Derive the content once so card geometry and React/SVG cannot disagree. */
export function tensorSummaryData(rawData: ModelNodeData, width = 200, compact = false) {
  const data = { ...rawData, ...tensorShapes(rawData) };
  const evidence = tensorEvidence(data), coverage = coverageLabel(data);
  const features = data.layerType?.id === 'linear';
  const channels = /^conv[123]d$/.test(data.layerType?.id ?? '');
  const parameters = data.parameters ?? {};
  const before = features ? parameters.in_features : channels ? parameters.in_channels : undefined;
  const after = features ? parameters.out_features : channels ? parameters.out_channels : undefined;
  const parameterPair = typeof before === 'number' && typeof after === 'number';
  const evidenceLabel = evidence === 'stale' ? evidence : coverageLabel(data, true)
    || (parameterPair && !data.inputShape && !data.outputShape ? 'Parameters' : evidence === 'measured' ? '' : evidence);
  const operation = operationTensorRows(data);
  const shapeRows = [
    ...(data.inputShape ? [{ id: 'in', direction: 'in', label: 'IN', value: tensorShapeValue(data.inputShape), shape: data.inputShape }] : []),
    ...(data.outputShape ? [{ id: 'out', direction: 'out', label: 'OUT', value: tensorShapeValue(data.outputShape), shape: data.outputShape }] : []),
  ];
  const rows = (operation ?? shapeRows).map(row => ({ ...row, ...tensorRowLayout(row.label, row.value, width, compact ? 16.5 : 21) }));
  const rowGap = compact ? 4 : 6;
  const rowsHeight = rows.reduce((sum, row) => sum + row.height, 0) + Math.max(0, rows.length - 1) * rowGap;
  const gap = compact ? 4 : 7, padding = compact ? 14 : 24;
  const failure = Boolean(data.error && data.captureDiagnostic);
  const warning = Boolean(data.error || data.moduleStatus === 'error' || data.moduleStatus === 'warning');
  const bodyHeight = failure
    ? (operation !== undefined ? (rows.length ? rowsHeight : 16) + gap : 0) + Math.max(16, nodeFailureLines(data).length * 16)
    : (evidenceLabel ? 12 + gap : 0) + (operation !== undefined ? rows.length ? rowsHeight : 16
      : rows.length ? rowsHeight : parameterPair ? compact ? 21 : 26 : 16) + (warning ? gap + 14 : 0);
  return { data, evidence, coverage, features, before, after, parameterPair, evidenceLabel, operation, rows,
    rowGap, rowsHeight, height: padding + bodyHeight, failure };
}
