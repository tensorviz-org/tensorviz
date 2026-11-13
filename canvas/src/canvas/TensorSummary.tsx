import React from 'react';
import { ArrowRight, AlertTriangle } from 'lucide-react';
import { ModelNodeData } from '../types';
import { changedDimensions } from './graphSemantics';
import { nodeFailureLines } from './graphIssues';
import { operationTensorRows } from './operationTensorData';
import { OperationTensorRows } from './OperationTensorRows';
import { tensorSummaryData } from './tensorSummaryData';

export function TensorSummary({ data: rawData, compact = false, width = 200 }: { data: ModelNodeData; compact?: boolean; width?: number }) {
  const summary = tensorSummaryData(rawData, width, compact);
  const { data, evidence, coverage, features, before, after, parameterPair, evidenceLabel } = summary;
  const changed = changedDimensions(data.inputShape, data.outputShape);
  const shapeLines = (values: number[], lines: string[]) => {
    let offset = 1;
    const ranges = values.flatMap((value, index) => {
      const length = String(value < 0 ? '?' : value).length;
      const range = { start: offset, end: offset + length };
      offset += length + 2;
      return changed.includes(index) ? [range] : [];
    });
    let cursor = 0;
    return lines.map((line, index) => {
      const start = cursor; cursor += line.length;
      const segments: Array<{ changed: boolean; text: string }> = [];
      for (let i = 0; i < line.length; i++) {
        const hot = ranges.some(range => start + i >= range.start && start + i < range.end);
        const previous = segments[segments.length - 1];
        if (previous?.changed === hot) previous.text += line[i];
        else segments.push({ changed: hot, text: line[i] });
      }
      return <span key={index}>{segments.map((segment, i) => segment.changed
        ? <span key={i} className="graph-dimension-changed">{segment.text}</span> : segment.text)}</span>;
    });
  };
  if (data.error && data.captureDiagnostic) return <div className={`graph-tensor-summary${compact ? ' compact' : ''}`}>
    {operationTensorRows(data) !== undefined && <OperationTensorRows data={data} width={width} compact={compact} />}
    <div className="graph-summary-warning graph-summary-failure">
      <AlertTriangle size={12} /><span className="graph-failure-facts">{nodeFailureLines(data).map(line =>
        <span key={line} title={line}>{line}</span>)}</span>
    </div>
  </div>;
  return <div className={`graph-tensor-summary${compact ? ' compact' : ''}`} data-evidence={evidence}>
    {evidenceLabel && <div className="graph-summary-label"><span>{data.inputShape || data.outputShape ? 'Tensor shape' : parameterPair ? features ? 'Features' : 'Channels' : 'Tensor shape'}</span><span className="graph-evidence" title={evidence === 'stale' ? 'Measurements belong to an earlier source or run.' : coverage}>{evidenceLabel}</span></div>}
    {operationTensorRows(data) !== undefined ? <OperationTensorRows data={data} width={width} compact={compact} /> : data.inputShape || data.outputShape ? <div className="graph-shape-flow">
      {summary.rows.map(row => <span key={row.id} data-direction={row.direction} style={{ height: row.height }}>
        <span className="graph-flow-label" style={{ width: row.labelWidth }}>{row.label}</span>
        <code className="card-content-lines" title={row.value} style={{ lineHeight: `${row.lineHeight}px` }}>{shapeLines((row.direction === 'in' ? data.inputShape : data.outputShape)!, row.valueLines)}</code>
      </span>)}
    </div> : parameterPair ? <div className="graph-parameter-flow"><code>{before}</code><ArrowRight size={14} /><code className={before !== after ? 'graph-dimension-changed' : undefined}>{after}</code></div>
      : <div className="graph-shapes-unknown">Not recorded</div>}
    {(data.error || data.moduleStatus === 'error' || data.moduleStatus === 'warning') && <div className="graph-summary-warning" title={data.error}>
      <AlertTriangle size={12} /><span>{data.error ? 'Shape check failed' : `${data.moduleStatus} inside module`}</span>
    </div>}
  </div>;
}
