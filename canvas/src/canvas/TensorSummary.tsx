import React from 'react';
import { ArrowRight, AlertTriangle } from 'lucide-react';
import { ModelNodeData } from '../types';
import { changedDimensions, tensorEvidence, tensorShapes, coverageLabel } from './graphSemantics';
import { nodeFailureLines } from './graphIssues';
import { operationTensorRows } from './operationTensorData';
import { OperationTensorRows } from './OperationTensorRows';

export function TensorSummary({ data: rawData, compact = false }: { data: ModelNodeData; compact?: boolean }) {
  const data = { ...rawData, ...tensorShapes(rawData) };
  const evidence = tensorEvidence(data);
  const coverage = coverageLabel(data);
  const changed = changedDimensions(data.inputShape, data.outputShape);
  const shape = (values: number[]) => <>{'('}{values.map((value, i) =>
    <React.Fragment key={i}>{i > 0 && ', '}<span className={changed.includes(i) ? 'graph-dimension-changed' : undefined}>{value < 0 ? '?' : value}</span></React.Fragment>)}{')'}</>;
  const params = data.parameters;
  const features = data.layerType.id === 'linear';
  const channels = /^conv[123]d$/.test(data.layerType.id);
  const before = features ? params.in_features : channels ? params.in_channels : undefined;
  const after = features ? params.out_features : channels ? params.out_channels : undefined;
  const parameterPair = typeof before === 'number' && typeof after === 'number';
  const evidenceLabel = evidence === 'stale' ? evidence : coverageLabel(data, true)
    || (parameterPair && !data.inputShape && !data.outputShape ? 'Parameters' : evidence === 'measured' ? '' : evidence);
  if (data.error && data.captureDiagnostic) return <div className={`graph-tensor-summary${compact ? ' compact' : ''}`}>
    {operationTensorRows(data) !== undefined && <OperationTensorRows data={data} />}
    <div className="graph-summary-warning graph-summary-failure">
      <AlertTriangle size={12} /><span className="graph-failure-facts">{nodeFailureLines(data).map(line =>
        <span key={line} title={line}>{line}</span>)}</span>
    </div>
  </div>;
  return <div className={`graph-tensor-summary${compact ? ' compact' : ''}`} data-evidence={evidence}>
    {evidenceLabel && <div className="graph-summary-label"><span>{data.inputShape || data.outputShape ? 'Tensor shape' : parameterPair ? features ? 'Features' : 'Channels' : 'Tensor shape'}</span><span className="graph-evidence" title={evidence === 'stale' ? 'Measurements belong to an earlier source or run.' : coverage}>{evidenceLabel}</span></div>}
    {operationTensorRows(data) !== undefined ? <OperationTensorRows data={data} /> : data.inputShape || data.outputShape ? <div className="graph-shape-flow">
      {data.inputShape && <span data-direction="in"><span className="graph-flow-label">IN</span><code>{shape(data.inputShape)}</code></span>}
      {data.outputShape && <span data-direction="out"><span className="graph-flow-label">OUT</span><code>{shape(data.outputShape)}</code></span>}
    </div> : parameterPair ? <div className="graph-parameter-flow"><code>{before}</code><ArrowRight size={14} /><code className={before !== after ? 'graph-dimension-changed' : undefined}>{after}</code></div>
      : <div className="graph-shapes-unknown">Not recorded</div>}
    {(data.error || data.moduleStatus === 'error' || data.moduleStatus === 'warning') && <div className="graph-summary-warning" title={data.error}>
      <AlertTriangle size={12} /><span>{data.error ? 'Shape check failed' : `${data.moduleStatus} inside module`}</span>
    </div>}
  </div>;
}
