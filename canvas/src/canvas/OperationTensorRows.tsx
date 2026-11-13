import React from 'react';
import type { ModelNodeData } from '../types';
import { operationTensorRows } from './operationTensorData';
import { tensorRowLayout } from './cardContentLayout';

export function OperationTensorRows({ data, width = 200, compact = false }: { data: ModelNodeData; width?: number; compact?: boolean }) {
  const rows = operationTensorRows(data) ?? [];
  return <div className="operation-tensor-rows" style={{ gap: compact ? 4 : 6 }} role="group" aria-label="Operation input and output shapes">
    {rows.map(row => {
      const layout = tensorRowLayout(row.label, row.value, width, compact ? 16.5 : 21);
      return <div key={row.id} data-direction={row.direction} title={row.title} style={{ height: layout.height }}>
        <span className="operation-tensor-label card-content-lines" style={{ width: layout.labelWidth, lineHeight: '14px' }}>{layout.labelLines.length === 1 ? row.label : layout.labelLines.map((line, i) => <span key={i}>{line}</span>)}</span>
        <code className="card-content-lines" title={row.value} style={{ lineHeight: `${layout.lineHeight}px` }}>{layout.valueLines.length === 1 ? row.value : layout.valueLines.map((line, i) => <span key={i}>{line}</span>)}</code>
      </div>;
    })}
    {!rows.length && <span className="graph-shapes-unknown">No tensor shapes recorded</span>}
  </div>;
}
