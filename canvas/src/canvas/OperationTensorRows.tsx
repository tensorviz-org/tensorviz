import React from 'react';
import type { ModelNodeData } from '../types';
import { operationTensorRows } from './operationTensorData';

export function OperationTensorRows({ data }: { data: ModelNodeData }) {
  const rows = operationTensorRows(data) ?? [];
  return <div className="operation-tensor-rows" role="group" aria-label="Operation input and output shapes">
    {rows.map(row => <div key={row.id} data-direction={row.direction} title={row.title}>
      <span className="operation-tensor-label">{row.label}</span><code title={row.value}>{row.value}</code>
    </div>)}
    {!rows.length && <span className="graph-shapes-unknown">No tensor shapes recorded</span>}
  </div>;
}
