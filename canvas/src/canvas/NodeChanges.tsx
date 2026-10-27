import React from 'react';
import type { ModelNodeData } from '../types';

export function NodeChanges({ data, compact = false }: { data: ModelNodeData; compact?: boolean }) {
  if (!data.debugChanges?.length || data.debugEvidence === 'previous') return null;
  return <span className={`graph-node-changes${compact ? ' compact' : ''}`}
    title={`Observed changes since the previous complete run. These do not prove the cause of a failure.\n${data.debugChanges.join('\n')}`}>
    <strong>Changed</strong>{!compact && <span>{data.debugChanges[0]}{data.debugChanges.length > 1 ? ` (+${data.debugChanges.length - 1})` : ''}</span>}
  </span>;
}
