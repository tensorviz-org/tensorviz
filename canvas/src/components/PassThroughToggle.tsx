import React from 'react';
import type { ModelNode } from '../types';
import { Check, Eye } from 'lucide-react';

export function PassThroughToggle({ nodes, onToggle, className, menuItem = false }: {
  nodes: ModelNode[]; onToggle: (shown: boolean) => void; className?: string; menuItem?: boolean;
}) {
  const calls = nodes.filter(node => node.data.capturePassThrough);
  if (!calls.length) return null;
  const shown = calls.some(node => node.data.passThroughShown);
  const label = `${shown ? 'Hide' : 'Show'} unchanged calls (${calls.length})`;
  return <button className={className} type="button" aria-label={label} role={menuItem ? 'menuitemcheckbox' : undefined}
    aria-pressed={menuItem ? undefined : shown} aria-checked={menuItem ? shown : undefined}
    title="Recorded Identity and Dropout calls that returned their input unchanged. Dashed connectors identify the original input."
    data-testid="toggle-unchanged-calls" onClick={() => onToggle(!shown)}>
    {menuItem && <Eye size={15} aria-hidden="true" />}<span>{label}</span>{menuItem && shown && <Check size={14} aria-hidden="true" />}
  </button>;
}
