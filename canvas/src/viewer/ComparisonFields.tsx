import React from 'react';
import type { FieldChange } from '../services/graphComparison';

export default function ComparisonFields({ changes, absent = 'Not present', previewLimit = Infinity }: {
  changes: FieldChange[]; absent?: string; previewLimit?: number;
}): React.ReactElement {
  let truncated = false;
  const value = (item: FieldChange['before']) => {
    const text = item.present ? JSON.stringify(item.value, null, 2) ?? 'Not recorded' : absent;
    if (text.length <= previewLimit) return text;
    truncated = true; return `${text.slice(0, previewLimit)}\n… [preview limited]`;
  };
  const rows = changes.map(change => <tr key={change.field}><th>{change.field}</th>
    <td><pre>{value(change.before)}</pre></td><td><pre>{value(change.after)}</pre></td></tr>);
  return <><table className="graph-comparison-fields"><thead><tr><th>Field</th><th>Before</th><th>After</th></tr></thead>
    <tbody>{rows}</tbody></table>
    {truncated && <p className="graph-panel-hint">Values above are previews. Use Inspect before or Inspect after to read the full retained tree. Comparison used the complete values.</p>}
  </>;
}
