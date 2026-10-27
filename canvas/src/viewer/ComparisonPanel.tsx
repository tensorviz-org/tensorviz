import React, { useMemo, useState } from 'react';
import { GraphFile } from '../services/graphFile';
import { compareGraphFiles } from '../services/graphComparison';
import { GraphTarget, targetLabel } from '../services/graphCollaboration';
import Fields from './ComparisonFields';
import CaptureComparisonPanel from './CaptureComparisonPanel';

export type ComparisonSide = 'before' | 'after';
interface Props {
  before: GraphFile | null; after: GraphFile; side: ComparisonSide;
  onChoose: () => void; onNavigate: (side: ComparisonSide, target?: GraphTarget) => void; onClose: () => void;
}
export default function ComparisonPanel({ before, after, side, onChoose, onNavigate, onClose }: Props): React.ReactElement {
  const result = useMemo(() => before ? compareGraphFiles(before, after) : null, [before, after]);
  const [limit, setLimit] = useState(100);
  const nodeButton = (side: ComparisonSide, nodeId: string) => <button onClick={() => onNavigate(side, { kind: 'node', nodeId })}>Show {side}</button>;
  return <aside className="graph-viewer-panel graph-comparison-panel" aria-label="Architecture comparison">
    <div className="graph-panel-heading"><h2>Compare snapshots</h2><button aria-label="Close comparison" onClick={onClose}>×</button></div>
    <p className="graph-panel-hint">The current file is After. Choose an earlier graph file as Before. Comparison leaves both files and your notes unchanged.</p>
    <button onClick={onChoose}>{before ? 'Choose another Before file' : 'Choose Before file'}</button>
    {before && result && <>
      <div className="graph-comparison-versions">
        {(['before', 'after'] as const).map(version => { const f = version === 'before' ? before : after;
          return <section key={version}><h3>{version === 'before' ? 'Before' : 'After'}: {f.document.semantic.name}</h3>
            <p>{f.status} · {new Date(f.exportedAt).toLocaleString()}</p>
            <button aria-pressed={side === version} onClick={() => onNavigate(version)}>View {version}</button></section>;
        })}
      </div>
      {result.modelNamesDiffer && <p className="graph-context-warning">Different model names. Check that these snapshots are intended to be compared.</p>}
      {result.capture ? <CaptureComparisonPanel before={before} after={after} result={result.capture} onNavigate={onNavigate} /> : <>
      {result.architectureChanged !== null && <p className="graph-panel-hint">Layers match by portable ID. Unmatched identities remain added/removed; a rename is never assumed.</p>}
      <section aria-label="Comparison summary" className="graph-comparison-summary">
        {result.architectureChanged === null ? <p>Both snapshots need canonical captures for this comparison. Use Before and After to inspect the available evidence.</p> : <>
        <p>{result.architectureChanged ? 'Architecture changes found' : 'No architecture changes'}</p>
        <p>{result.addedLayers.length} added layers · {result.removedLayers.length} removed · {result.changedLayers.length} changed</p>
        <p>{result.addedConnections.reduce((n, e) => n + e.count, 0)} added connections · {result.removedConnections.reduce((n, e) => n + e.count, 0)} removed</p>
        </>}
      </section>
      {result.uncertainMatches.length > 0 && <section aria-label="Uncertain matches"><h3>Uncertain matches</h3>
        <p className="graph-panel-hint">These unmatched layers have identical structure and parameters. They may be renamed or separate layers. They remain in the added/removed lists.</p>
        {result.uncertainMatches.slice(0, limit).map((group, i) => <p className="graph-context-warning" key={i}>Before: {group.beforeIds.slice(0, 10).join(', ')}{group.beforeIds.length > 10 ? ` (+${group.beforeIds.length - 10})` : ''}<br />
          After: {group.afterIds.slice(0, 10).join(', ')}{group.afterIds.length > 10 ? ` (+${group.afterIds.length - 10})` : ''}</p>)}
      </section>}
      {result.addedLayers.length > 0 && <section aria-label="Added layers"><h3>Added layers ({result.addedLayers.length})</h3><ul className="graph-comparison-list">{result.addedLayers.slice(0, limit).map(node => <li key={node.node_id}><strong>{node.node_id}</strong><span>{node.op}</span>{nodeButton('after', node.node_id)}</li>)}</ul></section>}
      {result.removedLayers.length > 0 && <section aria-label="Removed layers"><h3>Removed layers ({result.removedLayers.length})</h3><ul className="graph-comparison-list">{result.removedLayers.slice(0, limit).map(node => <li key={node.node_id}><strong>{node.node_id}</strong><span>{node.op}</span>{nodeButton('before', node.node_id)}</li>)}</ul></section>}
      {result.changedLayers.length > 0 && <section aria-label="Changed layers"><h3>Changed layers ({result.changedLayers.length})</h3>{result.changedLayers.slice(0, limit).map(node => <article key={node.nodeId} aria-label={`Changed layer ${node.nodeId}`} className="graph-comparison-item">
        <h4>{node.nodeId}</h4><div className="graph-panel-actions">{nodeButton('before', node.nodeId)}{nodeButton('after', node.nodeId)}</div>
        <Fields changes={node.changes} />
      </article>)}</section>}
      {(['added', 'removed'] as const).map(kind => { const edges = kind === 'added' ? result.addedConnections : result.removedConnections; const version = kind === 'added' ? 'after' : 'before';
        return edges.length > 0 && <section key={kind} aria-label={`${kind === 'added' ? 'Added' : 'Removed'} connections`}><h3>{kind === 'added' ? 'Added' : 'Removed'} connections</h3>
          <ul className="graph-comparison-list">{edges.slice(0, limit).map((edge, i) => { const ambiguous = (version === 'after' ? edge.afterCount : edge.beforeCount) !== 1;
            return <li key={i}><strong>{targetLabel(edge.target)}</strong><span>{edge.count} {kind} · {edge.beforeCount} → {edge.afterCount} copies</span>
              <button disabled={ambiguous} onClick={() => onNavigate(version, edge.target)}>Show {version}</button>
              {ambiguous && <small>Multiple connections share these endpoints; no unique edge to highlight.</small>}</li>;
          })}</ul></section>;
      })}
      {(result.recordedShapes.length > 0 || result.snapshotEvidenceChanged) && <section aria-label="Recorded evidence"><h3>Recorded evidence</h3>
        <p className="graph-panel-hint">Shapes and trace availability can differ with inputs or recording conditions. They are not counted as architecture changes.</p>
        <p>Status: {before.status} → {after.status} · Resolution warnings: {before.warningCount} → {after.warningCount}</p>
        {result.recordedShapes.slice(0, limit).map(node => <article key={node.nodeId} className="graph-comparison-item"><h4>{node.nodeId}</h4><Fields changes={node.changes} absent="Not recorded" /></article>)}
      </section>}
      {[result.addedLayers.length, result.removedLayers.length, result.changedLayers.length, result.addedConnections.length, result.removedConnections.length, result.recordedShapes.length, result.uncertainMatches.length].some(count => count > limit)
        && <button onClick={() => setLimit(limit + 100)}>Show more changes (first {limit} per section shown)</button>}
      </>}
    </>}
  </aside>;
}
