import React, { useEffect, useState } from 'react';
import type { GraphFile } from '../services/graphFile';
import type { GraphTarget } from '../services/graphCollaboration';
import { capturedRecordTarget, capturedUseTarget } from '../services/capturedGraphComparison';
import type { CaptureRecordChange, CapturedUseChange, SemanticCaptureComparison } from '../ide/captureComparison.generated';
import ComparisonFields from './ComparisonFields';
import { CaptureDetails } from '../canvas/CaptureDetails';
import { createOfflineCaptureClient } from './offlineCaptureClient';
import type { LocalBrowserCaptureClient } from '../ide/localBrowserCaptureClient';

type Side = 'before' | 'after';
interface Props { before: GraphFile; after: GraphFile; result: SemanticCaptureComparison; onNavigate: (side: Side, target?: GraphTarget) => void }
export default function CaptureComparisonPanel({ before, after, result, onNavigate }: Props): React.ReactElement {
  const [limit, setLimit] = useState(25);
  const [clients, setClients] = useState<Record<Side, LocalBrowserCaptureClient> | null>(null);
  useEffect(() => {
    const clients = { before: createOfflineCaptureClient(before.capture!.capture), after: createOfflineCaptureClient(after.capture!.capture) };
    setClients(clients); setLimit(25);
    return () => { clients.before.dispose(); clients.after.dispose(); };
  }, [before.capture, after.capture]);
  const file = (side: Side) => side === 'before' ? before : after;
  const inspect = (side: Side, kind: CaptureRecordChange['kind'], id: string) => {
    const snapshot = file(side), capture = snapshot.capture!.capture;
    const query = kind === 'capture' ? { path: [] } : kind === 'module'
      ? { path: ['modules', capture.modules.findIndex(module => module.id === id)] }
      : { entity: { kind, id }, path: [] };
    return clients && <CaptureDetails reference={{ version: 1, captureId: capture.id, sourceId: capture.source.id, kind: 'input', semanticId: capture.id }}
      label={`${side === 'before' ? 'Before' : 'After'} · ${label(side, kind, id)}`} client={clients[side]} stale={snapshot.status === 'stale'}
      initialQuery={query} buttonLabel={`Inspect ${side}`} />;
  };
  const label = (side: Side, kind: CaptureRecordChange['kind'], id: string) => {
    if (kind === 'capture') return 'Whole capture';
    const snapshot = file(side);
    if (kind === 'module') {
      const paths = snapshot.capture!.capture.modules.find(module => module.id === id)?.paths;
      if (paths?.length) return `Module ${paths.slice(0, 2).map(path => path || '(root)').join(' / ')}`.slice(0, 200);
    }
    const target = capturedRecordTarget(snapshot, kind, id);
    const node = target && snapshot.document.semantic.nodes.find(node => node.node_id === target.nodeId);
    const name = kind === 'value' ? `Tensor ${id}` : String(node?.metadata.captureLabel ?? node?.op ?? id);
    return name.slice(0, 200);
  };
  const button = (side: Side, target: GraphTarget | undefined) => <button disabled={!target} onClick={() => target && onNavigate(side, target)}>Show {side}</button>;
  const recordButtons = (record: CaptureRecordChange) => <div className="graph-panel-actions">
    {button('before', capturedRecordTarget(before, record.kind, record.beforeId))}
    {button('after', capturedRecordTarget(after, record.kind, record.afterId))}
    {inspect('before', record.kind, record.beforeId)}{inspect('after', record.kind, record.afterId)}
  </div>;
  const records = (items: CaptureRecordChange[], title: string, hint: string, unknown = false) => items.length > 0 && <section aria-label={title}>
    <h3>{title} ({items.length})</h3><p className="graph-panel-hint">{hint}</p>
    {items.slice(0, limit).map((record, index) => <article className="graph-comparison-item" key={`${record.kind}:${record.beforeId}:${index}`}
      aria-label={`${title}: ${label('after', record.kind, record.afterId)}`}>
      <h4>{label('after', record.kind, record.afterId)}</h4><p className="graph-panel-hint">{record.kind} · {record.beforeId} → {record.afterId}</p>{recordButtons(record)}
      {unknown ? <p>Unresolved fields: {record.fields.map(field => field.field).join(', ')}</p>
        : <details><summary>{record.fields.map(field => field.field).join(', ')}</summary><ComparisonFields changes={record.fields} previewLimit={4000} absent="Not recorded" /></details>}
    </article>)}
  </section>;
  const connections = (items: CapturedUseChange[], unknown = false) => items.length > 0 && <section aria-label={unknown ? 'Unresolved connections' : 'Changed tensor uses'}>
    <h3>{unknown ? 'Unresolved connections' : 'Changed tensor uses'} ({items.length})</h3>
    <p className="graph-panel-hint">{unknown ? 'These uses lack a confident counterpart. Absence does not prove a removed connection.' : 'Each argument position is separate, including repeated references to the same tensor.'}</p>
    {items.slice(0, limit).map((change, index) => <article className="graph-comparison-item" key={index}>
      {(['before', 'after'] as const).map(side => {
        const use = change[side];
        return <div key={side}><strong>{side === 'before' ? 'Before' : 'After'}</strong>{use
          ? <><p><code>{use.entity.id} · {JSON.stringify(use.path)} · {use.valueId}</code></p>{button(side, capturedUseTarget(file(side), use))}</>
          : <p>{unknown ? 'No confident counterpart' : 'No tensor use at this position'}</p>}</div>;
      })}
    </article>)}
  </section>;
  const unmatched = (side: Side) => Object.entries(result.correspondence.unmatched[side]).flatMap(([kind, ids]) =>
    ids.map(id => ({ kind: kind as CaptureRecordChange['kind'], id })));
  const beforeUnmatched = unmatched('before'), afterUnmatched = unmatched('after');
  return <>
    <section aria-label="Comparison summary" className="graph-comparison-summary">
      <p>{result.equivalent === true ? 'Captured computation matches' : result.equivalent === false ? 'Captured computation differs' : 'Comparison is incomplete'}</p>
      <p>{result.changes.length} changed records · {result.valueUses.changes.length} changed tensor uses</p>
      <p>{result.correspondence.pairs.filter(pair => pair.kind === 'operation').length} matched operations · {result.correspondence.pairs.filter(pair => pair.kind === 'invocation').length} matched calls</p>
      <p className="graph-panel-hint">This compares captured computation. It does not prove numerical equality or behavior for other inputs. Layout and notes are excluded.</p>
      {!result.coverageComplete && <p className="graph-context-warning">At least one capture is incomplete. Missing records do not prove that a layer was removed.</p>}
    </section>
    <p className="graph-panel-hint">Before/After follows a selected record when its counterpart is known; otherwise it restores that version's own view. Repeated calls and numbered containers can remain unmatched; their positions do not establish identity.</p>
    {records(result.changes, 'Computation changes', 'Ordered arguments, outputs, effects, ownership, sharing and logical versions are compared exactly.')}
    {connections(result.valueUses.changes)}
    {records(result.observations, 'Recorded observations', 'Shapes, dtype, device, mode, outcomes and coverage can differ with execution conditions.')}
    <section aria-label="Execution conditions"><h3>Execution conditions</h3>
      <p className="graph-panel-hint">Matching these recorded controls does not establish identical custom attributes, randomness or external state.</p>
      <div className="graph-panel-actions">{inspect('before', 'capture', before.capture!.capture.id)}{inspect('after', 'capture', after.capture!.capture.id)}</div>
      {result.controls.map(control => <details key={control.field}><summary>{control.field}: {control.status}</summary>
        <ComparisonFields changes={[{ field: control.field, before: { present: true, value: control.before }, after: { present: true, value: control.after } }]} previewLimit={4000} />
      </details>)}
    </section>
    {records(result.unresolved, 'Unresolved fields', 'These fields cannot be compared confidently. Inspect each retained record separately.', true)}
    {(beforeUnmatched.length > 0 || afterUnmatched.length > 0) && <section aria-label="Unmatched records"><h3>Unmatched records</h3>
      <p className="graph-panel-hint">Unmatched does not mean added or removed. {result.correspondence.uncertain.length} candidate groups have uncertain identity.</p>
      {(['before', 'after'] as const).map(side => {
        const items = side === 'before' ? beforeUnmatched : afterUnmatched;
        return <div key={side}><h4>{side === 'before' ? 'Before' : 'After'} ({items.length})</h4><ul className="graph-comparison-list">
          {items.slice(0, limit).map(item => <li key={`${item.kind}:${item.id}`}><strong>{label(side, item.kind, item.id)}</strong><span>{item.kind} · {item.id}</span>
            <div className="graph-panel-actions">{button(side, capturedRecordTarget(file(side), item.kind, item.id))}{inspect(side, item.kind, item.id)}</div></li>)}
        </ul></div>;
      })}
    </section>}
    {connections(result.valueUses.unresolved, true)}
    {[result.changes.length, result.observations.length, result.unresolved.length, result.valueUses.changes.length,
      result.valueUses.unresolved.length, beforeUnmatched.length, afterUnmatched.length].some(count => count > limit)
      && <button onClick={() => setLimit(limit + 25)}>Show more records (first {limit} per section shown)</button>}
  </>;
}
