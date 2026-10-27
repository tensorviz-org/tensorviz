import React from 'react';
import { GraphFile } from '../services/graphFile';
import { anchorStatus, canonical, edgeTarget, GraphAnchor, GraphTarget, targetLabel } from '../services/graphCollaboration';

export interface DiscussionDraft { anchor: GraphAnchor; threadId?: string; author: string; body: string }
interface Props {
  file: GraphFile; selected: GraphTarget | null; draft: DiscussionDraft | null;
  onDraft: (draft: DiscussionDraft | null) => void;
  onStart: (target: GraphTarget, threadId?: string) => void;
  onSubmit: () => void; onResolve: (id: string, resolved: boolean) => void;
  onFocus: (target: GraphTarget) => void; onClose: () => void;
}
const contextLabels = { unchanged: 'Context unchanged', changed: 'Context changed', missing: 'Target missing', ambiguous: 'Target ambiguous' };
export default function DiscussionPanel({ file, selected, draft, onDraft, onStart, onSubmit, onResolve, onFocus, onClose }: Props): React.ReactElement {
  const connections = selected?.kind === 'node' ? file.document.semantic.edges.filter(e => e.source_node_id === selected.nodeId || e.target_node_id === selected.nodeId) : [];
  return <aside className="graph-viewer-panel" aria-label="Layer discussions">
    <div className="graph-panel-heading"><h2>Discussions</h2><button aria-label="Close discussions" onClick={onClose}>×</button></div>
    <p className="graph-panel-hint">Replies travel in the downloaded graph file. Share the updated file with your teammate; comments do not sync online.</p>
    {draft ? <section className="graph-discussion-draft" aria-label="Draft message">
      <h3>{draft.threadId ? 'Reply' : 'New discussion'}</h3><p className="graph-target-label">{targetLabel(draft.anchor.target)}</p>
      <label>Your display name<input maxLength={80} placeholder="Anonymous" value={draft.author} onChange={event => onDraft({ ...draft, author: event.target.value })} /></label>
      <small>Names are not verified.</small>
      <label>Message<textarea autoFocus rows={5} maxLength={4000} value={draft.body} placeholder="Ask a question or explain a decision…" onChange={event => onDraft({ ...draft, body: event.target.value })} /></label>
      <div className="graph-panel-actions"><button disabled={!draft.body.trim()} onClick={onSubmit}>Add message</button><button onClick={() => onDraft(null)}>Discard draft</button></div>
    </section> : selected ? <section className="graph-discussion-new">
      <p className="graph-target-label">Selected: {targetLabel(selected)}</p><button onClick={() => onStart(selected)}>Start discussion</button>
      {connections.length > 0 && <details><summary>Discuss a connected edge</summary><ul className="graph-connection-list">{connections.map(edge => {
        const target = edgeTarget(edge);
        return <li key={edge.edge_id}><button onClick={() => { onFocus(target); onStart(target); }}>{targetLabel(target)}</button></li>;
      })}</ul></details>}
    </section> : <p>Select a layer or an expanded connection to start a discussion.</p>}
    <div className="graph-discussion-list">{file.collaboration.threads.length === 0 && <p className="graph-panel-hint">No discussions in this file yet.</p>}
      {file.collaboration.threads.map(thread => {
        const status = anchorStatus(file, thread.anchor);
        const matchesSelection = selected && canonical(selected) === canonical(thread.anchor.target);
        return <article key={thread.id} className={`graph-discussion${matchesSelection ? ' is-selected' : ''}`} aria-label={`Discussion on ${targetLabel(thread.anchor.target)}`}>
          <div className="graph-panel-heading"><h3>{targetLabel(thread.anchor.target)}</h3><span>{thread.resolved ? 'Resolved' : 'Open'}</span></div>
          <p className={status === 'unchanged' ? 'graph-panel-hint' : 'graph-context-warning'}>{contextLabels[status]}</p>
          <small>Original snapshot: {new Date(thread.anchor.exportedAt).toLocaleString()}</small>
          {status !== 'unchanged' && <p className="graph-panel-hint">This discussion keeps its original context. Review it against the current architecture before relying on it.</p>}
          <div className="graph-panel-actions"><button disabled={status === 'missing' || status === 'ambiguous'} onClick={() => onFocus(thread.anchor.target)}>Show current target</button><button onClick={() => onResolve(thread.id, !thread.resolved)}>{thread.resolved ? 'Reopen discussion' : 'Resolve discussion'}</button></div>
          <ol className="graph-messages">{thread.messages.map(message => <li key={message.id}>
            <div><strong>{message.author}</strong><time dateTime={message.createdAt}>{new Date(message.createdAt).toLocaleString()}</time></div>
            <p className="graph-note">{message.body}</p>
          </li>)}</ol>
          <button disabled={Boolean(draft)} onClick={() => onStart(thread.anchor.target, thread.id)}>Reply</button>
          {status !== 'unchanged' && <details><summary>Original target details</summary><pre className="graph-original-context">{JSON.stringify(JSON.parse(thread.anchor.context), null, 2)}</pre></details>}
        </article>;
      })}
    </div>
  </aside>;
}
