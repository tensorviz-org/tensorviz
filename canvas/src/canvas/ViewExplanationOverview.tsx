import React from 'react';
import { ChevronDown, FileCode2, Sparkles } from 'lucide-react';
import type { ModelNode } from '../types';
import type { NodeExplanationCitation } from '../ide/nodeExplanation.generated';
import { NodeExplanationContext } from './NodeExplanationContext';
import { nodePresentation } from './nodePresentation';
import { applicableViewExplanation, viewCitationState } from './viewExplanationPresentation';
import './viewExplanationOverview.css';

/** Source-linked commentary uses the existing selection, never new model topology. */
export function ViewExplanationOverview({ nodes, focused, selectedNodeIds = [], onNavigate }: {
  nodes: readonly ModelNode[];
  focused: boolean;
  selectedNodeIds?: readonly string[];
  onNavigate(nodeId: string): void;
}) {
  const context = React.useContext(NodeExplanationContext);
  const state = context?.model?.state;
  const view = applicableViewExplanation(state, context?.annotationsVisible, nodes);
  const [expanded, setExpanded] = React.useState(!focused);
  const [pending, setPending] = React.useState<{ nodeId: string; citation: NodeExplanationCitation; requestId: string; previousNodeRequestId?: string } | null>(null);
  const bodyId = React.useId();
  React.useEffect(() => { if (focused) setExpanded(false); }, [focused]);
  React.useEffect(() => {
    if (!pending) return;
    if (!view || !context || state?.requestId !== pending.requestId) { setPending(null); return; }
    if (!selectedNodeIds.includes(pending.nodeId)) return;
    const match = viewCitationState(pending.citation, state.revision, [context.states], [pending.nodeId]);
    // Local selection precedes the host's debounced selection work. A retained
    // batch/previous-selection result is proof, but cannot authorize a reveal
    // for the new selection until the host issues its current request.
    if (match?.state.nodeId === pending.nodeId && match.state.requestId !== pending.previousNodeRequestId) {
      setPending(null); context.reveal(match.state, match.citationId);
    }
  }, [pending, state, view, context, selectedNodeIds]);
  if (!view || !state || !context) return null;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const callable = (node: ModelNode) => {
    const facts = nodePresentation(node.data, node.id);
    return facts.callable || facts.title;
  };
  const identity = (id: string) => {
    const node = byId.get(id)!;
    const name = callable(node), instance = node.data.moduleInstanceId;
    const ambiguous = view.nodeIds.some(other => other !== id && callable(byId.get(other)!) === name);
    return ambiguous && instance && instance !== name ? `${name} · ${instance}` : name;
  };
  const sources = (citations: readonly NodeExplanationCitation[], nodeIds: readonly string[]) => citations.map(citation => {
    const match = viewCitationState(citation, state.revision, [context.states, context.annotations ?? {}], nodeIds);
    if (!match) return null;
    const lines = citation.startLine === citation.endLine ? `${citation.startLine}` : `${citation.startLine}–${citation.endLine}`;
    const label = `${citation.file}:${lines}`;
    return <button type="button" key={citation.id} className="view-explanation-source" title={label} aria-label={`Open ${label}`}
      onClick={() => {
        if (selectedNodeIds.includes(match.state.nodeId) && context.states[match.state.nodeId] === match.state
          && match.state.requestId !== state.requestId) {
          context.reveal(match.state, match.citationId);
        } else {
          setPending({ nodeId: match.state.nodeId, citation, requestId: state.requestId,
            previousNodeRequestId: context.states[match.state.nodeId]?.requestId ?? match.state.requestId });
          onNavigate(match.state.nodeId);
        }
      }}><FileCode2 size={12} aria-hidden="true" /><span>{citation.file.split(/[\\/]/).pop()}:{lines}</span></button>;
  });
  return <section className="view-explanation-overview" aria-label="Explanation of this view">
    <button type="button" className="view-explanation-toggle" aria-expanded={expanded} aria-controls={bodyId}
      onClick={() => setExpanded(value => !value)}>
      <Sparkles size={13} aria-hidden="true" /><strong>This view</strong><span>{view.nodeIds.length} components</span>
      <ChevronDown size={13} aria-hidden="true" />
    </button>
    {expanded && <div id={bodyId} className="view-explanation-body">
      <p className="view-explanation-summary">{view.summary}</p>
      {view.callouts.map((callout, index) => <details className="view-explanation-callout" key={`${callout.title}-${index}`}>
        <summary><span>{callout.title}</span><ChevronDown size={13} aria-hidden="true" /></summary>
        <div className="view-explanation-callout-body">
          <p>{callout.body}</p>
          <div className="view-explanation-node-links" aria-label={`Components for ${callout.title}`}>
            {callout.nodeIds.map(id => <button type="button" key={id} title={identity(id)} aria-label={`Select ${identity(id)}`}
              onClick={() => onNavigate(id)}>{identity(id)}</button>)}
          </div>
          <div className="view-explanation-sources">{sources(callout.citations, callout.nodeIds)}</div>
        </div>
      </details>)}
      {view.connections.length > 0 && <details className="view-explanation-connections">
        <summary>Connection meanings</summary>
        <ol>{view.connections.map(connection => <li key={JSON.stringify([connection.sourceNodeId, connection.targetNodeId])}>
          <div className="view-explanation-connection-route"><button type="button" onClick={() => onNavigate(connection.sourceNodeId)}>{identity(connection.sourceNodeId)}</button>
            <span aria-hidden="true">→</span><button type="button" onClick={() => onNavigate(connection.targetNodeId)}>{identity(connection.targetNodeId)}</button></div>
          <strong>{connection.label}</strong>{connection.detail && <p>{connection.detail}</p>}
          <div className="view-explanation-sources">{sources(connection.citations, [connection.sourceNodeId, connection.targetNodeId])}</div>
        </li>)}</ol>
      </details>}
    </div>}
  </section>;
}
