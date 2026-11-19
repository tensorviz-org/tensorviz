import React from 'react';
import { ArrowDownToLine, ArrowRight, ArrowUpFromLine, FileText, LoaderCircle } from 'lucide-react';
import type { ModelEdge, ModelNode } from '../types';
import type { ModelExplanationState, NodeExplanationCitation, NodeExplanationState, NodeSourceDocumentation } from '../ide/nodeExplanation.generated';
import { sourceParameterValues } from '../ide/cardParams';
import { NodeExplanationContext, sameNodeExplanationResult } from './NodeExplanationContext';
import { NodeAIExplanation } from './NodeAIExplanation';
import { nodeEvidenceLabel, nodeInputOutput, nodePresentation, sourceDocumentationSummary } from './nodePresentation';
import { NodeSourceLink } from './NodeSourceControls';
import { customCardModel } from './customCard';
import { moduleContents } from '../services/modulePresentation';
import { tensorEvidence } from './graphSemantics';
import './nodeUnderstanding.css';

function InputOutput({ inputs, outputs, label, description }: { inputs: string[]; outputs: string[]; label: string; description?: string }) {
  const descriptionId = React.useId();
  if (!inputs.length && !outputs.length) return null;
  return <div className="node-understanding-io-group">
    {description && <span className="node-understanding-attribution" id={descriptionId}>{description}</span>}
    <dl className="node-understanding-io" aria-label={label} aria-describedby={description ? descriptionId : undefined}>
      {inputs.length > 0 && <div><dt>Inputs</dt><dd>{inputs.join(' · ')}</dd></div>}
      {inputs.length > 0 && outputs.length > 0 && <span aria-hidden="true">→</span>}
      {outputs.length > 0 && <div><dt>Outputs</dt><dd>{outputs.join(' · ')}</dd></div>}
    </dl>
  </div>;
}

const format = (value: unknown): string => typeof value === 'string' ? value : JSON.stringify(value) ?? String(value);
const wrappedCode = (value: string) => value.split(/([._])/).map((part, index) => <React.Fragment key={index}>
  {part}{(part === '.' || part === '_') && <wbr />}
</React.Fragment>);

const DOCUMENTATION_REASONS: Record<Extract<NodeSourceDocumentation, { status: 'unavailable' }>['reason'], string> = {
  ambiguous: 'Several source definitions match this call.',
  dynamic_binding: 'This callable is selected at runtime.',
  decorated: 'A decorator may change this callable.',
  unresolved_import: 'The imported definition is unavailable in the current source.',
  unresolved_symbol: 'The callable definition was not resolved.',
  unsupported_callable: 'Source documentation is unavailable for this callable.',
  analysis_limit: 'Source analysis reached its limit.',
};

function DocumentationCitation({ state, citation, compact = false }: { state: NodeExplanationState; citation: NodeExplanationCitation; compact?: boolean }) {
  const context = React.useContext(NodeExplanationContext);
  const lines = `:${citation.startLine}${citation.endLine !== citation.startLine ? `–${citation.endLine}` : ''}`;
  const label = `${citation.file}${lines}`;
  return <div className="node-understanding-sources"><button type="button" title={label} aria-label={label}
    onClick={() => context?.reveal(state, citation.id)}>{compact ? `${citation.file.split(/[\\/]/).pop()}${lines}` : label}</button></div>;
}

/** Keep scroll keys in the source excerpt rather than moving the selected graph. */
function sourceScrollKey(event: React.KeyboardEvent) {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) event.stopPropagation();
}

/** A selected node's overview; interpretation and source/capture facts stay distinct. */
export function NodeUnderstanding({ node, nodes = [], edges = [], onNavigate, detailsLabel = 'More details', hideActions = false }: {
  node: ModelNode;
  nodes?: readonly ModelNode[];
  edges?: readonly ModelEdge[];
  onNavigate?(nodeId: string): void;
  detailsLabel?: string;
  hideActions?: boolean;
  /** Compatibility only: display attribution comes from verified context provenance. */
  explanationExecution?: ModelExplanationState['execution'];
}) {
  const context = React.useContext(NodeExplanationContext);
  const state = context?.states[node.id];
  const retained = context?.annotations?.[node.id];
  // Interpretation and current source documentation have independent request/citation identities.
  const interpretationState = state?.status === 'ready' && state.explanation ? state
    : retained?.status === 'ready' && retained.nodeId === node.id && (!state || retained.revision === state.revision) ? retained : undefined;
  const explanation = interpretationState?.explanation;
  const receipt = context?.verifiedResults?.[node.id];
  // A pass summary (or an action prop) alone cannot attribute the visible reading.
  const execution = receipt && (sameNodeExplanationResult(interpretationState, receipt.result)
    || sameNodeExplanationResult(interpretationState, receipt.adoption)) ? receipt.execution : undefined;
  const documentation = state?.documentation;
  const source = documentation?.status === 'resolved' ? documentation : undefined;
  const sourceSummary = sourceDocumentationSummary(source?.docstring);
  const facts = nodePresentation(node.data, node.id);
  const custom = customCardModel(node.data.layerType?.id, node.data.parameters);
  const contents = node.type === 'moduleNode' ? moduleContents(node.data) : undefined;
  const io = nodeInputOutput(node.data);
  const evidence = tensorEvidence(node.data);
  const ioDescription = evidence === 'measured' ? 'Names · recorded shapes'
    : evidence === 'inferred' ? 'Names · inferred shapes' : evidence === 'stale' ? 'Names · previous shapes'
      : node.data.debugEvidence === 'source' || node.data.sourceLocation || facts.rawCall ? 'Source names' : 'Recorded names';
  const params = Object.entries(sourceParameterValues(node.data.parameters ?? {})).filter(([name]) => !name.startsWith('_'));
  const loading = !hideActions && (state?.status === 'queued' || state?.status === 'loading');
  const connected = (direction: 'in' | 'out') => [...new Set(edges.filter(edge => direction === 'in'
    ? edge.target === node.id : edge.source === node.id).map(edge => direction === 'in' ? edge.source : edge.target))]
    .map(id => nodes.find(peer => peer.id === id)).filter((peer): peer is ModelNode => !!peer);
  const upstream = connected('in'), downstream = connected('out');
  const peers = (items: ModelNode[], direction: 'in' | 'out') => items.map(peer => {
    // Wiring IDs never become a visible fallback identity.
    const identity = nodePresentation(peer.data, '');
    const params = peer.data.parameters;
    // Keep each file paired with its own line; declaration and call-site evidence differ.
    const location = [peer.data.sourceLocation,
      { file: params?._source_file, line: params?._source_line ?? params?._lineno },
      { file: params?._decl_file, line: params?._decl_line },
    ].find(candidate => typeof candidate?.file === 'string' && candidate.file.trim()
      && typeof candidate.line === 'number' && Number.isInteger(candidate.line) && candidate.line > 0);
    const technical = [...new Set([identity.callable, identity.path].filter((value): value is string => !!value))];
    return <button type="button" key={peer.id} className="node-understanding-connection" disabled={!onNavigate}
      aria-label={`Show in graph: ${direction === 'in' ? 'Incoming from' : 'Outgoing to'} ${identity.path || identity.callable || identity.title}`}
      title={identity.rawCall}
      onClick={() => onNavigate?.(peer.id)}>
      <span className="node-understanding-connection-body">
        <span className="node-understanding-connection-title">{identity.title}</span>
        {technical.filter(value => value !== identity.title).map(value => <code key={value}>{wrappedCode(value)}</code>)}
        {location && <span className="node-understanding-connection-location">{`${location.file}:${location.line}`}</span>}
      </span>
      <ArrowRight size={12} className="node-understanding-connection-action" aria-hidden="true" />
    </button>;
  });
  const unavailableNotice = !hideActions && state?.status === 'unavailable' && state.detail && <div className="node-understanding-unavailable">
    <span title={state.detail}>{state.detail}</span>
    {!hideActions && context && <button type="button" onClick={() => context.retry(state)}>Retry explanation</button>}
  </div>;
  return <div className="node-understanding" aria-label="About this node">
    <NodeAIExplanation result={interpretationState} execution={execution}
      onReveal={citationId => { if (interpretationState) context?.reveal(interpretationState, citationId); }} />
    <div className="node-understanding-evidence">
      <span>{nodeEvidenceLabel(node.data)}</span>
    </div>
    {loading && <span className="node-understanding-attribution" role="status"
      title="Reading source and preparing an explanation; this does not run the model or record tensor shapes.">
      <LoaderCircle size={13} className="node-understanding-loading" aria-hidden="true" />Reading component
    </span>}
    {facts.callable && facts.callable !== (explanation?.title ?? facts.title)
      && <code className="node-understanding-callable" title={facts.callable}>{wrappedCode(facts.callable)}</code>}
    <NodeSourceLink nodeId={node.id} data={node.data} />
    {!explanation && <>
      {sourceSummary && source && state ? <div className="node-source-documentation" aria-label="Source documentation overview">
        <span className="node-understanding-attribution"><FileText size={12} aria-hidden="true" />Source documentation</span>
        <p className="node-source-summary">{sourceSummary}</p>
        <DocumentationCitation state={state} citation={source.citation} compact />
      </div> : facts.purpose && <p className="node-understanding-reference" title="Layer reference">{facts.purpose}</p>}
      <InputOutput {...io} description={ioDescription} label="Graph inputs and outputs" />
    </>}
    {(upstream.length > 0 || downstream.length > 0) && <section className="node-understanding-connections" aria-label="Connected components">
      <span className="node-understanding-detail-label">Connected components · from graph</span>
      <div className="node-understanding-neighbors" aria-label="Upstream and downstream components">
        {upstream.length > 0 && <section aria-label="Incoming connections">
          <div className="node-understanding-flow-heading"><ArrowDownToLine size={14} aria-hidden="true" /><strong>Incoming</strong><span>To selected node</span></div>
          {peers(upstream, 'in')}
        </section>}
        {downstream.length > 0 && <section aria-label="Outgoing connections">
          <div className="node-understanding-flow-heading"><ArrowUpFromLine size={14} aria-hidden="true" /><strong>Outgoing</strong><span>From selected node</span></div>
          {peers(downstream, 'out')}
        </section>}
      </div>
    </section>}
    {!documentation && unavailableNotice}
    <details className="node-understanding-details" key={`${node.id}:${state?.revision ?? ''}`}>
      <summary>{detailsLabel}</summary>
      <div className="node-understanding-detail-body">
        {source && state && <section className="node-source-documentation" aria-label="Source documentation details">
          <span className="node-understanding-attribution"><FileText size={12} aria-hidden="true" />Source documentation</span>
          <code className="node-source-qualified-name">{source.qualifiedName}</code>
          <span className="node-understanding-detail-label">{source.kind === 'class' ? 'Class declaration' : 'Declared signature'}</span>
          <pre className="node-source-signature" tabIndex={0} role="region" aria-label={source.kind === 'class' ? 'Class declaration' : 'Declared signature'} onKeyDown={sourceScrollKey}>{source.signature}</pre>
          {source.docstring && <>
            <span className="node-understanding-detail-label">Docstring</span>
            <pre className="node-source-docstring" tabIndex={0} role="region" aria-label="Full source documentation" onKeyDown={sourceScrollKey}>{source.docstring}</pre>
          </>}
          {source.truncated && <span className="node-understanding-attribution">Source documentation was truncated.</span>}
          <DocumentationCitation state={state} citation={source.citation} />
        </section>}
        {documentation?.status === 'unavailable' && <div className="node-source-documentation-unavailable">
          <span className="node-understanding-detail-label">Source documentation</span><p>{DOCUMENTATION_REASONS[documentation.reason]}</p>
        </div>}
        <dl className="node-understanding-facts">
          <div><dt>Identity</dt><dd><code>{facts.callable || node.data.layerType.name}</code></dd></div>
          {facts.path && facts.path !== facts.callable
            && <div><dt>Graph path</dt><dd><code>{facts.path}</code></dd></div>}
        </dl>
        {contents?.detail && <p>{contents.detail}</p>}
        {(facts.rawCall || custom?.snippet) && <div><span className="node-understanding-detail-label">Source call</span><pre>{facts.rawCall || custom?.snippet}</pre></div>}
        {explanation && (io.inputs.length > 0 || io.outputs.length > 0) && <div><span className="node-understanding-detail-label">Graph evidence</span><InputOutput {...io} description={ioDescription} label="Graph inputs and outputs" /></div>}
        {params.length > 0 && <section aria-label="Parameters">
          <span className="node-understanding-detail-label">Parameters · {params.length}</span>
          <div className="node-understanding-parameters" role="region" aria-label="Parameter values" tabIndex={0} onKeyDown={sourceScrollKey}>
            <dl className="node-understanding-facts">
              {params.map(([name, value]) => <div key={name}><dt>{name}</dt><dd><code>{format(value)}</code></dd></div>)}
            </dl>
          </div>
        </section>}
        {custom?.validatable && custom.validatable.state !== 'unsupported' && <section aria-label="Validation">
          <span className="node-understanding-detail-label">Validation</span>
          <p>{custom.validatable.detail || (custom.validatable.state === 'ready' ? 'Ready to validate' : 'Cannot validate')}</p>
          {custom.validatable.remedy && <pre>{custom.validatable.remedy}</pre>}
        </section>}
        {(['_act_stats', '_grad_stats'] as const).map((key, index) => node.data.parameters?.[key]
          ? <section key={key}><span className="node-understanding-detail-label">{index === 0 ? 'Activation' : 'Gradient'} statistics</span><pre>{format(node.data.parameters[key])}</pre></section> : null)}
        {documentation && unavailableNotice}
      </div>
    </details>
  </div>;
}
