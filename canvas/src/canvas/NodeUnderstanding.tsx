import React from 'react';
import { FileText, LoaderCircle, MessageCircle, Sparkles } from 'lucide-react';
import type { ModelNode } from '../types';
import type { NodeExplanationCitation, NodeExplanationState, NodeSourceDocumentation } from '../ide/nodeExplanation.generated';
import { sourceParameterValues } from '../ide/cardParams';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { NodeExplanationContext } from './NodeExplanationContext';
import { nodeEvidenceLabel, nodeInputOutput, nodePresentation, sourceDocumentationSummary } from './nodePresentation';
import { NodeSourceLink } from './NodeSourceControls';
import { customCardModel } from './customCard';
import './nodeUnderstanding.css';

function InputOutput({ inputs, outputs, label }: { inputs: string[]; outputs: string[]; label: string }) {
  return <dl className="node-understanding-io" aria-label={label}>
    <div><dt>In</dt><dd>{inputs.length ? inputs.join(' · ') : 'Not recorded'}</dd></div>
    <span aria-hidden="true">→</span>
    <div><dt>Out</dt><dd>{outputs.length ? outputs.join(' · ') : 'Not recorded'}</dd></div>
  </dl>;
}

const format = (value: unknown): string => typeof value === 'string' ? value : JSON.stringify(value) ?? String(value);

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
export function NodeUnderstanding({ node }: { node: ModelNode }) {
  const context = React.useContext(NodeExplanationContext);
  const local = React.useContext(LocalBrowserContext);
  const state = context?.states[node.id];
  const explanation = state?.status === 'ready' ? state.explanation : undefined;
  const documentation = state?.documentation;
  const source = documentation?.status === 'resolved' ? documentation : undefined;
  const sourceSummary = sourceDocumentationSummary(source?.docstring);
  const facts = nodePresentation(node.data, node.id);
  const custom = customCardModel(node.data.layerType?.id, node.data.parameters);
  const io = nodeInputOutput(node.data);
  const params = Object.entries(sourceParameterValues(node.data.parameters ?? {})).filter(([name]) => !name.startsWith('_'));
  const loading = state?.status === 'queued' || state?.status === 'loading';
  const unavailableNotice = state?.status === 'unavailable' && <div className="node-understanding-unavailable">
    <span title={state.detail}>{state.detail || 'Explanation unavailable.'}</span>
    {context && <button type="button" onClick={() => context.retry(state)}>Retry explanation</button>}
  </div>;
  return <div className="node-understanding" aria-label="About this node">
    <div className="node-understanding-evidence">
      <span>{nodeEvidenceLabel(node.data)}</span>
      {loading && <LoaderCircle size={13} className="node-understanding-loading" role="img" aria-label="Reading node source" />}
    </div>
    {facts.callable && facts.callable !== (explanation?.title ?? facts.title)
      && <code className="node-understanding-callable" title={facts.callable}>{facts.callable}</code>}
    <NodeSourceLink nodeId={node.id} data={node.data} />
    {explanation ? <div className="node-understanding-interpretation">
      <span className="node-understanding-attribution"><Sparkles size={12} aria-hidden="true" />Agent interpretation</span>
      <p>{explanation.purpose}</p>
      <InputOutput inputs={explanation.inputs} outputs={explanation.outputs} label="Interpreted inputs and outputs" />
    </div> : <>
      {sourceSummary && source && state ? <div className="node-source-documentation" aria-label="Source documentation overview">
        <span className="node-understanding-attribution"><FileText size={12} aria-hidden="true" />Source documentation</span>
        <p className="node-source-summary">{sourceSummary}</p>
        <DocumentationCitation state={state} citation={source.citation} compact />
      </div> : facts.purpose && <p className="node-understanding-reference" title="Layer reference">{facts.purpose}</p>}
      <InputOutput {...io} label="Graph inputs and outputs" />
    </>}
    {!documentation && unavailableNotice}
    <details className="node-understanding-details" key={`${node.id}:${state?.revision ?? ''}`}>
      <summary>More details</summary>
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
          {node.data.captureLabel && node.data.captureLabel !== facts.callable && !/^#?python@/.test(node.data.captureLabel)
            && <div><dt>Graph path</dt><dd><code>{node.data.captureLabel}</code></dd></div>}
        </dl>
        {(facts.rawCall || custom?.snippet) && <div><span className="node-understanding-detail-label">Source call</span><pre>{facts.rawCall || custom?.snippet}</pre></div>}
        {explanation && <div><span className="node-understanding-detail-label">Graph evidence</span><InputOutput {...io} label="Graph inputs and outputs" /></div>}
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
        {explanation && state && <div className="node-understanding-interpretation">
          <span className="node-understanding-attribution"><Sparkles size={12} aria-hidden="true" />Agent interpretation</span>
          {explanation.steps.length > 0 && <ol aria-label="How it works">{explanation.steps.map((step, index) => <li key={index}>{step}</li>)}</ol>}
          {explanation.uncertainties.length > 0 && <div><span className="node-understanding-detail-label">Uncertainties</span>
            <ul>{explanation.uncertainties.map((uncertainty, index) => <li key={index}>{uncertainty}</li>)}</ul></div>}
          <div className="node-understanding-sources" aria-label="Explanation sources">
            {explanation.citations.map(citation => <button type="button" key={citation.id} title={`${citation.file}:${citation.startLine}–${citation.endLine}`}
              onClick={() => context?.reveal(state, citation.id)}>{citation.file}:{citation.startLine}{citation.endLine !== citation.startLine ? `–${citation.endLine}` : ''}</button>)}
          </div>
        </div>}
      </div>
    </details>
    {context && local?.sourceAction !== 'copy' && <button type="button" className="node-understanding-discuss" onClick={() => {
      window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'report-selection' } }));
      window.dispatchEvent(new CustomEvent('tensorviz-agent-compose'));
    }}><MessageCircle size={13} aria-hidden="true" />Discuss in chat</button>}
  </div>;
}
