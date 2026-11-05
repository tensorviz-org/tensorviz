import React from 'react';
import { LoaderCircle, MessageCircle, Sparkles } from 'lucide-react';
import type { ModelNode } from '../types';
import { cardParams } from '../ide/cardParams';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { NodeExplanationContext } from './NodeExplanationContext';
import { nodeEvidenceLabel, nodeInputOutput, nodePresentation } from './nodePresentation';
import { NodeSourceLink } from './NodeSourceControls';
import './nodeUnderstanding.css';

function InputOutput({ inputs, outputs, label }: { inputs: string[]; outputs: string[]; label: string }) {
  return <dl className="node-understanding-io" aria-label={label}>
    <div><dt>In</dt><dd>{inputs.length ? inputs.join(' · ') : 'Not recorded'}</dd></div>
    <span aria-hidden="true">→</span>
    <div><dt>Out</dt><dd>{outputs.length ? outputs.join(' · ') : 'Not recorded'}</dd></div>
  </dl>;
}

const format = (value: unknown): string => typeof value === 'string' ? value : JSON.stringify(value) ?? String(value);

/** A selected node's overview; interpretation and source/capture facts stay distinct. */
export function NodeUnderstanding({ node }: { node: ModelNode }) {
  const context = React.useContext(NodeExplanationContext);
  const local = React.useContext(LocalBrowserContext);
  const state = context?.states[node.id];
  const explanation = state?.status === 'ready' ? state.explanation : undefined;
  const facts = nodePresentation(node.data, node.id);
  const io = nodeInputOutput(node.data);
  const params = cardParams(node.data.parameters).entries;
  const loading = state?.status === 'queued' || state?.status === 'loading';
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
      {facts.purpose && <p className="node-understanding-reference" title="Layer reference">{facts.purpose}</p>}
      <InputOutput {...io} label="Graph inputs and outputs" />
    </>}
    {state?.status === 'unavailable' && <div className="node-understanding-unavailable">
      <span title={state.detail}>{state.detail || 'Explanation unavailable.'}</span>
      {context && <button type="button" onClick={() => context.retry(state)}>Retry explanation</button>}
    </div>}
    <details className="node-understanding-details" key={`${node.id}:${state?.revision ?? ''}`}>
      <summary>More details</summary>
      <div className="node-understanding-detail-body">
        <dl className="node-understanding-facts">
          <div><dt>Identity</dt><dd><code>{facts.callable || node.data.layerType.name}</code></dd></div>
          {node.data.captureLabel && node.data.captureLabel !== facts.callable && !/^#?python@/.test(node.data.captureLabel)
            && <div><dt>Graph path</dt><dd><code>{node.data.captureLabel}</code></dd></div>}
        </dl>
        {facts.rawCall && <div><span className="node-understanding-detail-label">Source call</span><pre>{facts.rawCall}</pre></div>}
        {explanation && <div><span className="node-understanding-detail-label">Graph evidence</span><InputOutput {...io} label="Graph inputs and outputs" /></div>}
        {params.length > 0 && <dl className="node-understanding-facts" aria-label="Parameters">
          {params.map(([name, value]) => <div key={name}><dt>{name}</dt><dd><code>{format(value)}</code></dd></div>)}
        </dl>}
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
