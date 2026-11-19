import React from 'react';
import { ChevronDown, FileText } from 'lucide-react';
import type { ModelExplanationState, NodeExplanationState } from '../ide/nodeExplanation.generated';
import './nodeAIExplanation.css';

/** Only a verified node result belongs here; operation feedback lives on Explain. */
export function NodeAIExplanation({ result, execution, onReveal }: {
  result?: NodeExplanationState;
  execution?: ModelExplanationState['execution'];
  onReveal(citationId: string): void;
}) {
  const heading = React.useId();
  const explanation = result?.status === 'ready' ? result.explanation : undefined;
  if (!explanation) return null;
  const metadata = execution ? [execution.provider === 'codex' ? 'Codex' : 'Claude', execution.model, execution.effort].filter(Boolean).join(' · ') : undefined;
  const hasDetails = explanation.steps.length > 0 || explanation.uncertainties.length > 0 || !!explanation.concepts?.length;
  return <section className="node-ai-explanation" aria-labelledby={heading}>
    <h3 id={heading}>AI explanation</h3>
    {metadata && <div className="node-ai-metadata">{metadata}</div>}
    <div className="node-ai-result">
      <p className="node-ai-purpose">{explanation.purpose}</p>
      {(explanation.inputs.length > 0 || explanation.outputs.length > 0) && <dl className="node-ai-io" aria-label="Interpreted inputs and outputs">
        {explanation.inputs.length > 0 && <div><dt>Inputs</dt><dd>{explanation.inputs.join(' · ')}</dd></div>}
        {explanation.outputs.length > 0 && <div><dt>Outputs</dt><dd>{explanation.outputs.join(' · ')}</dd></div>}
      </dl>}
      {hasDetails && <details className="node-ai-details">
        <summary>Explanation details<ChevronDown size={13} aria-hidden="true" /></summary>
        <div className="node-ai-detail-body">
          {explanation.steps.length > 0 && <section><h4>How it works</h4><ol aria-label="How it works">{explanation.steps.map((step, index) => <li key={index}>{step}</li>)}</ol></section>}
          {!!explanation.concepts?.length && <section aria-label="Related concepts"><h4>Related concepts</h4>
            {explanation.concepts.map((concept, index) => <details className="node-ai-concept" key={index}><summary>{concept.term}</summary><p>{concept.explanation}</p></details>)}
          </section>}
          {explanation.uncertainties.length > 0 && <section><h4>Uncertainties</h4><ul>{explanation.uncertainties.map((item, index) => <li key={index}>{item}</li>)}</ul></section>}
        </div>
      </details>}
      {explanation.citations.length > 0 && <div className="node-ai-sources" aria-label="Explanation sources">
        <span>Sources</span>{explanation.citations.map(citation => {
          const lines = `:${citation.startLine}${citation.endLine !== citation.startLine ? `–${citation.endLine}` : ''}`;
          return <button type="button" key={citation.id} aria-label={`${citation.file}${lines}`} title={`${citation.file}${lines}`}
            onClick={() => onReveal(citation.id)}><FileText size={11} aria-hidden="true" /><span>{citation.file.split(/[\\/]/).pop()}{lines}</span></button>;
        })}
      </div>}
      <p className="node-ai-attribution">AI-generated interpretation · verify against source</p>
    </div>
  </section>;
}
