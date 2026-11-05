import React from 'react';
import { Sparkles } from 'lucide-react';
import type { NodeExplanation } from '../ide/nodeExplanation.generated';
import { nodeAnnotationLayout } from './nodeExplanationPresentation';
import './nodeExplanationAnnotation.css';

/** Interpretation stays separate from source identity and captured values. */
export function NodeExplanationAnnotation({ explanation, width }: { explanation?: NodeExplanation; width: number }) {
  if (!explanation) return null;
  const layout = nodeAnnotationLayout(explanation, width);
  const lines = (values: string[]) => values.map((line, index) => <span key={index}>{line}</span>);
  return <section className="node-explanation-annotation" aria-label="Agent interpretation" style={{ height: layout.height }} title={explanation.purpose}>
    <div className="node-explanation-role"><Sparkles size={11} aria-hidden="true" /><strong>{lines(layout.titleLines)}</strong></div>
    {layout.input && <div className="node-explanation-flow" title={explanation.inputs.join(' · ')}><span>In</span><span>{lines(layout.inputLines)}</span></div>}
    {layout.output && <div className="node-explanation-flow" title={explanation.outputs.join(' · ')}><span>Out</span><span>{lines(layout.outputLines)}</span></div>}
  </section>;
}
