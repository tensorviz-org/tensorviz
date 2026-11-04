import React from 'react';
import { Check, Circle, CircleDot } from 'lucide-react';
import type { GraphOpeningState } from './protocol.generated';
import { graphSetupWorkflow } from './graphSetupWorkflow.generated';
import './GraphSetupProgress.css';

/** Progress comes from host checks, never from the existence of an edited draft. */
export function GraphSetupProgress({ state }: { state: GraphOpeningState }) {
  const { stages } = graphSetupWorkflow(state);
  const pythonReady = stages.python === 'ready';
  const pythonPending = stages.python === 'pending';
  const inputsReady = stages.inputs === 'saved';
  const steps = [
    { label: 'Python', complete: pythonReady, current: pythonPending },
    { label: 'Model inputs', complete: inputsReady, current: !pythonPending && !inputsReady },
  ];
  return <ol className="graph-setup-progress" aria-label="Project setup progress">
    {steps.map(({ label, complete, current }) => {
      const Icon = complete ? Check : current ? CircleDot : Circle;
      return <li key={label} className={complete ? 'complete' : current ? 'current' : ''}
        aria-current={current ? 'step' : undefined}>
        <Icon size={15} aria-hidden="true" /><span>{label}</span>
        <span className="graph-setup-sr-only">{complete ? ' complete' : current ? ' in progress' : ' pending'}</span>
      </li>;
    })}
  </ol>;
}
