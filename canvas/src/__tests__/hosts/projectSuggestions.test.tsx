import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';

import { InferredEdge, ProjectSuggestions, inferredEdgesOf } from '../../ide/ProjectSuggestions';

const EDGES: InferredEdge[] = [
  { id: 'e1', from: 'make_loader', to: 'TinyLM', relationship: 'feeds' },
  { id: 'e2', from: 'LanguageModelTrainer', to: 'TinyLM', relationship: 'trains' },
];

describe('ProjectSuggestions', () => {
  it('shows nothing at all when the graph was read rather than guessed at', () => {
    // A project whose dataflow is entirely static deserves no chrome: there is
    // nothing to confirm.
    const { container } = render(<ProjectSuggestions edges={[]} onDeclare={() => undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('says how many guesses there are before it is opened', () => {
    render(<ProjectSuggestions edges={EDGES} onDeclare={() => undefined} />);
    expect(screen.getByRole('button', { name: /2 guessed connections/ })).toBeInTheDocument();
  });

  it('explains why it is asking', () => {
    render(<ProjectSuggestions edges={EDGES} onDeclare={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: /guessed connections/ }));
    expect(screen.getByText(/Nothing in the source states these/)).toBeInTheDocument();
  });

  it('reports the whole edge when one is confirmed, not just its id', () => {
    const declared: InferredEdge[] = [];
    render(<ProjectSuggestions edges={EDGES} onDeclare={(edge) => declared.push(edge)} />);
    fireEvent.click(screen.getByRole('button', { name: /guessed connections/ }));
    fireEvent.click(screen.getAllByRole('button', { name: 'Declare' })[0]);

    expect(declared).toEqual([EDGES[0]]);
  });

  it('shows the exact statement it would write, before the user commits', () => {
    render(<ProjectSuggestions edges={EDGES} onDeclare={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: /guessed connections/ }));
    expect(screen.getAllByRole('button', { name: 'Declare' })[0]).toHaveAttribute(
      'title',
      "Write tensorviz.edge('make_loader', 'TinyLM', 'feeds')",
    );
  });

  it('cannot queue the same proposal twice', () => {
    render(<ProjectSuggestions edges={EDGES} onDeclare={() => undefined} pending={['e1']} />);
    fireEvent.click(screen.getByRole('button', { name: /guessed connections/ }));
    expect(screen.getByRole('button', { name: 'proposed' })).toBeDisabled();
  });
});

describe('inferredEdgesOf', () => {
  const document = {
    nodes: [
      { id: 'n1', data: { label: 'make_loader' } },
      { id: 'n2', data: { label: 'TinyLM' } },
    ],
    edges: [
      { id: 'e1', source: 'n1', target: 'n2', label: 'feeds', data: { _provenance: 'inferred' } },
      { id: 'e2', source: 'n2', target: 'n1', label: 'trains', data: { _provenance: 'static' } },
    ],
  };

  it('offers only the guesses, never what was read from the code', () => {
    const edges = inferredEdgesOf(document);
    expect(edges).toHaveLength(1);
    expect(edges[0].id).toBe('e1');
  });

  it('resolves node ids to the labels a person recognises', () => {
    expect(inferredEdgesOf(document)[0]).toMatchObject({
      from: 'make_loader',
      to: 'TinyLM',
      relationship: 'feeds',
    });
  });

  it('survives a document that is missing or malformed', () => {
    expect(inferredEdgesOf(null)).toEqual([]);
    expect(inferredEdgesOf({})).toEqual([]);
    expect(inferredEdgesOf({ nodes: [], edges: [] })).toEqual([]);
  });
});
