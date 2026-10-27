/** What the graph guessed, offered for confirmation.
 *
 * The inferred spine reasons from kinds — there is a dataset, there is a model,
 * datasets feed models — and on a hand-written training loop that is the only thing
 * standing between the user and an empty pipeline. It is also, unavoidably, a guess.
 *
 * Rather than leave guesses as quiet assertions the user must learn to distrust,
 * this lists them and offers to make each one true: confirming writes
 * `tensorviz.edge(...)` into the entrypoint through the ordinary review path, and
 * the next index reads it as `declared`, which outranks every rule. The guess stops
 * being recomputed forever and becomes source text — reviewable, correctable,
 * committed alongside the code it describes.
 *
 * A list rather than an affordance on each edge, because the useful gesture is
 * confirming several at once, and because that is the shape a model writing them in
 * bulk will use.
 *
 * This proposes. It never writes: the host turns a confirmation into a native diff
 * the user accepts or discards.
 */

import React from 'react';

import './ProjectSuggestions.css';

export interface InferredEdge {
  id: string;
  from: string;
  to: string;
  relationship: string;
}

interface ProjectSuggestionsProps {
  edges: InferredEdge[];
  onDeclare(edge: InferredEdge): void;
  /** Ids already sent. Kept so a second click cannot queue a duplicate proposal. */
  pending?: readonly string[];
}

export function ProjectSuggestions({ edges, onDeclare, pending = [] }: ProjectSuggestionsProps) {
  const [open, setOpen] = React.useState(false);

  if (edges.length === 0) {
    // Nothing guessed means the graph was read rather than reasoned about, which is
    // the good case and deserves no chrome at all.
    return null;
  }

  const waiting = new Set(pending);

  return (
    <section className="tv-guess">
      <button
        type="button"
        className="tv-guess__disclosure"
        aria-expanded={open}
        onClick={() => setOpen((was) => !was)}
      >
        {open ? '▾' : '▸'} {edges.length} guessed connection{edges.length === 1 ? '' : 's'}
      </button>

      {open && (
        <>
          <p className="tv-guess__why">
            Nothing in the source states these. Confirming one writes it into the
            entrypoint, so it stops being a guess.
          </p>
          <ul className="tv-guess__list">
            {edges.map((edge) => (
              <li key={edge.id} className="tv-guess__item">
                <span className="tv-guess__edge">
                  <span className="tv-guess__node">{edge.from}</span>
                  <span className="tv-guess__rel">{edge.relationship}</span>
                  <span className="tv-guess__node">{edge.to}</span>
                </span>
                <button
                  type="button"
                  className="tv-guess__confirm"
                  disabled={waiting.has(edge.id)}
                  onClick={() => onDeclare(edge)}
                  title={`Write tensorviz.edge('${edge.from}', '${edge.to}', '${edge.relationship}')`}
                >
                  {waiting.has(edge.id) ? 'proposed' : 'Declare'}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

/** The inferred edges of a project document, in the shape this panel takes.
 *
 * Only edges: an inferred *node* is a classification the user corrects by editing
 * their own code, while an inferred *edge* is a relationship nothing in the source
 * states and which a declaration is exactly the right way to settle.
 */
export function inferredEdgesOf(document: unknown): InferredEdge[] {
  const doc = document as
    | { nodes?: Array<{ id: string; data?: { label?: string } }>; edges?: Array<Record<string, unknown>> }
    | null;
  if (!doc?.edges || !doc.nodes) {
    return [];
  }
  const labels = new Map(doc.nodes.map((node) => [node.id, node.data?.label ?? node.id]));
  return doc.edges
    .filter((edge) => (edge.data as { _provenance?: string } | undefined)?._provenance === 'inferred')
    .map((edge) => ({
      id: String(edge.id),
      from: labels.get(String(edge.source)) ?? String(edge.source),
      to: labels.get(String(edge.target)) ?? String(edge.target),
      relationship: String(edge.label ?? 'feeds'),
    }));
}
