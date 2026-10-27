/** The experiment rail: what this project contains, and which part is on screen.
 *
 * The canvas draws one experiment at a time; this is how you see there are
 * others, and how you say which. It is built from the index alone — no graph —
 * so it stays cheap on a workspace with hundreds of files.
 *
 * Three things it refuses to hide, because each one is a lie by omission the
 * user could otherwise only discover by noticing something missing:
 *
 *  - **candidates.** A file whose *name* suggested an entrypoint is shown as
 *    needing confirmation rather than asserted as an experiment. The badge says
 *    so, because the alternative is a graph that quietly claims more than it knows.
 *  - **unreached files.** Files nothing imports still exist in the editor. They
 *    are listed, collapsed, so the project never looks smaller than it is.
 *  - **truncation.** If the source set was capped, the rail says how much it saw.
 */

import React from 'react';

import { ProjectExperiment, ProjectIndexSummary } from './localBrowserState';
import './ProjectRail.css';

interface ProjectRailProps {
  index: ProjectIndexSummary;
  focusedId: string;
  onFocus(experimentId: string): void;
  /** Files actually indexed, and files enumeration saw before any budget applied.
   *  Two numbers rather than one flag: "part of the project" is not something a
   *  reader can act on, and 592 of 596 is a rounding error where 200 of 4,000 is a
   *  different tool. */
  indexed?: number;
  discovered?: number;
  /** The experiment asked for whose document has not arrived yet, if any.
   *
   *  Clicking sends `projectFocus` and changes nothing on the page: the rail's
   *  highlight follows `page.project`, which only moves when the host answers. On a
   *  large project that is a long silence in which the only honest reading is that
   *  the click missed — so people click again, and again, and each click queues
   *  another document. */
  pendingId?: string | null;
}

function experimentTitle(experiment: ProjectExperiment): string {
  const where = experiment.root ? `${experiment.root}/${experiment.entrypoint}` : experiment.entrypoint;
  return experiment.status.declared
    ? `${where} — named by tensorviz.experiment()`
    : `${where} — discovered from the code`;
}

/** Does this experiment match what was typed?
 *
 *  Label, entrypoint and root, because those are the three things a person knows
 *  about an experiment they are looking for: what it is called, which file runs it,
 *  and which project it belongs to. Case-insensitive and substring, not fuzzy — a
 *  rail is a list of names, and fuzzy matching over 138 of them offers as many
 *  wrong answers as right ones.
 */
export function experimentMatches(experiment: ProjectExperiment, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return (
    experiment.label.toLowerCase().includes(q)
    || experiment.entrypoint.toLowerCase().includes(q)
    || experiment.root.toLowerCase().includes(q)
  );
}

export function ProjectRail({ index, focusedId, onFocus, indexed, discovered, pendingId }: ProjectRailProps) {
  const [showUnreached, setShowUnreached] = React.useState(false);
  /** What the user is looking for. `huggingface/trl` indexes to 138 experiments and
   *  really does ship that many recipes, so this is not a symptom of bad derivation
   *  to be fixed elsewhere — it is a list nobody can read without a filter. */
  const [query, setQuery] = React.useState('');

  // Group by derived project root, but only show the grouping when there is more
  // than one — a flat repository should not be made to look like a monorepo.
  const allRoots = Array.from(new Set(index.experiments.map((e) => e.root))).sort();
  const grouped = allRoots.length > 1;

  // Which project, when the workspace holds several. The same shape as the SCM
  // view's repository picker, and for the same reason: a monorepo is several
  // projects in one window, and reading all of them at once is reading none of
  // them. `''` means every root, which is the honest default — a workspace that
  // was never narrowed has not been narrowed.
  const [onlyRoot, setOnlyRoot] = React.useState('');
  const roots = onlyRoot ? allRoots.filter((root) => root === onlyRoot) : allRoots;

  const unparseable = index.unreached.filter((entry) => entry.error);
  const matching = index.experiments.filter((experiment) => experimentMatches(experiment, query));

  return (
    <nav className="tv-rail" aria-label="Experiments">
      <div className="tv-rail__head">
        <span className="tv-rail__count">
          {/* The count says what is SHOWN when a filter is on, with the total beside
              it — a filtered rail that reports the whole number is a rail that looks
              like it lost something. */}
          {query.trim()
            ? `${matching.length} of ${index.experiments.length} experiments`
            : `${index.experiments.length} experiment${index.experiments.length === 1 ? '' : 's'}`}
        </span>
        {grouped && (
          <label className="tv-rail__picker">
            <span className="tv-rail__picker-label">Project</span>
            <select
              value={onlyRoot}
              onChange={(event) => setOnlyRoot(event.target.value)}
              aria-label="Which project to show"
            >
              <option value="">All {allRoots.length} projects</option>
              {allRoots.map((root) => (
                <option key={root || '<root>'} value={root}>
                  {root || 'project root'}
                </option>
              ))}
            </select>
          </label>
        )}
        {index.truncated && (
          <span className="tv-rail__truncated" role="status">
            {indexed && discovered
              ? `indexed ${indexed.toLocaleString()} of ${discovered.toLocaleString()} files`
              : 'indexed part of the project'}
          </span>
        )}
      </div>

      {index.experiments.length > 8 && (
        <div className="tv-rail__filter">
          <input
            type="search"
            className="tv-rail__filter-input"
            value={query}
            placeholder="Filter experiments"
            aria-label="Filter experiments"
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
      )}

      {/* A group whose every experiment was filtered out is not drawn: an empty
          heading under a search reads as a project that lost its contents. */}
      {roots.filter((root) => matching.some((e) => e.root === root)).map((root) => (
        <section key={root || '<root>'} className="tv-rail__group">
          {grouped && <h3 className="tv-rail__root">{root || 'project root'}</h3>}
          <ul className="tv-rail__list">
            {matching
              .filter((experiment) => experiment.root === root)
              .map((experiment) => {
                const loading = experiment.experiment_id === pendingId;
                // The row the user asked for reads as chosen the moment they ask,
                // not when the answer lands — but it says it is still loading, so
                // "chosen" is never mistaken for "drawn".
                const selected = loading || (!pendingId && experiment.experiment_id === focusedId);
                return (
                  <li key={experiment.experiment_id}>
                    <button
                      type="button"
                      className={`tv-rail__item${selected ? ' tv-rail__item--on' : ''}${loading ? ' tv-rail__item--loading' : ''}`}
                      aria-current={selected ? 'true' : undefined}
                      aria-busy={loading || undefined}
                      title={experimentTitle(experiment)}
                      // A second click on the row already loading would queue a
                      // second document for the same answer.
                      disabled={loading}
                      onClick={() => onFocus(experiment.experiment_id)}
                    >
                      <span className="tv-rail__label">{experiment.label}</span>
                      <span className="tv-rail__badges">
                        {loading ? (
                          <span className="tv-badge tv-badge--loading" role="status">
                            loading…
                          </span>
                        ) : (
                          <>
                            <span
                              className={`tv-badge tv-badge--${experiment.status.state}`}
                            >
                              {experiment.status.state}
                            </span>
                            {experiment.status.declared && (
                              <span className="tv-badge tv-badge--declared">declared</span>
                            )}
                          </>
                        )}
                      </span>
                      <span className="tv-rail__meta">
                        {experiment.closure.length} file
                        {experiment.closure.length === 1 ? '' : 's'}
                      </span>
                    </button>
                  </li>
                );
              })}
          </ul>
        </section>
      ))}

      {index.experiments.length > 0 && matching.length === 0 && (
        <p className="tv-rail__empty">
          No experiment matches “{query.trim()}”.
        </p>
      )}

      {index.experiments.length === 0 && (
        <p className="tv-rail__empty">
          No experiment found. A file becomes one when it declares itself with
          <code> tensorviz.experiment()</code>, guards a <code>__main__</code> block,
          or runs work at import time.
        </p>
      )}

      {index.unreached.length > 0 && (
        <section className="tv-rail__group">
          <button
            type="button"
            className="tv-rail__disclosure"
            aria-expanded={showUnreached}
            onClick={() => setShowUnreached((open) => !open)}
          >
            {showUnreached ? '▾' : '▸'} {index.unreached.length} file
            {index.unreached.length === 1 ? '' : 's'} no experiment reaches
            {unparseable.length > 0 && (
              <span className="tv-rail__warn"> · {unparseable.length} will not parse</span>
            )}
          </button>
          {showUnreached && (
            <ul className="tv-rail__unreached">
              {index.unreached.map((entry) => (
                <li key={entry.file} title={entry.error ?? undefined}>
                  {entry.file}
                  {entry.error && <span className="tv-rail__warn"> — {entry.error}</span>}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </nav>
  );
}
