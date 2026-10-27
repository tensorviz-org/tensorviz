/**
 * The panel registry: which surface shows a run record (task.md M9).
 *
 * Why a registry at all, when there are exactly two entries and the milestone adds
 * one of them: M9's job is to make a data-pipeline or an eval view an ADDITION
 * rather than a rewrite, and the thing that decides that is whether "what shows a
 * traced program" is a lookup or an `if` spread across `extension.ts`. It is a
 * lookup. It is also deliberately just a lookup — no lazy loading, no activation,
 * no ordering rules — because a registry that does more than map is the speculative
 * abstraction CLAUDE.md rule 2 forbids, and M9 is already an agreed exception to it.
 *
 * There are exactly TWO registrants and a test that says so. A third is a milestone,
 * not a commit.
 */

/** The kinds of program the worker can trace (`services/ingest/spec.py::TraceKind`). */
export type RecordKind = 'model' | 'data' | 'train_step';

export const RECORD_KINDS: readonly RecordKind[] = ['model', 'data', 'train_step'];

export function isRecordKind(value: unknown): value is RecordKind {
  return typeof value === 'string' && (RECORD_KINDS as readonly string[]).includes(value);
}

export interface PanelRegistrant {
  /** Stable id, for a log line and for the test that pins the set. */
  id: string;
  /** The view or webview id this registrant owns. */
  view: string;
  /** What a user would call it. */
  title: string;
  /** Whether this registrant can show a record of that kind. */
  handles(kind: RecordKind): boolean;
}

/**
 * The canvas.
 *
 * It handles the kinds whose record carries a MODEL graph, which is `model` and —
 * measurably, not by assumption — `train_step`: a training step's graph is the
 * graph of the model it trains, node for node and edge for edge, because the
 * backward and the optimiser step are measured rather than drawn
 * (`services/ingest/trace.py`, and `test_trace_record.py` asserts the two are
 * identical). A `data` record's graph is a pipeline of source lines, with no class
 * to parse and therefore nothing for `/api/ide/project` to merge it with; drawing it
 * on the class canvas would be a graph the code panel could not follow.
 */
export const GRAPH_PANEL: PanelRegistrant = {
  id: 'tensorviz.panel.graph',
  view: 'tensorviz.sidebarCanvas',
  title: 'tensorViz Canvas',
  handles: (kind) => kind === 'model' || kind === 'train_step',
};

/** The Runs list. It handles everything: a run that no panel can open is still a
 *  run that happened, and a list that hid it would be lying about what was run. */
export const RUNS_PANEL: PanelRegistrant = {
  id: 'tensorviz.panel.runs',
  view: 'tensorviz.runs',
  title: 'tensorViz Runs',
  handles: () => true,
};

export const PANELS: readonly PanelRegistrant[] = [GRAPH_PANEL, RUNS_PANEL];

/** Every registrant that can show this kind, in registration order. */
export function panelsFor(kind: RecordKind): PanelRegistrant[] {
  return PANELS.filter((panel) => panel.handles(kind));
}

/** The one to open when the user picks a record, or undefined if none can. */
export function primaryPanelFor(kind: RecordKind): PanelRegistrant | undefined {
  return panelsFor(kind)[0];
}
