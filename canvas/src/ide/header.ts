/**
 * What the canvas header says it is showing.
 *
 * task.md P0 hygiene / §3.8. In IDE mode the header read "Untitled Model", which is a name
 * from the web product's own storage: there, a graph really is an untitled document until
 * someone names it. In the IDE nothing is untitled — the user opened `Net` in
 * `models/resid.py`, and the editor is showing that file two panes away. Reporting it as
 * untitled is the canvas saying it does not know what the rest of the window plainly does.
 *
 * Kept as a pure function in `src/ide/` because the components that render the header are
 * source-text pinned (task-phase1 §8.1: `ModelBuilder`, `BuilderPageV2`, `WorkbenchShell`,
 * `App.tsx`), and this is a naming rule, not a rendering change.
 */

/** The web product's placeholder, which must never be shown in the IDE. */
export const UNTITLED = 'Untitled Model';

export interface HeaderSource {
  /** The class the projection is of, e.g. `Net`. */
  readonly cls?: string;
  /** Workspace-relative path, e.g. `models/resid.py`. */
  readonly file?: string;
  /** Whatever the stored graph calls itself. In the IDE this is usually the placeholder. */
  readonly graphName?: string;
  /** `window.__TENSORVIZ_IDE__`. Outside the IDE the web product's naming is correct and
   *  this function leaves it entirely alone. */
  readonly ide: boolean;
}

export interface HeaderTitle {
  /** What to show large. */
  readonly title: string;
  /** What to show beside or beneath it, or undefined when there is nothing to add. */
  readonly subtitle?: string;
}

/**
 * The name and the location, from whatever the projection knows.
 *
 * Precedence in the IDE: the class name always wins, because it is the thing the user
 * clicked and the thing the code editor is showing. A stored graph name is used only when
 * it is a real name someone chose — never the placeholder, which is the whole bug.
 */
export function headerTitle(source: HeaderSource): HeaderTitle {
  if (!source.ide) {
    return { title: source.graphName || UNTITLED };
  }
  const named = source.graphName && source.graphName !== UNTITLED ? source.graphName : undefined;
  const title = source.cls || named || source.file || 'Graph';
  // The file is the subtitle, not part of the title: two classes in one file, or the same
  // class name in two files, are both ordinary, and the path is what tells them apart.
  const subtitle = source.file && source.file !== title ? source.file : undefined;
  return { title, subtitle };
}
