/** The bottom bar: what the page is doing, and what is taking up the screen.
 *
 * Two problems, one strip, and they belong together.
 *
 * **Nothing said the canvas was working.** Clicking an experiment sends
 * `projectFocus` and changes nothing on the page — the rail's highlight follows
 * the host's answer, so on a large project a click is followed by a long silence
 * whose only honest reading is that it missed. The rail now marks the row it is
 * waiting on; this says the same thing where the eye already is when a graph
 * refuses to change.
 *
 * **Panels and a canvas do not fit.** A rail is worth its width when you are
 * choosing an experiment and not while you are reading one, which is exactly the
 * case a toggle answers and a layout cannot. VS Code and Zed both put those
 * toggles in the status bar; this follows that, because it is where people already
 * look for them.
 *
 * Both surfaces mount it — the panel inside the editor and the external browser
 * page — and each passes its own panels. They do not have the same ones.
 *
 * Deliberately NOT a fourth floating overlay. It is a row in the page's column
 * layout, so it takes its own height once and never covers the canvas — the
 * mistake the Inspector made, which cost 352px of reserved width and a strip of
 * unclickable canvas beneath it.
 */

import React from 'react';
import { PanelLeft, PanelRight, PanelBottom, Loader2 } from 'lucide-react';

import './ProjectStatusBar.css';

/** One panel this bar can show or hide. */
export interface StatusBarToggle {
  id: string;
  label: string;
  /** Which edge the panel is docked to, which is the whole of the icon's job. */
  edge: 'left' | 'right' | 'bottom';
  open: boolean;
  /** What hiding it would cost, when that is a number worth carrying on the button. */
  count?: number;
  onToggle(): void;
}

interface ProjectStatusBarProps {
  /** How many experiments the index holds, for the resting message. */
  experiments: number;
  /** The label being loaded, or null when the canvas shows what was asked for. */
  loadingLabel: string | null;
  /** The experiment on screen, when nothing is loading. */
  focusedLabel: string | null;
  /** Left to right, as they appear. Each surface has its own panels: the panel
   *  inside the editor has a rail and a diagnostics drawer, the browser page has a
   *  rail and a suggestions column. One bar, because it is one idea. */
  toggles: StatusBarToggle[];
  /** Where a surface may dock its toolbar, as a real flex child of this bar. A
   *  callback ref: the toolbar is rendered by the same component that renders the
   *  bar, so it portals in one frame after the slot exists. Overlaying the toolbar
   *  over the bar instead put it over the status text on every narrow screen. */
  toolsSlot?: (element: HTMLDivElement | null) => void;
}

const ICONS = { left: PanelLeft, right: PanelRight, bottom: PanelBottom };

function Toggle({ toggle }: { toggle: StatusBarToggle }) {
  const Icon = ICONS[toggle.edge];
  return (
    <button
      type="button"
      className={`tv-statusbar__toggle${toggle.open ? ' tv-statusbar__toggle--on' : ''}`}
      onClick={toggle.onToggle}
      aria-pressed={toggle.open}
      data-testid={`statusbar-toggle-${toggle.id}`}
      title={`${toggle.open ? 'Hide' : 'Show'} ${toggle.label.toLowerCase()}`}
    >
      <Icon size={13} aria-hidden />
      <span>{toggle.label}</span>
      {/* The count is on the closed toggle too — hiding a panel must not hide
          that it has something in it. */}
      {toggle.count ? <span className="tv-statusbar__count">{toggle.count}</span> : null}
    </button>
  );
}

export function ProjectStatusBar({
  experiments,
  loadingLabel,
  focusedLabel,
  toggles,
  toolsSlot,
}: ProjectStatusBarProps) {
  // A toggle sits on the side its panel is docked to, which is Zed's rule and the
  // only one that needs no learning: the button for the thing on the left is on the
  // left. Grouped rather than sorted, so the bar reads as two ends with the status
  // between them instead of a row of controls that happens to be ordered.
  const left = toggles.filter((toggle) => toggle.edge === 'left');
  const right = toggles.filter((toggle) => toggle.edge !== 'left');

  return (
    <footer className="tv-statusbar">
      {left.length > 0 && (
        <div className="tv-statusbar__toggles">
          {left.map((toggle) => <Toggle key={toggle.id} toggle={toggle} />)}
        </div>
      )}
      {/* aria-live, so the wait is announced and not only drawn: the canvas keeps
          the previous graph while a document is built, and a screen reader would
          otherwise have no signal at all that anything had happened. */}
      <div className="tv-statusbar__status" role="status" aria-live="polite">
        {loadingLabel ? (
          <>
            <Loader2 size={12} className="tv-statusbar__spin" aria-hidden />
            <span>Loading {loadingLabel}…</span>
          </>
        ) : (
          <span className="tv-statusbar__rest">
            {focusedLabel ? `${focusedLabel} · ` : ''}
            {experiments} experiment{experiments === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {toolsSlot && <div className="tv-statusbar__tools" ref={toolsSlot} />}
      {right.length > 0 && (
        <div className="tv-statusbar__toggles">
          {right.map((toggle) => <Toggle key={toggle.id} toggle={toggle} />)}
        </div>
      )}
    </footer>
  );
}
