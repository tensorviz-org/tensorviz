/** Host-owned graph, source proposal and validation state shared with the canvas. */
import React from 'react';
import { LocalBrowserPageState } from './localBrowserState';
import { LocalBrowserViewStatus } from './localBrowserViewTransport';
import { LocalBrowserSourceEditClient } from './localBrowserSourceEditClient';
import type { LocalBrowserValidationEvidence } from './protocol.generated';
import type { LocalBrowserCaptureClient } from './localBrowserCaptureClient';
import type { LocalBrowserPortableClient } from './localBrowserPortableClient';

export interface LocalBrowserContextValue {
  codeFocus?: {
    options: import('./protocol.generated').CodeFocusOptions;
    parameter: boolean;
    detail: string;
    change(options: import('./protocol.generated').CodeFocusOptions): void;
  };
  page: LocalBrowserPageState;
  sourceAction?: 'reveal' | 'copy';
  reviewing?: boolean;
  status: LocalBrowserViewStatus | null;
  sourceEdits?: LocalBrowserSourceEditClient | null;
  captures?: LocalBrowserCaptureClient | null;
  portableExports?: LocalBrowserPortableClient | null;
  navigation?: {
    header: HTMLDivElement | null;
    footer: HTMLDivElement | null;
    /** Icon-only controls: the header is too narrow for its labels on one row. */
    compact?: boolean;
    inputsOpen: boolean;
    changeInputs(): void;
  };
  validation?: {
    running: boolean;
    /** Run fresh after a reused result: same review, but the host executes regardless. */
    runFresh?(): void;
    /** Why the button is disabled, or null when it is not. */
    disabledReason: string | null;
    /** What the Inspector says about this graph: "Not run", "Running…", the verdict,
     *  or "Not run for this version" once the source moved on. Computed where the
     *  receipt lives so exactly one place decides what a receipt means — the line it
     *  replaces was the string literal "Not run", which stayed true forever. */
    statusLabel: string;
    /** The settled host result for this source, restored with reconnect snapshots.
     * The toolbar keeps its explanation available after the transient toast. */
    lastResult: { status: 'passed' | 'failed' | 'refused' | 'cancelled'; detail: string; runId?: string } | null;
    /** Run reopens the model, which runs its saved setup at once instead of opening the review. */
    reopens?: boolean;
    /** Retained explanation, labelled when its source revision is no longer current. */
    latestResult?: { status: 'passed' | 'failed' | 'refused' | 'cancelled'; detail: string;
      runId?: string; evidence?: LocalBrowserValidationEvidence; showingMeasurements?: boolean } | null;
    run(): void;
    cancel(): void;
  } | null;
}

export const LocalBrowserContext =
  React.createContext<LocalBrowserContextValue | null>(null);
