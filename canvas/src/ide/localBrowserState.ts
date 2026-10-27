/** The local-browser page's rendering state, reduced from broker traffic.
 *
 * Host (extension) messages arrive sequenced and deduplicated from
 * localBrowserViewTransport; this reducer turns them into the one state the
 * page renders. The rules, in product language:
 *
 *  - a snapshot REPLACES state: a refreshed tab shows exactly the atomic state
 *    the host answered with, never a merge with what some earlier epoch drew;
 *  - every targeted message is matched against the active target by
 *    workspaceFolder+file+cls: a message from an answer the user has already
 *    navigated away from is dropped rather than rendered (the extension's
 *    generation guard is the real gate; this is the page's own refusal, so a
 *    dropped late answer cannot paint another class's graph under this class's
 *    name);
 *  - a retarget keeps the last projection on screen — clearing the canvas for
 *    the sub-second gap before the new projection lands flashes an empty page —
 *    but stops claiming anything ABOUT it: the sync label and the stale reason
 *    reset until the new class's own messages arrive;
 *  - a stale message puts the greyed last-valid projection back with its
 *    reason, so a buffer that stops parsing visibly dims instead of blanking;
 *  - focusNode (navigation; LocalBrowserApp routes it straight to the canvas)
 *    and acknowledgements/errors carry no graph state and change nothing here.
 *
 * Connection reading is deliberately absent: the transport owns the live
 * phase/hostLink and the strip renders those; copying the snapshot's
 * connectionState here would be a stale copy of a reading the page already
 * has fresh.
 */

import {
  LocalBrowserProjectIdentity,
  LocalBrowserTargetIdentity,
  ProjectionPayload,
  GraphOpeningState,
} from './protocol.generated';
import { LocalBrowserViewMessage } from './localBrowserViewTransport';

export type LocalBrowserSyncLabel = 'source-only' | 'runtime-traced' | 'stale' | null;

/** One experiment as the rail lists it. Mirrors research/contracts.py's
 *  `Experiment` minus its graph, which arrives separately and only for the
 *  focused one — the rail must stay cheap on a large workspace. */
export interface ProjectExperiment {
  experiment_id: string;
  label: string;
  entrypoint: string;
  root: string;
  status: { state: 'active' | 'candidate'; declared: boolean };
  closure: string[];
}

export interface ProjectIndexSummary {
  experiments: ProjectExperiment[];
  unreached: Array<{ file: string; error?: string | null }>;
  roots: string[];
  truncated: boolean;
}

export interface LocalBrowserPageState {
  activeTarget: LocalBrowserTargetIdentity | null;
  opening?: GraphOpeningState;
  /** Set once a project index arrives. Null on the per-class surface, which is
   *  how the page knows which of the two it is showing. */
  project: LocalBrowserProjectIdentity | null;
  projectIndex: ProjectIndexSummary | null;
  /** Files actually indexed, and files enumeration saw. Only the host can know the
   *  second: the backend sees the capped set and cannot know what was left out. */
  projectIndexed: number;
  projectDiscovered: number;
  /** Frame ids the host says are open, echoed from the document. */
  expanded: string[];
  projection: ProjectionPayload | null;
  traced: boolean;
  greyed: boolean;
  syncState: LocalBrowserSyncLabel;
  /** The stale explanation (`models/net.py:4 does not parse …`); null when healthy. */
  reason: string | null;
}

/** Before any host word at all: an empty page connecting to its session. */
export const EMPTY_LOCAL_BROWSER_PAGE_STATE: LocalBrowserPageState = {
  activeTarget: null,
  project: null,
  projectIndex: null,
  projectIndexed: 0,
  projectDiscovered: 0,
  expanded: [],
  projection: null,
  traced: false,
  greyed: false,
  syncState: null,
  reason: null,
};

/** The identity triple a message and the page must agree on. Versions,
 *  digests and revisions roll forward freely; these three say WHOSE graph. */
function sameTarget(
  a: LocalBrowserTargetIdentity | null,
  b: LocalBrowserTargetIdentity | null,
): boolean {
  return (
    a !== null &&
    b !== null &&
    a.workspaceFolder === b.workspaceFolder &&
    a.file === b.file &&
    a.cls === b.cls
  );
}

export function reduceLocalBrowserViewMessage(
  state: LocalBrowserPageState,
  message: LocalBrowserViewMessage,
): LocalBrowserPageState {
  switch (message.type) {
    case 'snapshot': {
      const snap = message.state;
      return {
        activeTarget: snap.activeTarget ?? snap.opening?.target ?? null,
        opening: snap.opening,
        // A snapshot REPLACES state, so the project half resets with the rest: a
        // refreshed tab must not keep a rail from an epoch the host has forgotten.
        project: null,
        projectIndex: null,
        projectIndexed: 0,
        projectDiscovered: 0,
        expanded: [],
        projection: (snap.projection as ProjectionPayload | null) ?? null,
        traced: snap.traced,
        greyed: snap.greyed,
        syncState: snap.syncState,
        reason: 'reason' in snap ? snap.reason ?? null : null,
      };
    }
    case 'graphOpening': {
      const target = message.state.target, current = state.activeTarget;
      // Setup retains its opening anchor while source projection advances.
      // Selection/reveal must keep the current graph identity for that source.
      const keepCurrent = current && sameTarget(current, target)
        && current.documentVersion === target.documentVersion && current.sourceDigest === target.sourceDigest
        && current.backendRevision > target.backendRevision;
      return { ...state, activeTarget: keepCurrent ? current : target, opening: message.state };
    }
    case 'activeTarget': {
      if (sameTarget(state.activeTarget, message.target)) {
        // The retarget anchor for the class already on screen: identity rolls
        // forward (revision/digest), the claims stand.
        return { ...state, activeTarget: message.target };
      }
      return {
        ...state,
        activeTarget: message.target,
        opening: undefined,
        project: null, projectIndex: null, projectIndexed: 0, projectDiscovered: 0, expanded: [],
        projection: state.project || state.activeTarget?.workspaceFolder !== message.target.workspaceFolder ? null : state.projection,
        traced: state.project || state.activeTarget?.workspaceFolder !== message.target.workspaceFolder ? false : state.traced,
        greyed: state.project ? false : state.greyed,
        syncState: null,
        reason: null,
      };
    }
    case 'projection': {
      if (!sameTarget(state.activeTarget, message.target)) {
        return state;
      }
      return {
        ...state,
        activeTarget: message.target,
        projection: message.projection as ProjectionPayload,
        traced: message.traced,
        greyed: message.greyed,
        // A fresh projection is never stale; the syncState message that always
        // follows re-arms the label with the tier this projection is on.
        reason: null,
      };
    }
    case 'projectIndex': {
      // The rail's content. Deliberately does NOT clear the projection: the
      // canvas keeps drawing the focused experiment while a re-index lands, the
      // same rule the class surface follows on a retarget.
      return {
        ...state,
        activeTarget: null,
        opening: undefined,
        projection: state.project?.workspaceFolder === message.project.workspaceFolder && message.project.experimentId ? state.projection : null,
        traced: false, greyed: false, syncState: null, reason: null,
        expanded: state.project?.workspaceFolder === message.project.workspaceFolder ? state.expanded : [],
        project: message.project,
        projectIndex: message.index as ProjectIndexSummary,
        projectIndexed: message.indexed ?? 0,
        projectDiscovered: message.discovered ?? 0,
      };
    }
    case 'projectDocument': {
      // A document for a project the page is no longer looking at is dropped
      // rather than drawn, exactly as a stale class projection is.
      if (!state.project || state.project.workspaceFolder !== message.project.workspaceFolder ||
        state.project.indexDigest !== message.project.indexDigest || state.project.indexRevision !== message.project.indexRevision) {
        return state;
      }
      return {
        ...state,
        project: message.project,
        activeTarget: null,
        opening: undefined,
        projection: { ...(message.document as ProjectionPayload), traced: false, greyed: false },
        traced: false,
        syncState: 'source-only',
        expanded: message.expanded,
        greyed: false,
        reason: null,
      };
    }
    case 'syncState': {
      if (!sameTarget(state.activeTarget, message.target)) {
        return state;
      }
      return {
        ...state,
        activeTarget: message.target,
        syncState: message.state,
        // A first-load failure has no staleState payload. A retarget may still
        // display the previous graph, which must be dimmed on this path too.
        ...(message.state === 'stale' ? {
          greyed: true,
          projection: state.projection && !state.projection.greyed
            ? { ...state.projection, greyed: true } : state.projection,
        } : {}),
        reason: message.reason ?? null,
      };
    }
    case 'staleState': {
      if (!sameTarget(state.activeTarget, message.target)) {
        return state;
      }
      const restored = message.lastValidProjection as ProjectionPayload;
      return {
        ...state,
        activeTarget: message.target,
        projection: restored,
        traced: restored.traced,
        greyed: true,
        syncState: 'stale',
        reason: message.reason,
      };
    }
    case 'error':
      if (message.target || message.code !== 'INVALID_MESSAGE') return state;
      return { ...state, syncState: 'stale', greyed: true, reason: message.message,
        projection: state.projection ? { ...state.projection, greyed: true } : null };
    case 'focusNode':
    case 'ack':
      return state;
    default:
      return state;
  }
}
