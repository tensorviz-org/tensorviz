import ExecutionPanel from './ExecutionPanel';
import { AGENT_COMPOSE_EVENT, CanvasAgentChat } from './CanvasAgentChat';
import { buildGraphReview, type ReviewMode } from './graphReview';
import { capturedGraphReview, draftAssignmentReview } from './capturedGraphReview';
import type { GraphReviewState, ProjectionPayload } from './protocol.generated';
import GraphOpeningPanel from './GraphOpeningPanel';
import { GraphSetupBoundary } from './GraphSetupBoundary';
import { GraphRunNotice } from './GraphRunNotice';
import { newLocalBrowserMessageId, DEFAULT_CODE_FOCUS, type CodeFocusState, type GraphSetupAction } from './protocol.generated';
import { createLocalBrowserViewControl } from './localBrowserViewControl';
/** External browser view of one VS Code window's authenticated broker session.
 * The page presents the host-owned projection, freshness and validation evidence.
 * Navigation, source proposals and explicit validation use typed host requests;
 * source changes require native diff review and Apply/Discard in VS Code.
 * The browser cannot write workspace files or grant execution permission.
 * Reconnect restores the session snapshot; an ended session requires Open Canvas in Browser.
 */

import React from 'react';
import { graphMcpHelpRequest } from './graphHelpRequest.generated';
import { FileCode2, GitCompareArrows, GitPullRequest, Unplug, Workflow } from 'lucide-react';
import CanvasStateScreen from './CanvasStateScreen';
import { postToHost } from './bridge';
import type { LocalBrowserValidationSummary } from './protocol.generated';

import '../App.css';
import { DraftGraphToolbar, GraphReviewActions, GraphReviewToolbar } from './GraphReviewToolbar';
import BuilderPageV2 from '../components/ui2/BuilderPageV2';
import '../styles/ui2.tokens.css';
import { ProjectRail } from './ProjectRail';
import { ProjectStatusBar } from './ProjectStatusBar';
import { ProjectSuggestions, inferredEdgesOf } from './ProjectSuggestions';
import { receiveProjection, setBrowserRevealNodeSink, setBrowserProjectExpandSink } from './projection';
import { LocalBrowserContext } from './localBrowserContext';
import { CandidateReviewPanel } from './CandidateReviewPanel';
import { createLocalBrowserCaptureClient, LocalBrowserCaptureClient } from './localBrowserCaptureClient';
import { createLocalBrowserSourceEditClient, LocalBrowserSourceEditClient } from './localBrowserSourceEditClient';
import {
  DIAGNOSTIC_LABELS,
  diagnosticSummary,
  localBrowserSyncLabel,
  orderedDiagnostics,
  projectionForCanvas,
  graphOpeningPresentation,
  sourceFocusNode,
} from './localBrowserPresentation';
import {
  LocalBrowserViewStatus,
  LocalBrowserViewTransport,
  createLocalBrowserViewTransport,
} from './localBrowserViewTransport';
import {
  EMPTY_LOCAL_BROWSER_PAGE_STATE,
  reduceLocalBrowserViewMessage,
} from './localBrowserState';
import { createLocalBrowserValidationClient, LocalBrowserValidationClient, sameValidationTarget, ValidationReceipt } from './localBrowserValidationClient';
import { inputProvenanceNote } from './cardParams';
import { GraphDocumentV2 } from '../types/graphDocument';
import './LocalBrowserApp.css';
import { nativeCanvasMode } from './nativeCanvasPresentation';
import { canExportGraphFile } from './graphFileExport';
import LocalBrowserModelMenu from './LocalBrowserModelMenu';
import PublishGraphButton from './PublishGraphButton';
import { useCompactHeader } from './useCompactHeader';
import { createLocalBrowserPortableClient, type LocalBrowserPortableClient } from './localBrowserPortableClient';

export interface LocalBrowserAppProps {
  hostKind?: 'extension' | 'standalone';
  hostControls?: React.ReactNode;
  /** Transport factory; tests inject a scripted transport, production uses the
   *  real one (fragment bootstrap, cookie, WebSocket — all inside there). */
  createTransport?: () => LocalBrowserViewTransport;
}

/** What the Inspector calls each outcome. "Passed"/"Failed" are about the model;
 *  "Refused"/"Cancelled" are about the run, and must not read as a verdict on the
 *  model — a refused run has checked nothing. */
const VALIDATION_STATUS_LABELS: Record<'passed' | 'failed' | 'refused' | 'cancelled', string> = {
  passed: 'Passed',
  failed: 'Failed',
  refused: 'Could not run',
  cancelled: 'Cancelled',
};

const labelColors: Record<'ok' | 'warn' | 'muted', string> = {
  ok: 'var(--canvas-muted)',
  warn: 'var(--canvas-warning)',
  muted: 'var(--canvas-muted)',
};

/** Connection reading, only when it is not boring: "connected to a live host"
 *  is the page's resting state and says nothing. */
function connectionLabel(status: LocalBrowserViewStatus | null, standalone = false): string | null {
  if (status === null) return 'Connecting…';
  if (status.phase === 'idle' || status.phase === 'connecting') return 'Connecting…';
  if (status.phase === 'reconnecting') return 'Connection lost — reconnecting…';
  if (status.phase === 'connected' && status.hostLink === 'disconnected') {
    return standalone ? 'The TensorViz host disconnected' : 'The VS Code window closed or detached';
  }
  return null;
}

/** The whole answer when the session is over: no graph pretending to be live. */
function endedText(error: LocalBrowserViewStatus['error'], hostLink: LocalBrowserViewStatus['hostLink'], standalone = false): string {
  if (standalone) return 'This TensorViz session ended. Reconnect the plugin in your coding agent, open the model, and request a fresh browser view.';
  if (nativeCanvasMode()) return 'Canvas lost its host connection. Run “tensorViz: Show Canvas” in VS Code to reconnect.';
  if (hostLink === 'disconnected' && error === null) {
    return 'This page’s VS Code window closed or detached. Select the model, then run “tensorViz: Open Canvas in Browser” in the window that owns this model to view it again.';
  }
  if (error === 'CREDENTIAL_EXPIRED' || error === 'INVALID_CREDENTIALS') {
    return 'This tensorViz session ended or it never belonged to this browser. Select the model, then run “tensorViz: Open Canvas in Browser” in VS Code to open a fresh page.';
  }
  return 'This tensorViz page lost its session. Select the model, then run “tensorViz: Open Canvas in Browser” in VS Code to open a fresh page.';
}


/** A remembered panel toggle. Open by default, and open whenever the answer
 *  cannot be read — a preference that fails closed would hide the rail from
 *  someone who never chose to hide it. */
function readPanelPreference(name: string): boolean {
  try {
    return window.localStorage.getItem(`tensorviz.project.${name}`) !== 'hidden';
  } catch {
    return true;
  }
}

function writePanelPreference(name: string, open: boolean): void {
  try {
    window.localStorage.setItem(`tensorviz.project.${name}`, open ? 'shown' : 'hidden');
  } catch {
    /* A private window refuses to store it; the toggle still works for this visit. */
  }
}

/** Shared graph renderer, without hosted routes or saved-graph loading. */
function Canvas() {
  return <BuilderPageV2 />;
}

export default function LocalBrowserApp({
  createTransport,
  hostKind = 'extension',
  hostControls,
}: LocalBrowserAppProps): React.ReactElement {
  const standalone = hostKind === 'standalone';
  const [sourceNotice, setSourceNotice] = React.useState<string | null>(null);
  const [codeFocus, setCodeFocus] = React.useState<CodeFocusState | null>(null);
  const [page, dispatch] = React.useReducer(
    reduceLocalBrowserViewMessage,
    EMPTY_LOCAL_BROWSER_PAGE_STATE,
  );
  const [review, setReview] = React.useState<GraphReviewState | null>(null);
  const [reviewMode, setReviewMode] = React.useState<ReviewMode>('diff');
  const [reviewIndex, setReviewIndex] = React.useState(0);
  const [reviewCapture, setReviewCapture] = React.useState<ProjectionPayload | null>(null);
  // A draft that failed to project shows an explicit unavailable view, never an empty graph.
  const sourceReview = React.useMemo(() => review && !review.proposal?.unavailable ? buildGraphReview(review) : null, [review]);
  // A draft has never run, so the last capture cannot stand in for it. It only
  // supplies context when the draft changes nothing but provable nested assignments.
  const capturedReview = React.useMemo(() => !review || !sourceReview ? null : review.proposal
    ? draftAssignmentReview(review, sourceReview, reviewCapture) : capturedGraphReview(review, sourceReview, reviewCapture),
  [review, sourceReview, reviewCapture]);
  const usingCapturedReview = !!capturedReview;
  const reviewPresentation = capturedReview ?? sourceReview;
  const pageRef = React.useRef(page);
  const copySource = React.useCallback((nodeId: string) => {
    const target = pageRef.current.activeTarget;
    if (!target) return;
    const span = pageRef.current.projection?.spanmap[nodeId] as { file?: string; start?: number[] } | undefined;
    const reference = `${target.workspaceFolder}/${span?.file ?? target.file}${span?.start?.[0] ? `:${span.start[0]}` : ''}`;
    void navigator.clipboard.writeText(reference).then(() => setSourceNotice(`Copied ${reference}`), () => setSourceNotice(reference));
  }, []);
  pageRef.current = page;
  const reviewRef = React.useRef({ state: review, presentation: reviewPresentation, mode: reviewMode, index: reviewIndex });
  const focusedReview = React.useRef<string | null>(null);
  const acceptedFocus = React.useRef<{ workspaceFolder?: string; file: string; cls: string; sourceDigest: string; nodeIds: string[] } | null>(null);
  reviewRef.current = { state: review, presentation: reviewPresentation, mode: reviewMode, index: reviewIndex };
  // A reconnect can deliver the pending review before its retained projection.
  // Freeze that context once it arrives, just as for an already-open Canvas.
  React.useEffect(() => {
    const context = review?.liveContext, { activeTarget, projection, greyed } = page;
    const usable = (candidate: ProjectionPayload | null) => review?.proposal
      ? candidate && !candidate.greyed && (candidate.traced || candidate.debug?.previousDocument) : candidate?.traced;
    if (!usable(reviewCapture) && context && activeTarget && projection && usable({ ...projection, greyed })
      && context.workspaceFolder === activeTarget.workspaceFolder
      && context.file === activeTarget.file && context.cls === activeTarget.cls) {
      setReviewCapture(projectionForCanvas({ ...projection, greyed }));
      setReviewIndex(0);
    }
  }, [review, reviewCapture, page]);
  const focusReviewNode = React.useCallback((focus: string | string[]) => {
    const { state, presentation, mode } = reviewRef.current;
    const ids = Array.isArray(focus) ? focus : [focus];
    if (state && presentation) {
      const projection = presentation.projections[mode];
      const graph = projection.reviewGraph;
      const byId = new Map(graph.nodes.map(node => [node.id, node]));
      // Focus opens the changed calls' ancestors only. A neighbour inside another
      // module is framed as that module's card, and no ancestor frame is a target.
      const ancestors = new Set<string>();
      for (const id of ids) {
        for (let parent = byId.get(id)?.data.parentId; parent && !ancestors.has(parent); parent = byId.get(parent)?.data.parentId) ancestors.add(parent);
      }
      const shown = (id: string) => {
        let node = byId.get(id);
        while (node?.data.parentId && !ancestors.has(node.data.parentId) && byId.has(node.data.parentId)) node = byId.get(node.data.parentId);
        return node?.id ?? id;
      };
      const inContext = projection.capturedContext;
      const neighbors = inContext || !state.proposal ? [] : graph.edges.filter(edge => ids.includes(edge.source) || ids.includes(edge.target))
        .flatMap(edge => [edge.source, edge.target]).map(shown);
      window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'focus-selection', payload: {
        nodeIds: Array.from(new Set([...ids, ...neighbors])).filter(id => !ancestors.has(id)), target: { file: projection.file, cls: projection.cls,
          sourceDigest: projection.sourceDigest, backendRevision: projection.revision },
        ...(inContext ? { inset: 24 } : {}),
      } } }));
    } else window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'focus-node', payload: { nodeId: ids[0] } } }));
  }, []);
  const copyReviewSource = React.useCallback((side: 'before' | 'after', nodeId?: string) => {
    const state = reviewRef.current.state;
    const root = state?.workspaceRoot ?? state?.liveContext?.workspaceFolder;
    if (!state || !root) return;
    const projection = state[side];
    const span = projection.spanmap[nodeId ?? '@class'] as { file?: string; start?: number[] } | undefined;
    const reference = `${root}/${span?.file ?? projection.file}${span?.start?.[0] ? `:${span.start[0]}` : ''} (${side === 'before' ? state.beforeLabel : state.afterLabel})`;
    void navigator.clipboard.writeText(reference).then(() => setSourceNotice(`Copied ${reference}`), () => setSourceNotice(reference));
  }, []);
  const [status, setStatus] = React.useState<LocalBrowserViewStatus | null>(null);
  const [diagnosticsOpen, setDiagnosticsOpen] = React.useState(false);
  const [dismissedOpeningId, setDismissedOpeningId] = React.useState<string | null>(null);
  const [executionExpanded, setExecutionExpanded] = React.useState(false);
  React.useEffect(() => { setExecutionExpanded(false); }, [page.activeTarget?.workspaceFolder, page.activeTarget?.file, page.activeTarget?.cls, review?.id]);
  const [headerTools, setHeaderTools] = React.useState<HTMLDivElement | null>(null);
  const [headerElement, setHeaderElement] = React.useState<HTMLElement | null>(null);
  const compactHeader = useCompactHeader(headerElement);
  const [footerTools, setFooterTools] = React.useState<HTMLDivElement | null>(null);
  /** Edge ids already proposed, so a second click cannot queue a duplicate. */
  const [declared, setDeclared] = React.useState<string[]>([]);
  /** The experiment asked for whose document has not arrived. Page state, not page
   *  reducer state: the reducer's job is to say what the HOST has told us, and this
   *  is what the user has told us. It clears when the host's answer names the same
   *  experiment, which is also what makes a click that gets no answer stay visibly
   *  unanswered instead of quietly resolving. */
  const [pendingExperimentId, setPendingExperimentId] = React.useState<string | null>(null);
  /** Which rails are open. `localStorage` because it is a per-viewer convenience:
   *  a preference worth remembering between visits and worth nothing to anyone
   *  else, and a page that cannot read it must still render — a private window and
   *  blocked site data both throw here. */
  const [railOpen, setRailOpen] = React.useState(() => readPanelPreference('rail'));
  const [suggestionsOpen, setSuggestionsOpen] = React.useState(() =>
    readPanelPreference('suggestions'));
  const warningToggleRef = React.useRef<HTMLButtonElement | null>(null);
  const diagnosticsRef = React.useRef<HTMLElement | null>(null);
  const transportRef = React.useRef<LocalBrowserViewTransport | null>(null);
  const [agentTransport, setAgentTransport] = React.useState<LocalBrowserViewTransport | null>(null);
  const viewControlRef = React.useRef<ReturnType<typeof createLocalBrowserViewControl> | null>(null);

  // The request is answered when a document for that experiment arrives. Keyed on
  // the experiment rather than on "any document", so a re-index landing while the
  // user waits does not clear a spinner for an answer that never came.
  const arrivedExperimentId = page.syncState === 'source-only' ? page.project?.experimentId : undefined;
  React.useEffect(() => { setDeclared([]); setPendingExperimentId(null); }, [page.project?.indexRevision, page.reason]);
  // Labels, not ids: the bar is read by a person. Looked up in the index because
  // that is the only place the two are related, and `??` rather than `||` so an
  // experiment legitimately labelled '' still resolves to something.
  const labelOf = (experimentId: string | null | undefined): string | null =>
    page.projectIndex?.experiments.find((e) => e.experiment_id === experimentId)?.label ?? null;
  const pendingLabel = labelOf(pendingExperimentId);
  const focusedLabel = labelOf(arrivedExperimentId);
  React.useEffect(() => {
    setPendingExperimentId((pending) => (pending === arrivedExperimentId ? null : pending));
  }, [arrivedExperimentId]);

  // The identity the page is currently showing: a reveal must name exactly
  // this, or the host refuses the navigation as stale. A ref, because the
  // click sink below outlives any single render of this state.
  const projectRef = React.useRef(page.project);
  projectRef.current = page.greyed ? null : page.project;
  const targetRef = React.useRef(page.activeTarget);
  React.useEffect(() => {
    targetRef.current = review ? null : page.activeTarget;
  }, [page.activeTarget, review]);

  // The rail below reads the session through this state, so it observes the
  // same transport and page reducer the strip renders from.
  const [agentValidation, setAgentValidation] = React.useState<LocalBrowserValidationSummary | null>(null);
  const [validationReceipt, setValidationReceipt] = React.useState<ValidationReceipt | null>(null);
  const hostSummary = React.useRef<LocalBrowserValidationSummary | null>(null);
  const hostSummaryEpoch = React.useRef(0);
  const validationGeneration = React.useRef(0);
  const sweptReceipt = React.useRef<ValidationReceipt | null>(null);
  const [validation, setValidation] = React.useState<LocalBrowserValidationClient | null>(null);
  const [sourceEdits, setSourceEdits] = React.useState<LocalBrowserSourceEditClient | null>(null);
  const [captures, setCaptures] = React.useState<LocalBrowserCaptureClient | null>(null);
  const [portableExports, setPortableExports] = React.useState<LocalBrowserPortableClient | null>(null);

  React.useEffect(() => {
    const transport = (createTransport ?? createLocalBrowserViewTransport)();
    transportRef.current = transport;
    if (!standalone) setAgentTransport(transport);
    const viewControl = createLocalBrowserViewControl(transport, () => targetRef.current);
    viewControlRef.current = viewControl;
    const edits = standalone ? null : createLocalBrowserSourceEditClient(transport, () => targetRef.current);
    setSourceEdits(edits);
    const validator = createLocalBrowserValidationClient(transport, () => targetRef.current);
    setValidation(validator);
    const captureClient = createLocalBrowserCaptureClient(transport, () => targetRef.current);
    setCaptures(captureClient);
    const portableClient = standalone ? null : createLocalBrowserPortableClient(transport, () => targetRef.current);
    setPortableExports(portableClient);
    const offMessage = transport.onMessage(message => {
      if (message.type === 'codeFocusState' && message.viewId === transport.session?.viewId
        && targetRef.current && sameValidationTarget(message.target, targetRef.current)) setCodeFocus(message);
      dispatch(message);
      if (message.type === 'graphReviewState') {
        if (message.state?.id !== reviewRef.current.state?.id) {
          if (message.state) acceptedFocus.current = null;
          setReviewMode('diff'); setReviewIndex(0);
          const { activeTarget, projection, greyed } = pageRef.current;
          const context = message.state?.liveContext;
          setReviewCapture(context && activeTarget && projection && context.workspaceFolder === activeTarget.workspaceFolder
            && context.file === activeTarget.file && context.cls === activeTarget.cls ? projectionForCanvas({ ...projection, greyed }) : null);
        }
        setReview(message.state);
      }
      if (message.type === 'validationState') {
        hostSummary.current = message;
        hostSummaryEpoch.current++;
        setAgentValidation(message);
        if (message.status === 'running') setValidationReceipt(null);
        else setValidationReceipt({ status: message.status as ValidationReceipt['status'], detail: message.detail,
          runId: message.runId, target: message.target, projectionRevision: message.projectionRevision, evidence: message.evidence });
      } else if (message.type === 'snapshot') {
        const summary = message.state.validation ?? null;
        hostSummary.current = summary;
        hostSummaryEpoch.current++;
        validationGeneration.current++;
        setValidating(false);
        setAgentValidation(summary);
        const receipt: ValidationReceipt | null = summary && summary.status !== 'running'
          ? { ...summary, status: summary.status as ValidationReceipt['status'] } : null;
        // Restore the result without replaying a capture animation as a new run.
        sweptReceipt.current = receipt;
        setValidationReceipt(receipt);
      }
    });
    const reviewRendered = (event: Event) => {
      const reviewing = reviewRef.current, viewId = transport.session?.viewId;
      const applied = (event as CustomEvent).detail as ProjectionPayload;
      const accepted = acceptedFocus.current;
      if (!reviewing.state && accepted && !applied.traced && applied.file === accepted.file
        && applied.cls === accepted.cls && applied.sourceDigest === accepted.sourceDigest
        && pageRef.current.activeTarget?.workspaceFolder === accepted.workspaceFolder) {
        acceptedFocus.current = null;
        window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'focus-selection', payload: {
          nodeIds: accepted.nodeIds, inset: 24, target: { file: applied.file, cls: applied.cls,
            sourceDigest: applied.sourceDigest, backendRevision: applied.revision },
        } } }));
      }
      if (!reviewing.state || reviewing.state.stale
        || (event as CustomEvent).detail !== reviewing.presentation?.projections[reviewing.mode]) return;
      if (viewId) transport.send({ type: 'graphReviewApplied', viewId, reviewId: reviewing.state.id, mode: reviewing.mode });
      const presentation = reviewing.presentation!;
      const index = Math.max(0, presentation.changes.findIndex(change => presentation.initialChangeId
        ? change.id === presentation.initialChangeId : presentation.nodes.get(change.id)?.status !== 'unresolved'));
      const change = presentation.changes[index];
      const focusKey = `${reviewing.state.id}:${presentation.projections.diff.contextKey ?? presentation.projections.diff.graphId}`;
      if (change && focusedReview.current !== focusKey) {
        focusedReview.current = focusKey;
        setReviewIndex(index);
        focusReviewNode(reviewing.presentation!.focusNodes?.get(change.id) ?? change.id);
      }
    };
    window.addEventListener('tensorviz-projection-rendered', reviewRendered);
    const offStatus = transport.onStatusChange(setStatus);
    // Cursor focus from the editor focuses the node here — through the same
    // builder-command bus the framed bridge dispatches on (bridge.ts), so the
    // canvas never knows which surface it lives in. The bus itself cannot
    // answer: programmatic canvas focus is not a click, so no reveal echoes.
    const focusCanvasNode = (nodeId: string) => {
      window.dispatchEvent(
        new CustomEvent('builder-command', {
          detail: { command: 'focus-node', payload: { nodeId } },
        }),
      );
    };
    const offFocus = transport.onMessage((message) => {
      const reviewing = reviewRef.current;
      if (message.type === 'graphReviewFocus' && message.reviewId === reviewing.state?.id) {
        const ids = Array.from(reviewing.presentation?.nodes ?? []).filter(([id, record]) =>
          (!reviewing.presentation?.focusNodes || id.startsWith('review:')) && message.nodeIds.includes(record[message.side] ?? ''));
        // A change located on several captured calls lists each call under its own
        // ID with the change's record. The host matches by line, so lowered copies
        // and unchanged neighbours arrive too; only navigable changes decide.
        // Revealing a call moves the cursor here too, so keep the call the navigator
        // shows; otherwise start at the first call.
        const listed = ids.filter(([id]) => reviewing.presentation!.changes.some(change => change.id === id));
        if (ids.length === 1 || (listed.length && listed.every(([, record]) => record === listed[0][1]))) {
          const current = reviewing.presentation!.changes[reviewing.index]?.id;
          const [id, record] = listed.find(([id]) => id === current) ?? listed[0] ?? ids[0];
          if (reviewing.mode !== 'diff' && !record[reviewing.mode]) setReviewMode('diff');
          setReviewIndex(Math.max(0, reviewing.presentation!.changes.findIndex(change => change.id === id)));
          const focus = reviewing.presentation?.focusNodes?.get(id) ?? id;
          requestAnimationFrame(() => Array.isArray(focus) ? focusReviewNode(focus) : focusCanvasNode(focus));
        }
        return;
      }
      if (message.type === 'focusNode' && !reviewing.state) {
        const choices = message.nodeIds?.length ? message.nodeIds : [message.nodeId];
        const nodeId = sourceFocusNode(pageRef.current.projection?.document, choices);
        if (nodeId) focusCanvasNode(nodeId);
      }
    });
    // A node activated on this canvas asks the host to reveal its line: the
    // one outward seam (ide/projection.ts's reportNodeClick), reaching the
    // broker through this page's transport instead of a frame's postMessage.
    setBrowserRevealNodeSink((nodeId) => {
      const reviewing = reviewRef.current;
      if (reviewing.state) {
        const record = reviewing.presentation?.nodes.get(nodeId);
        const side = reviewing.mode === 'before' || !record?.after ? 'before' : 'after';
        if (record?.[side]) {
          if (standalone) copyReviewSource(side, record[side]);
          else transport.send({ type: 'graphReviewRequest', action: 'reveal', reviewId: reviewing.state.id, side, nodeId: record[side] });
        }
        const changes = reviewing.presentation?.changes ?? [];
        const shown = changes.filter(change => {
          const source = reviewing.presentation?.nodes.get(change.id);
          return change.id === nodeId || (record && source?.before === record.before && source?.after === record.after);
        });
        // Every captured call of one change shares its record; select the call clicked.
        const change = shown.find(change => [reviewing.presentation?.focusNodes?.get(change.id)].flat().includes(nodeId)) ?? shown[0];
        if (change) setReviewIndex(changes.indexOf(change));
        return;
      }
      const target = targetRef.current;
      if (target) {
        if (standalone) copySource(nodeId);
        else transport.send({ type: 'revealNode', target, nodeId });
      }
    });
    setBrowserProjectExpandSink((nodeId, expanded) => {
      const project = projectRef.current;
      if (project) transport.send({ type: 'projectExpand', project, nodeId, expanded });
    });
    setStatus(transport.status);
    // Terminal failures surface through status → the ended screen; the promise
    // rejection carries nothing extra to poll-handling here.
    transport.connect().catch(() => undefined);
    return () => {
      window.removeEventListener('tensorviz-projection-rendered', reviewRendered);
      if (transportRef.current === transport) transportRef.current = null;
      setBrowserRevealNodeSink(null);
      setBrowserProjectExpandSink(null);
      viewControl.dispose();
      viewControlRef.current = null;
      edits?.dispose();
      validator.dispose();
      captureClient.dispose();
      portableClient?.dispose();
      setCaptures(null);
      setValidation(null);
      setSourceEdits(null);
      offFocus();
      offMessage();
      offStatus();
      transport.dispose();
    };
  }, [createTransport, standalone, copySource, copyReviewSource, focusReviewNode]);

  // Every projection the reducer admits is handed to the canvas through the
  // same builder-command bus the framed bridge uses — one way into the canvas,
  // for both surfaces.
  const projection = page.projection;
  const openingAction = React.useCallback((action: GraphSetupAction) => {
    if (!page.opening) return;
    if (standalone && ['edit_config', 'edit_execution', 'agent_help', 'environment', 'save_sources'].includes(action.kind)) {
      const target = page.opening.target;
      const investigation = action.kind === 'agent_help' && (page.opening.phase === 'failed' || (page.opening.environment && page.opening.environment.reason !== 'ready'));
      const text = action.kind === 'agent_help' ? graphMcpHelpRequest({ opening: { ...page.opening, ...(action.draft ? { draft: action.draft } : {}) } })
        : action.kind === 'edit_execution' ? `${target.workspaceFolder}/tensorviz.toml · profiles.${page.opening.execution?.profileName}.${action.field.join('.')}`
        : action.kind === 'edit_config' ? `${target.workspaceFolder}/tensorviz.toml`
        : `Prepare TensorViz setup for ${target.file}:${target.cls} in ${target.workspaceFolder}. Use source evidence and preserve my edited draft below. Ask about missing intent. Save the setup; opening then runs it once to capture the graph, so do not start other runs.\n${JSON.stringify({ draft: page.opening.draft, discovery: page.opening.discovery })}`;
      void navigator.clipboard.writeText(text).then(() => setSourceNotice(action.kind === 'agent_help'
        ? investigation ? 'MCP investigation request copied. Paste it into your coding agent.'
          : 'Setup request copied. Paste it into your coding agent.'
        : `Copied: ${text}`), () => setSourceNotice(text));
      if (action.kind !== 'agent_help' || !action.draft) return;
    }
    if (action.kind === 'agent_help' && !standalone) {
      window.dispatchEvent(new CustomEvent(AGENT_COMPOSE_EVENT, { detail: { opening: true } }));
      setDismissedOpeningId(page.opening.id);
    }
    transportRef.current?.send({ type: 'graphSetupRequest', openingId: page.opening.id,
      target: page.opening.target, requestId: newLocalBrowserMessageId(), action: JSON.parse(JSON.stringify(action)) });
  }, [page.opening, standalone]);
  const openingBusy = !!page.opening && ['awaiting_execution', 'resolving', 'saving', 'capturing'].includes(page.opening.phase);
  // Without a current capture, Run reopens the model; opening runs its saved setup at once.
  const runReopens = standalone || (!!page.opening && (page.opening.phase !== 'ready' || !page.opening.hasCapture || page.greyed || !projection?.traced));
  const openingView = graphOpeningPresentation(page);
  const executionPending = !!page.opening?.execution && ['review', 'running', 'cancelling'].includes(page.opening.execution.phase);
  const hasOpeningPanel = !executionPending && !review && openingView.visible && !openingView.compact;
  const showOpening = hasOpeningPanel && dismissedOpeningId !== page.opening?.id;
  const showProjectionRecovery = !review && !!page.activeTarget && !projection && page.syncState === 'stale' && !showOpening;
  const checkingPrevious = (openingBusy && page.opening?.phase !== 'awaiting_execution') || page.opening?.phase === 'cancelled'
    || (page.opening?.phase === 'failed' && !page.opening.hasCapture);
  const closeInputs = () => {
    setDismissedOpeningId(page.opening?.id ?? null);
    document.querySelector<HTMLButtonElement>('[aria-label="Change inputs"]')?.focus();
  };
  React.useEffect(() => {
    if (reviewPresentation) {
      receiveProjection(reviewPresentation.projections[reviewMode]);
      return;
    }
    if (projection !== null) {
      receiveProjection(projectionForCanvas(checkingPrevious && projection.traced ? { ...projection, greyed: true } : projection));
      viewControlRef.current?.refresh();
    }
  }, [projection, checkingPrevious, reviewPresentation, reviewMode]);
  React.useEffect(() => {
    if (!review?.id || review.proposal || usingCapturedReview || reviewPresentation?.changes.length) return;
    const timer = window.setTimeout(() => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'fit-view' } })), 100);
    return () => window.clearTimeout(timer);
  }, [review?.id, review?.proposal, usingCapturedReview, reviewPresentation]);

  // Release the temporary stale presentation before delivering completion.
  // The capture receipt can arrive before graph opening leaves "capturing".
  // A result shown again (a dismissed review restores it) is not a new run.
  React.useEffect(() => {
    const receipt = validationReceipt;
    if (checkingPrevious || !receipt || sweptReceipt.current === receipt || (!!receipt.runId && sweptReceipt.current?.runId === receipt.runId)
      || !projection?.traced || page.greyed || page.syncState === 'stale'
      || !sameValidationTarget(receipt.target, page.activeTarget)
      || receipt.projectionRevision !== projection.revision
      || (receipt.status !== 'passed' && receipt.status !== 'failed')) return;
    sweptReceipt.current = receipt;
    window.dispatchEvent(new CustomEvent('builder-command', {
      detail: { command: 'validation-complete', payload: { projection } },
    }));
  }, [checkingPrevious, validationReceipt, projection, page.activeTarget, page.greyed, page.syncState]);

  // Escape dismisses the open warning drawer and returns focus to the toggle
  // that opened it — a dismissed overlay must not strand focus on BODY.
  React.useEffect(() => {
    if (!diagnosticsOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setDiagnosticsOpen(false);
      warningToggleRef.current?.focus();
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!diagnosticsRef.current?.contains(target) && !warningToggleRef.current?.contains(target)) setDiagnosticsOpen(false);
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('pointerdown', onPointerDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('pointerdown', onPointerDown);
    };
  }, [diagnosticsOpen]);

  const revealNode = React.useCallback((nodeId: string) => {
    const target = targetRef.current;
    if (target) {
      // The node id is the only source selector crossing the browser boundary;
      // the extension resolves and validates its cached file/digest/line.
      const transport = transportRef.current;
      if (standalone) copySource(nodeId);
      else transport?.send({ type: 'revealNode', target, nodeId });
    }
  }, [standalone, copySource]);

  // Validate lives here, not in the button: the client and the receipt are both
  // here, and the generation ref is what keeps a result that arrives after the
  // target moved from being shown against the new graph.
  const [validating, setValidating] = React.useState(false);
  React.useEffect(
    () => () => {
      validationGeneration.current += 1;
      validation?.cancel();
    },
    [validation],
  );

  const agentValidating = agentValidation?.status === 'running' &&
    sameValidationTarget(agentValidation.target, page.activeTarget) && !page.greyed &&
    status?.phase === 'connected' && status.hostLink === 'connected';

  const validationDisabledReason =
    !validation || status?.phase !== 'connected' || status.hostLink !== 'connected'
      ? standalone ? 'Reconnect the TensorViz plugin to capture.' : 'Connect to VS Code to validate.'
      : agentValidating
        ? 'Another validation is running. Wait for it to finish or cancel it where it was started.'
      // Refresh owns fresh saved-profile preflight; it does not validate the old
      // AST projection, which may be unavailable for an imported library class.
      : page.opening && sameValidationTarget(page.opening.target, page.activeTarget)
        ? null
      : page.greyed || page.syncState === 'stale'
        ? page.reason ?? (standalone ? 'Refresh the saved model through your coding agent.' : 'Wait for a successful graph refresh, or reopen the graph from VS Code.')
        : !page.projection || !page.activeTarget || page.syncState === null
          ? 'Wait for the graph to load.'
          : null;

  const captureMetadata = (projection?.document as GraphDocumentV2 | undefined)?.semantic?.metadata?.capture as { backend?: string } | undefined;
  const runValidation = React.useCallback(async (fresh = false) => {
    if (!validation || validating || executionPending || validationDisabledReason) return;
    setExecutionExpanded(true);
    const generation = validationGeneration.current;
    const summaryEpoch = hostSummaryEpoch.current;
    setValidating(true);
    setValidationReceipt(null);
    const receipt = await validation.validate(fresh);
    if (generation !== validationGeneration.current) return;
    setValidating(false);
    const summary = hostSummary.current;
    if (summaryEpoch !== hostSummaryEpoch.current && summary
      && (summary.status !== 'running' || summary.runId !== receipt.runId)) return;
    setValidationReceipt(receipt);
  }, [validation, validating, executionPending, validationDisabledReason]);

  /** What the graph on screen was measured on, when nobody chose it.
   *
   * A verdict is about one tensor. "Validation passed for the traced input" is
   * true of a shape the search picked out of its own pool just as much as of the
   * one the file declares, and a user reading the first has no way to tell. The
   * projection already carries the answer on the Input node, so the verdict says
   * it too rather than leaving it on a card the reader may not be looking at. */
  const guessedInput = React.useMemo(() => {
    const document = projection?.document as Partial<GraphDocumentV2> | null | undefined;
    const input = document?.semantic?.nodes?.find((node) => node.node_id === 'input');
    return inputProvenanceNote(input?.params as Record<string, unknown> | undefined);
  }, [projection]);

  /** A verdict needs fresh graph evidence. Refusal/cancellation is an explanation
   * of the attempted run and remains useful when refresh itself failed. */
  const receiptIsCurrent =
    validationReceipt !== null &&
    sameValidationTarget(validationReceipt.target, page.activeTarget) &&
    ((validationReceipt.status === 'refused' || validationReceipt.status === 'cancelled')
      || (!page.greyed && page.syncState !== 'stale'));

  const validationStatusLabel = page.opening?.execution?.phase === 'review' ? 'Not run' : page.opening?.execution?.phase === 'cancelling' ? 'Cancelling…' : validating || agentValidating
    ? 'Running…'
    : validationReceipt === null
      ? 'Not run'
      : receiptIsCurrent
        ? VALIDATION_STATUS_LABELS[validationReceipt.status]
        : 'Not run for this version';

  const localContext = React.useMemo(
    () => ({ page, status, sourceEdits, captures, portableExports,
      codeFocus: !standalone && !review && page.activeTarget ? {
        options: codeFocus?.options ?? DEFAULT_CODE_FOCUS,
        parameter: codeFocus && sameValidationTarget(codeFocus.target, page.activeTarget) ? codeFocus.parameter : false,
        detail: codeFocus && sameValidationTarget(codeFocus.target, page.activeTarget) ? codeFocus.detail : 'Select a graph node to focus its source.',
        change: (options: import('./protocol.generated').CodeFocusOptions) => {
          const transport = transportRef.current, target = targetRef.current, viewId = transport?.session?.viewId;
          // Report the canvas selection first. Otherwise the host applies these
          // options to the previous viewSelection, one click late.
          window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'report-selection' } }));
          if (transport && target && viewId && transport.send({ type: 'codeFocusRequest', viewId, target, options })) {
            setCodeFocus(previous => ({ protocolVersion: 1, workspaceSessionId: transport.session!.workspaceSessionId,
              messageId: newLocalBrowserMessageId(), type: 'codeFocusState', viewId, target,
              detail: previous?.detail ?? 'Select a graph node to focus its source.', parameter: previous?.parameter ?? false, options }));
          }
        },
      } : undefined,
      sourceAction: standalone ? 'copy' as const : 'reveal' as const,
      reviewing: !!review,
      navigation: { header: headerTools, footer: footerTools, compact: compactHeader,
        inputsOpen: showOpening,
        changeInputs: () => {
          if (hasOpeningPanel) {
            setDismissedOpeningId(null);
            requestAnimationFrame(() => document.querySelector<HTMLElement>('.graph-opening-panel input, .graph-opening-panel button')?.focus());
          } else openingAction({ kind: 'change_inputs' });
        },
      },
      validation: {
        running: validating || openingBusy || executionPending,
        runFresh: standalone ? undefined : () => { void runValidation(true); },
        disabledReason: validationDisabledReason,
        statusLabel: standalone ? page.opening?.execution?.phase === 'review' ? 'Not run' : openingBusy ? 'Running…' : page.opening?.phase === 'ready' ? 'Passed' : page.opening?.phase === 'failed' ? 'Failed' : 'Not run' : validationStatusLabel,
        latestResult: validationReceipt && page.activeTarget && validationReceipt.target
          && ['workspaceFolder', 'file', 'cls'].every(key =>
            validationReceipt.target![key as keyof typeof page.activeTarget] === page.activeTarget![key as keyof typeof page.activeTarget])
          ? { status: validationReceipt.status, runId: validationReceipt.runId, evidence: validationReceipt.evidence,
              showingMeasurements: !!projection?.traced,
              detail: `${receiptIsCurrent ? '' : 'This result does not verify the current graph. '}${validationReceipt.detail}` }
          : null,
        lastResult: validating || openingBusy || showOpening || page.opening?.phase === 'failed' || !receiptIsCurrent || validationReceipt === null
          ? null
          // Only where something actually ran: a refused or cancelled run checked
          // no input at all, so naming the one it would have used is noise.
          : { status: validationReceipt.status, runId: validationReceipt.runId,
              detail: guessedInput
                && (validationReceipt.status === 'passed' || validationReceipt.status === 'failed')
                ? `${validationReceipt.detail} ${guessedInput.detail}`
                : validationReceipt.detail },
        reopens: runReopens,
        run: () => {
          if (page.opening?.execution?.phase === 'review') { setExecutionExpanded(true); return; }
          // A run that starts at once leaves the details closed; a new review opens them.
          if (runReopens) { setExecutionExpanded(false); openingAction({ kind: 'retry' }); }
          else void runValidation();
        },
        cancel: () => { if (openingBusy) openingAction({ kind: 'cancel' }); else validation?.cancel(); },
      } }),
    [page, status, sourceEdits, captures, portableExports, codeFocus, validating, validationDisabledReason,
     validationStatusLabel, validationReceipt, receiptIsCurrent, runValidation, validation, executionPending,
     review, guessedInput, projection, openingBusy, runReopens, showOpening, hasOpeningPanel, openingAction, headerTools, footerTools, compactHeader, standalone],
  );

  if (status !== null && status.phase === 'ended') {
    return <CanvasStateScreen title="Reconnect to your model" icon={<Unplug />}>
      <p>{endedText(status.error, status.hostLink, standalone)}</p>
    </CanvasStateScreen>;
  }

  const previousMeasurements = !validating && receiptIsCurrent && projection?.traced
    && validationReceipt?.projectionRevision === undefined
    && !page.greyed && page.syncState === 'runtime-traced'
    && validationReceipt?.status !== 'passed';
  const label = review ? { tone: 'muted' as const, text: review.proposal ? 'Proposed source · Not run' : usingCapturedReview
    ? 'Graph diff'
    : `Source comparison · ${review.changedFiles.length} changed Python file${review.changedFiles.length === 1 ? '' : 's'} · Not executed` } : previousMeasurements
    ? { tone: 'muted' as const, text: captureMetadata?.backend === 'jaxpr' ? 'Previous JAX trace retained' : 'Previous measurements retained' }
    : localBrowserSyncLabel(page);
  const reviewRevisions = review ? `${review.title ? `${review.title} · ` : ''}${review.beforeLabel} → ${review.afterLabel}` : '';
  const closeReview = () => review && transportRef.current?.send({ type: 'graphReviewRequest', action: 'close', reviewId: review.id });
  const refreshReview = () => review && transportRef.current?.send({ type: 'graphReviewRequest', action: 'refresh', reviewId: review.id });
  const revealReviewSource = () => review && (standalone ? copyReviewSource(reviewMode === 'before' ? 'before' : 'after')
    : transportRef.current?.send({ type: 'graphReviewRequest', action: 'reveal', reviewId: review.id, side: reviewMode === 'before' ? 'before' : 'after' }));
  const changeReview = (index: number) => {
    if (!review || !reviewPresentation) return;
    setReviewIndex(index);
    const change = reviewPresentation.changes[index];
    if (!change) return;
    const record = reviewPresentation.nodes.get(change.id)!;
    const side = reviewMode === 'before' || !record.after ? 'before' : 'after';
    if (reviewMode !== 'diff' && !record[reviewMode]) setReviewMode('diff');
    requestAnimationFrame(() => focusReviewNode(reviewPresentation.focusNodes?.get(change.id) ?? change.id));
    if (!standalone && !review.proposal) transportRef.current?.send({ type: 'graphReviewRequest', action: 'reveal', reviewId: review.id, side, nodeId: record[side] });
  };
  const connection = connectionLabel(status, standalone);
  const diagnostics = orderedDiagnostics(review ? [...review.before.diagnostics ?? [], ...review.after.diagnostics ?? []] : projection?.diagnostics);
  const summary = diagnosticSummary(diagnostics);
  const selectedEnvironment = page.opening?.environment?.reason === 'ready'
    ? page.opening.environment.candidates.find(candidate => candidate.id === page.opening?.environment?.selectedId) : undefined;
  if (!review && nativeCanvasMode() && page.activeTarget === null && page.projectIndex === null) {
    if (status?.phase !== 'connected' || status.hostLink !== 'connected') {
      return <CanvasStateScreen title={status?.phase === 'connected' ? 'Reconnect to Canvas'
        : status?.phase === 'reconnecting' ? 'Reconnecting to Canvas' : 'Connecting to Canvas'} icon={<Unplug />}>
        <p>{status?.phase === 'connected' ? endedText(status.error, status.hostLink, standalone)
          : 'Your model will appear when the Canvas host is ready.'}</p>
      </CanvasStateScreen>;
    }
    return <CanvasStateScreen title="Explore your model" illustration actions={<>
      <button type="button" className="primary" disabled={status.hostLink !== 'connected'} onClick={() => postToHost({ type: 'openAsGraph' })}>
        <Workflow aria-hidden="true" />Open as Graph</button>
      <button type="button" disabled={status.hostLink !== 'connected'} onClick={() => transportRef.current?.send({ type: 'graphReviewRequest', action: 'openPR' })}>
        <GitPullRequest aria-hidden="true" />Open PR</button>
    </>}>
      <p>Open a Python model to explore its layers and connections, or review the model changes in a pull request.</p>
    </CanvasStateScreen>;
  }
  return (
    <div className={`local-browser-canvas${review ? ' graph-review-canvas' : ''}${nativeCanvasMode() ? ' native-canvas' : ''}`} style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header ref={setHeaderElement} className={`local-browser-header${compactHeader ? ' local-browser-header-compact' : ''}`} data-testid="local-browser-strip">
        <div className="local-browser-header-row">
          <div className="local-browser-header-identity">
            {review ? <div className="local-browser-compact-title" title={review ? reviewRevisions : page.activeTarget ? `${page.activeTarget.cls} · ${page.activeTarget.file}` : 'tensorViz'}>
              <strong>{review ? review.proposal ? 'Draft graph' : review.title || 'Graph diff' : page.activeTarget?.cls ?? 'tensorViz'}</strong>
              {review ? <span>{review.beforeLabel} → {review.afterLabel}</span> : page.activeTarget && <span>{page.activeTarget.file}</span>}
            </div> : <LocalBrowserModelMenu target={page.activeTarget}>
              {selectedEnvironment && <div className="local-browser-model-detail" title={selectedEnvironment.interpreter}>
                <p>Used by tensorViz · Python {selectedEnvironment.python} · Torch {selectedEnvironment.torch}</p>
                <button type="button" disabled={openingBusy || status?.phase !== 'connected' || status.hostLink !== 'connected'}
                  onClick={() => { setDismissedOpeningId(null); openingAction({ kind: 'refresh_environments' }); }}>Change environment</button>
              </div>}
              {openingView.compact && page.opening?.draft?.profileName && page.opening.configFileExists !== false && !page.greyed && <div className="local-browser-model-detail">
                <p>Saved as <strong>{page.opening.draft.profileName}</strong></p>
                <button type="button" disabled={status?.phase !== 'connected' || status.hostLink !== 'connected'}
                  onClick={() => openingAction({ kind: 'edit_config' })}>{standalone ? 'Copy config path' : 'Open tensorviz.toml'}</button>
              </div>}
            </LocalBrowserModelMenu>}
            {!review && page.opening?.execution && <ExecutionPanel state={page.opening.execution}
              bottomBoundary={footerTools?.closest('footer')}
              evidence={page.opening.execution.profileName && page.opening.execution.profileName === page.opening.draft?.profileName
                ? page.opening.evidence?.filter(item => item.field === 'call' || item.field === 'inputs' || item.field.startsWith('inputs.')) : undefined}
              expanded={executionExpanded} onExpandedChange={setExecutionExpanded}
              connected={status?.phase === 'connected' && status.hostLink === 'connected'} onAction={openingAction}
              onRepeat={localContext.validation.run}
              onRunFresh={localContext.validation.runFresh}
              repeatDisabled={localContext.validation.running || !!localContext.validation.disabledReason}
              onRecovery={() => { setDismissedOpeningId(null); openingAction({ kind: 'change_inputs' }); }} />}
          </div>
          <div className="local-browser-header-actions" role="group" aria-label={review ? 'Review actions' : 'Model tools'}>
          {review && !review.proposal && <GraphReviewActions sourceAction={standalone ? 'copy' : undefined} onClose={closeReview} onRefresh={refreshReview} onSource={revealReviewSource} />}
          <div className="local-canvas-tools" ref={setHeaderTools} />
          {hostControls}
          {nativeCanvasMode() && <button type="button" className="local-canvas-control" aria-label="Open PR" title="Review a pull request"
            onClick={() => transportRef.current?.send({ type: 'graphReviewRequest', action: 'openPR' })}><GitPullRequest size={15} aria-hidden="true" /><span className="local-canvas-control-label">Open PR</span></button>}
          {nativeCanvasMode() && !review && page.activeTarget && <button type="button" className="local-canvas-control" aria-label="Review model changes" title="Compare this model across Git revisions"
            onClick={() => transportRef.current?.send({ type: 'graphReviewRequest', action: 'open' })}><GitCompareArrows size={15} aria-hidden="true" /><span className="local-canvas-control-label">Review changes</span></button>}
          {!standalone && !review && <PublishGraphButton disabled={showOpening || openingBusy || !canExportGraphFile(page)} />}
          </div>
        </div>
        {!review && <>
        {page.opening?.configFileExists === false && !openingBusy && !page.greyed && <section className="local-browser-config-warning" role="status" aria-label="Missing model configuration">
          <div><strong>tensorviz.toml is missing</strong>
            <p>{page.opening.hasCapture ? 'The graph keeps its previous measurements. ' : 'Source inspection is available. '}
              Save a model setup and example inputs to capture tensor shapes. Use Agent to complete setup, or edit the file manually.</p></div>
          <div className="local-browser-config-actions">
            <button type="button" className="local-canvas-control" disabled={status?.phase !== 'connected' || status.hostLink !== 'connected'}
              onClick={() => openingAction({ kind: 'agent_help' })}>Use Agent</button>
            <button type="button" className="local-canvas-control" disabled={status?.phase !== 'connected' || status.hostLink !== 'connected'}
              onClick={() => openingAction({ kind: 'edit_config' })}>Create tensorviz.toml</button>
          </div>
        </section>}

        </>}
      </header>
      {review?.proposal && <DraftGraphToolbar state={review} presentation={reviewPresentation} index={reviewIndex} onChange={changeReview}
        onSource={() => transportRef.current?.send({ type: 'graphReviewRequest', action: 'reveal', reviewId: review.id, side: 'after' })}
        onAccept={() => {
          const sent = !!transportRef.current?.send({ type: 'graphReviewRequest', action: 'acceptDraft', reviewId: review.id });
          const change = reviewPresentation?.changes[reviewIndex];
          const nodeIds = change && reviewPresentation?.focusNodes?.get(change.id);
          acceptedFocus.current = sent && review.after.sourceDigest && reviewPresentation?.projections.diff.capturedContext && Array.isArray(nodeIds)
            ? { workspaceFolder: pageRef.current.activeTarget?.workspaceFolder, file: review.after.file,
              cls: review.after.cls, sourceDigest: review.after.sourceDigest, nodeIds } : null;
          return sent;
        }}
        onDiscard={() => {
          acceptedFocus.current = null;
          return !!transportRef.current?.send({ type: 'graphReviewRequest', action: 'discardDraft', reviewId: review.id });
        }}
        onContinueEditing={standalone || review.proposal?.canRevise === false ? undefined : () => window.dispatchEvent(new CustomEvent(AGENT_COMPOSE_EVENT))} />}
      {review && !review.proposal && reviewPresentation && <GraphReviewToolbar state={review} presentation={reviewPresentation} mode={reviewMode} index={reviewIndex}
        capturedContext={usingCapturedReview}
        onModel={modelId => transportRef.current?.send({ type: 'graphReviewRequest', action: 'selectModel', reviewId: review.id, modelId })}
        onMode={setReviewMode}
        onChange={changeReview}
        onRefresh={refreshReview} />}
      <LocalBrowserContext.Provider value={localContext}>
        {sourceNotice && <div role="status" className="local-browser-saved-profile"><span>{sourceNotice}</span><button type="button" onClick={() => setSourceNotice(null)}>Dismiss</button></div>}
        {!standalone && (!review || review.proposal?.canRevise === false) && <CandidateReviewPanel />}
        {!review && !executionPending && openingView.compact && <GraphRunNotice page={page} standalone={standalone} connected={status?.phase === 'connected' && status.hostLink === 'connected'} onAction={openingAction} />}
        <div className="local-browser-content">
          {hasOpeningPanel && page.opening && <div className="graph-opening-stage" hidden={!showOpening}>
            <GraphSetupBoundary state={page.opening} standalone={standalone} onAction={openingAction} onClose={closeInputs}
              connected={status?.phase === 'connected' && status.hostLink === 'connected'}>
              <GraphOpeningPanel state={page.opening} standalone={standalone} compact={openingView.compact} retained={openingView.retained}
                connected={status?.phase === 'connected' && status.hostLink === 'connected'} onAction={openingAction} onClose={closeInputs} />
            </GraphSetupBoundary>
          </div>}
          {/* The rail appears only once a project index has arrived, and wraps
              rather than replaces the content region, so the per-class surface
              keeps exactly the layout it had before the project canvas existed. */}
          {review?.models?.length === 0 ? <CanvasStateScreen embedded title="No model identified" icon={<GitCompareArrows />} actions={<>
            <button type="button" className="primary" onClick={revealReviewSource}><FileCode2 aria-hidden="true" />{standalone ? 'Copy source reference' : 'View code changes'}</button>
            <button type="button" onClick={closeReview}>Return to Canvas</button>
          </>}>
            <p>No affected model could be identified in this comparison.</p>
            <p>{review.notes?.length ? 'Review the coverage notes above and the code diff for changes that could not be analyzed.' : 'The changed files do not affect a discovered local model.'}</p></CanvasStateScreen>
          : !review && page.projectIndex && page.project ? (
            <>
            <div className="local-browser-split">
              {railOpen && <ProjectRail
                index={page.projectIndex}
                focusedId={page.project.experimentId}
                indexed={page.projectIndexed}
                discovered={page.projectDiscovered}
                pendingId={pendingExperimentId}
                onFocus={(experimentId) => {
                  const transport = transportRef.current;
                  const project = page.project;
                  if (!transport || !project || page.greyed || status?.phase !== 'connected' || status.hostLink !== 'connected') {
                    return;
                  }
                  // The page names an experiment and nothing else. Which files
                  // that means, and whether the index is still current, are the
                  // host's to decide — the rule navigation already follows.
                  setPendingExperimentId(experimentId);
                  transport.send({ type: 'projectFocus', project, experimentId });
                }}
              />}
              {suggestionsOpen && <ProjectSuggestions
                edges={inferredEdgesOf(page.projection)}
                pending={declared}
                onDeclare={(edge) => {
                  const transport = transportRef.current;
                  const project = page.project;
                  if (!transport || !project || page.greyed || status?.phase !== 'connected' || status.hostLink !== 'connected') {
                    return;
                  }
                  // A proposal. The host validates the identity, asks the backend
                  // for a patch, and puts a native diff in front of the user; the
                  // page never learns which file it landed in.
                  transport.send({
                    type: 'projectDeclare',
                    project,
                    declaration: 'edge',
                    args: [edge.from, edge.to, edge.relationship],
                  });
                  setDeclared((sent) => [...sent, edge.id]);
                }}
              />}
              <Canvas />
            </div>
            <ProjectStatusBar
              experiments={page.projectIndex.experiments.length}
              loadingLabel={pendingLabel}
              focusedLabel={focusedLabel}
              toggles={[
                {
                  id: 'rail',
                  label: 'Experiments',
                  edge: 'left',
                  open: railOpen,
                  onToggle: () => setRailOpen((open) => {
                    writePanelPreference('rail', !open);
                    return !open;
                  }),
                },
                {
                  id: 'suggestions',
                  label: 'Suggestions',
                  edge: 'right',
                  open: suggestionsOpen,
                  count: inferredEdgesOf(page.projection).length,
                  onToggle: () => setSuggestionsOpen((open) => {
                    writePanelPreference('suggestions', !open);
                    return !open;
                  }),
                },
              ]}
            />
            </>
          ) : standalone && !review && !projection ? (
            <CanvasStateScreen embedded title="Open your first model" icon={<Workflow />}><p>No model is open.</p>
              <p>Ask your coding agent to open a model or compare saved Git changes.</p></CanvasStateScreen>
          ) : (
            <>
            {review?.proposal?.unavailable && <div className="graph-review-empty" role="status"><p>Graph preview unavailable for this draft.</p>
              <p>The proposed source could not be projected. Review it in the code diff before you accept or discard it.</p></div>}
            {/* The canvas stays mounted under an unavailable draft so the live graph is intact when the proposal ends. */}
            <div className="graph-opening-canvas" style={{ visibility: (!review && openingView.waiting && !showProjectionRecovery) || review?.proposal?.unavailable ? 'hidden' : 'visible' }}
              aria-hidden={(!review && openingView.waiting && !showProjectionRecovery) || !!review?.proposal?.unavailable} data-awaiting-capture={!review && openingView.waiting}>
              <Canvas />
              {showProjectionRecovery && <div className="graph-projection-recovery"><CanvasStateScreen embedded title="Graph preview unavailable" icon={<FileCode2 />} actions={page.opening && <>
                <button type="button" className="primary" disabled={openingBusy || status?.phase !== 'connected' || status.hostLink !== 'connected'} onClick={() => openingAction({ kind: 'retry' })}>Retry graph</button>
                <button type="button" disabled={openingBusy || status?.phase !== 'connected' || status.hostLink !== 'connected'} onClick={() => {
                  setDismissedOpeningId(null); openingAction({ kind: 'change_inputs' });
                }}>Set up model</button>
              </>}>
                <p>{page.reason ?? 'This model could not be drawn from its source.'}</p>
                <p>{openingBusy ? page.opening?.detail : 'You can still set up this model and capture a graph in a compatible Python environment.'}</p>
              </CanvasStateScreen></div>}
            </div>
            </>
          )}
          {/* A draft keeps chat bound to its model; typed drafts permit inspection without code revision. */}
          {!standalone && <CanvasAgentChat transport={agentTransport} target={review && !review.proposal ? null : page.activeTarget}
            draftReview={review?.proposal ? { id: review.proposal.id, stale: review.stale, canRevise: review.proposal.canRevise } : undefined}
            connected={status?.phase === 'connected' && status.hostLink === 'connected'} />}
        </div>
      </LocalBrowserContext.Provider>
      <footer className="local-browser-footer" aria-label="Canvas status and navigation">
        <div className="local-browser-status">
          {label !== null && (
            <span className="local-browser-sync" style={{ color: labelColors[label.tone] }} title={page.reason ?? undefined}>
              {label.text}
            </span>
          )}
          {diagnostics.length > 0 && (
            <button
              type="button"
              ref={warningToggleRef}
              className="local-browser-warning-toggle"
              aria-expanded={diagnosticsOpen}
              aria-controls="local-browser-diagnostics"
              title={summary}
              onClick={() => setDiagnosticsOpen((open) => !open)}
            >
              {diagnostics.some(item => item.severity === 'error') ? summary : `${diagnostics.length} source note${diagnostics.length === 1 ? '' : 's'}`}
            </button>
          )}
          {connection !== null && <span className="local-browser-connection">{connection}</span>}
        </div>
        <div className="local-canvas-footer-tools" ref={setFooterTools} />
        {diagnosticsOpen && diagnostics.length > 0 && (
          <section
            ref={diagnosticsRef}
            id="local-browser-diagnostics"
            className="local-browser-diagnostics"
            aria-label="Static graph resolution warnings"
          >
            {diagnostics.map((diagnostic, index) => (
              <article
                className={`local-browser-diagnostic ${diagnostic.severity}`}
                key={`${diagnostic.code}:${diagnostic.file}:${diagnostic.line}:${diagnostic.node ?? index}`}
              >
                <div className="local-browser-diagnostic-title">
                  <strong>{DIAGNOSTIC_LABELS[diagnostic.code]}</strong>
                  <span>{diagnostic.file ? `${diagnostic.file}${diagnostic.line ? `:${diagnostic.line}` : ''}` : 'workspace'}</span>
                </div>
                <p>{diagnostic.detail}</p>
                {diagnostic.import_text && <code>{diagnostic.import_text}</code>}
                {diagnostic.node && (
                  <div className="local-browser-diagnostic-actions">
                    <button type="button" onClick={() => {
                      window.dispatchEvent(new CustomEvent('builder-command', {
                        detail: { command: 'focus-node', payload: { nodeId: diagnostic.node } },
                      }));
                      setDiagnosticsOpen(false);
                      warningToggleRef.current?.focus();
                    }}>Locate in graph</button>
                    <button type="button" onClick={() => revealNode(diagnostic.node!)}>{standalone ? 'Copy source reference' : 'Show source'}</button>
                  </div>
                )}
              </article>
            ))}
          </section>
        )}
      </footer>
    </div>
  );
}
