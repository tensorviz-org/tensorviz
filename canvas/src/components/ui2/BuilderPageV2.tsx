import { findQuickJumpNodes } from './quickJump';
import { operationDisplayName } from '../../canvas/operationDisplayName';
import GraphDetailToggle from '../GraphDetailToggle';
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useWorkbenchServices } from '../../workbench/services';
import { torchClassForLayer } from '../../utils/torchLayerClass';
import * as hostServices from '../../ide/hostServices';
import { ideMode, localBrowserMode, reportNodeClick } from '../../ide/projection';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import LocalCanvasNavigation from '../../ide/LocalCanvasNavigation';
import LocalCanvasMenu from '../../ide/LocalCanvasMenu';
import { searchGraph } from '../../canvas/graphSearch';
import { graphSearchQuery } from '../../canvas/graphQueryState';
import { formatParams, formatSize } from '../../services/graphMetrics';
import {
  CheckCircle2,
  Check,
  ChevronDown,
  Copy,
  Code2,
  Package,
  Cpu,
  Boxes,
  Crosshair,
  Expand,
  FileCode2,
  GitBranchPlus,
  Ungroup,
  Layers3,
  MousePointer2,
  MemoryStick,
  Play,
  Plus,
  Minus,
  RotateCcw,
  RotateCw,
  Trash2,
  Sparkles,
  Scan,
  Workflow,
  Wrench,
  Maximize2,
  History,
  Save,
  Search,
  MoreHorizontal,
  ClipboardList,
  ImageDown,
  Image as ImageIcon,
} from 'lucide-react';
import WorkbenchShell from './WorkbenchShell';
import ModelBuilder, { ModelBuilderSnapshot } from '../ModelBuilder';
import { SavedGraph } from '../../types';
import BuilderPaletteV2 from './BuilderPaletteV2';
import { GraphDraftStore } from '../../services/localDraft';
// The draft <-> graph value conversion lives in its own module so it can be
// unit-tested; BuilderPageV2 cannot be imported under the jest transform.
import { formatValue } from './parameterValues';
import { dispatchBuilderCommand } from './builderCommand';
import { PassThroughToggle } from '../PassThroughToggle';
import { graphWarnings } from '../../utils/graphWarnings';
import { DIAGNOSTIC_LABELS } from '../../ide/localBrowserPresentation';

interface BuilderPageV2Props {
  initialGraph?: SavedGraph | null;
  draftStore?: GraphDraftStore;
  autoOpenImport?: boolean;
  /** Called when a from-scratch graph is first persisted, so the route can adopt
   *  its id (/builder/:id) and a reload doesn't lose the work. */
  onGraphCreated?: (id: string) => void;
}

type DiagnosticsTab = 'logs' | 'shapes' | 'errors' | 'warnings';

type LogEntry = {
  id: string;
  label: string;
  detail: string;
  status: 'ok' | 'info' | 'warn' | 'error';
  at: string;
};

type DiagnosticRow = {
  nodeId: string | null;
  name: string;
  message: string;
  severity?: 'info' | 'warning' | 'error';
};

const createInitialSnapshot = (): ModelBuilderSnapshot => ({
  backendConnected: false,
  isInferringShapes: false,
  shapeInferenceHasRun: false,
  verifying: false,
  exporting: false,
  nodeCount: 0,
  edgeCount: 0,
  errorCount: 0,
  warningCount: 0,
  staticDiagnostics: [],
  repairCount: 0,
  selectedNodeCount: 0,
  nodeIssues: [],
  selectedNode: null,
  historyState: { undo_count: 0, redo_count: 0, event_count: 0 },
  lastHistorySummary: 'No changes yet',
  canvasViewMode: 'overview',
  verifyResult: null,
  saveState: 'idle',
  ideStatus: null,
});

function formatShape(shape?: number[]) {
  return Array.isArray(shape) && shape.length > 0 ? `[${shape.join(', ')}]` : '—';
}


export default function BuilderPageV2({
  initialGraph,
  draftStore,
  autoOpenImport,
  onGraphCreated,
}: BuilderPageV2Props) {
  const [snapshot, setSnapshot] = useState<ModelBuilderSnapshot>(createInitialSnapshot);
  const [diagnosticsTab, setDiagnosticsTab] = useState<DiagnosticsTab>('logs');
  const [logsCollapsed, setLogsCollapsed] = useState(true);
  const services = useWorkbenchServices();
  const HistoryPanel = services?.HistoryPanel;
  const [moreOpen, setMoreOpen] = useState(false);
  // Mirrors ModelBuilder's expanded-overview mode so the menu can label the toggle.
  const [svgOverview, setSvgOverview] = useState(false);
  // The ⋯ menu is portalled to document.body and positioned from the trigger's
  // rect. It cannot render inside the toolbar: .ui2-builder-toolbar sets
  // `overflow-x: auto` so the ~995px of controls stay reachable on a narrow
  // window, and CSS computes the other axis to `overflow-y: hidden` — which
  // clipped the menu, opening upward at `bottom: 100%`, out of existence. It was
  // still in the DOM with a correct getBoundingClientRect, which is why
  // toBeVisible() passed and roughly ten controls were silently unclickable.
  const moreButtonRef = useRef<HTMLButtonElement | null>(null);
  const morePanelRef = useRef<HTMLDivElement | null>(null);
  const [moreAnchor, setMoreAnchor] = useState<React.CSSProperties | null>(null);
  // The diagnostics drawer is bottom-anchored and grows upward, while the
  // floating toolbar sits 18px off the same edge at a higher z-index — so an open
  // drawer rendered its top log rows UNDERNEATH the toolbar. Measure the drawer
  // and lift the toolbar clear of it.
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const [drawerHeight, setDrawerHeight] = useState(0);
  const [viewTab, setViewTab] = useState<'graph' | 'code'>('graph');
  // In the IDE the real file is one pane away, so the honest thing this button can do is go
  // there. Outside it there is no editor to reveal, and the tab keeps showing the generated
  // per-layer snippet — labelled for what it is, never as a whole-model "Code View".
  const inIde = ideMode();
  const readOnlyLocalBrowser = localBrowserMode();
  // Null outside LocalBrowserApp, which is every mode but this one.
  const localBrowser = useContext(LocalBrowserContext);
  const localBrowserValidation = localBrowser?.validation ?? null;
  const codeTabLabel = inIde ? 'Reveal in editor' : 'Layer Code';
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [chromeHidden, setChromeHidden] = useState(false);
  const [quickJumpOpen, setQuickJumpOpen] = useState(false);
  const findButtonRef = useRef<HTMLButtonElement>(null);
  const wasQuickJumpOpen = useRef(false);
  const [searchPosition, setSearchPosition] = useState<React.CSSProperties | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [quickJumpQuery, setQuickJumpQuery] = useState('');
  const [searchPage, setSearchPage] = useState(0);
  const searchListRef = useRef<HTMLDivElement>(null);
  useEffect(() => { setSearchPage(0); }, [quickJumpQuery, quickJumpOpen]);
  const [isCanvasFullscreen, setIsCanvasFullscreen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [toasts, setToasts] = useState<LogEntry[]>([]);
  const [canvasGraph, setCanvasGraph] = useState<{ nodes: any[]; edges: any[] } | null>(null);
  const [copiedCode, setCopiedCode] = useState(false);
  const prevSnapshotRef = useRef<ModelBuilderSnapshot>(createInitialSnapshot());

  React.useLayoutEffect(() => {
    if (!readOnlyLocalBrowser || !quickJumpOpen) return;
    const place = () => {
      const rect = findButtonRef.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(400, window.innerWidth - 16);
      setSearchPosition({ width, left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)),
        top: rect.bottom + 8, maxHeight: Math.max(80, window.innerHeight - rect.bottom - 16) });
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); };
  }, [readOnlyLocalBrowser, quickJumpOpen]);
  useEffect(() => {
    if (!quickJumpOpen && wasQuickJumpOpen.current) findButtonRef.current?.focus();
    wasQuickJumpOpen.current = quickJumpOpen;
  }, [quickJumpOpen]);

  useEffect(() => {
    if (!initialGraph?.conflictedDraftAvailable) return;
    setToasts((current) => [{
      id: 'conflicted-draft-preserved',
      label: 'Opened the saved graph',
      detail: 'Edits made here in another tab could not be merged. They were archived locally, not applied.',
      status: 'warn' as const,
      at: '',
    }, ...current].slice(0, 4));
  }, [initialGraph?.conflictedDraftAvailable]);

  useEffect(() => {
    if (!initialGraph?.legacyRecoveryAvailable) return;
    setToasts((current) => [{
      id: 'legacy-recovery-preserved',
      label: 'Saved graph opened',
      detail: 'An older local recovery draft was preserved separately and was not applied.',
      status: 'warn' as const,
      at: '',
    }, ...current].slice(0, 4));
  }, [initialGraph?.legacyRecoveryAvailable]);

  useEffect(() => {
    const el = drawerRef.current;
    if (logsCollapsed || !el) {
      setDrawerHeight(0);
      return;
    }
    const sync = () => setDrawerHeight(el.getBoundingClientRect().height);
    sync();
    const observer = new ResizeObserver(sync);
    observer.observe(el);
    return () => observer.disconnect();
    // ResizeObserver already fires on any content growth, so the drawer's row
    // counts do not belong in this dep array — and warningRows is declared far
    // below this effect, so naming it here was a TDZ error.
  }, [logsCollapsed, diagnosticsTab]);

  useEffect(() => {
    const handler = (event: Event) => {
      const custom = event as CustomEvent<{ graph?: { nodes?: any[]; edges?: any[] } }>;
      if (!custom.detail?.graph) return;
      setCanvasGraph({
        nodes: Array.isArray(custom.detail.graph.nodes) ? custom.detail.graph.nodes : [],
        edges: Array.isArray(custom.detail.graph.edges) ? custom.detail.graph.edges : [],
      });
    };
    window.addEventListener('canvas-graph-updated', handler);
    return () => window.removeEventListener('canvas-graph-updated', handler);
  }, []);

  useEffect(() => {
    const onToast = (event: Event) => {
      const d = (event as CustomEvent).detail || {};
      setToasts((prev) => [
        { id: `toast_${Date.now()}`, label: d.label || 'Done', detail: d.detail || '', status: (d.status as LogEntry['status']) || 'ok', at: '' },
        ...prev,
      ].slice(0, 4));
    };
    window.addEventListener('builder-toast', onToast);
    return () => window.removeEventListener('builder-toast', onToast);
  }, []);

  // Read inside the snapshot effect, which is keyed on [snapshot] alone. A ref
  // rather than a dependency: adding initialGraph would re-run the whole log
  // reducer whenever the route re-fetched the graph object, and this effect's job
  // is to react to snapshot TRANSITIONS.
  const graphIdentityRef = useRef<{ id?: string; revision: number }>({
    id: initialGraph?.id,
    revision: initialGraph?.currentRevision ?? 0,
  });
  graphIdentityRef.current = {
    id: initialGraph?.id,
    revision: initialGraph?.currentRevision ?? 0,
  };

  useEffect(() => {
    const prev = prevSnapshotRef.current;
    const next = snapshot;
    const additions: LogEntry[] = [];
    const now = new Date().toLocaleTimeString();

    if (prev.backendConnected !== next.backendConnected) {
      additions.push({
        id: `backend-${Date.now()}`,
        label: 'Backend',
        detail: next.backendConnected ? 'Connected successfully' : 'Disconnected',
        status: next.backendConnected ? 'ok' : 'warn',
        at: now,
      });
    }

    if (!prev.isInferringShapes && next.isInferringShapes) {
      additions.push({
        id: `infer-start-${Date.now()}`,
        label: 'Shape Inference',
        detail: `Running across ${next.nodeCount} layers`,
        status: 'info',
        at: now,
      });
    }
    if (prev.isInferringShapes && !next.isInferringShapes) {
      additions.push({
        id: `infer-end-${Date.now()}`,
        label: 'Shape Inference',
        detail: next.errorCount > 0 ? `Completed with ${next.errorCount} errors` : 'Completed successfully',
        status: next.errorCount > 0 ? 'warn' : 'ok',
        at: now,
      });
    }

    if (!prev.verifying && next.verifying) {
      additions.push({
        id: `verify-start-${Date.now()}`,
        label: 'Validation',
        detail: 'Verification started',
        status: 'info',
        at: now,
      });
    }

    if (prev.verifying && !next.verifying && next.verifyResult) {
      additions.push({
        id: `verify-end-${Date.now()}`,
        label: 'Validation',
        detail: next.verifyResult.ok
          ? 'All shapes verified'
          : `Failed at ${next.verifyResult.stage}`,
        status: next.verifyResult.ok ? 'ok' : 'error',
        at: now,
      });
      // Diagnostics drawer auto-opens ONLY on validation failure, never on success.
      if (!next.verifyResult.ok) {
        setDiagnosticsTab('errors');
        setLogsCollapsed(false);
      }
      // Remember the outcome so /projects and the home cards can say Verified /
      // N errors / Never run instead of treating all three as the same unknown.
      // Recorded against the revision it describes, so a later edit demotes it to
      // "edited since verify" rather than leaving a stale green pill.
      const identity = graphIdentityRef.current;
      if (identity.id) {
        const result = next.verifyResult;
        const firstError = result.errors?.[0];
        services?.recordVerification(identity.id, {
          ok: result.ok,
          revision: identity.revision,
          errorCount: result.errors?.length ?? 0,
          firstError: firstError?.message,
          firstErrorNodeId: firstError?.node_id,
          summary: result.summary
            ? {
                layerCount: result.summary.layer_count,
                params: result.summary.total_parameters,
                sizeMb: result.summary.model_size_mb,
              }
            : undefined,
        });
      }
    }

    if (prev.lastHistorySummary !== next.lastHistorySummary && next.lastHistorySummary !== 'No changes yet') {
      additions.push({
        id: `history-${Date.now()}`,
        label: 'Graph Change',
        detail: next.lastHistorySummary,
        status: 'info',
        at: now,
      });
    }

    if (additions.length > 0) {
      setLogs((prevLogs) => [...additions, ...prevLogs].slice(0, 60));
      // Shape inference is not toastable any more. It used to run only when the
      // user pressed Refresh, so a "Running…" / "Completed" pair per press was
      // information. It now runs after every edit, and toasting it stacked three
      // cards over the Properties panel the user was editing in. It stays in the
      // log, which is where continuous background activity belongs; a shape that
      // actually failed is already red on its own card.
      const pushable = additions.filter((entry) => entry.label !== 'Shape Inference');
      if (pushable.length > 0) setToasts((prev) => [...pushable, ...prev].slice(0, 4));
    }
    prevSnapshotRef.current = next;
  }, [snapshot, services]);

  // Show one verdict per attempt and remove it when another opening starts.
  const lastValidation = localBrowserValidation?.lastResult ?? null;
  const validationStatus = lastValidation?.status;
  const validationDetail = lastValidation?.detail;
  const validationRunId = lastValidation?.runId;
  useEffect(() => {
    setToasts((current) => {
      const retained = current.filter(toast => !toast.id.startsWith('local-browser-validate-'));
      if (!validationStatus || !validationDetail) return retained.length === current.length ? current : retained;
      return [{
        id: `local-browser-validate-${validationRunId ?? validationStatus}`,
        label: 'Validation',
        detail: validationDetail,
        status: validationStatus === 'passed'
          ? ('ok' as const)
          : validationStatus === 'failed'
            ? ('error' as const)
            : ('warn' as const),
        at: '',
      }, ...retained].slice(0, 4);
    });
  }, [validationStatus, validationDetail, validationRunId]);

  useEffect(() => {
    if (toasts.length === 0) return;
    const timer = window.setTimeout(() => {
      setToasts((prev) => prev.slice(0, -1));
    }, 2600);
    return () => window.clearTimeout(timer);
  }, [toasts]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      const isTyping = tag === 'input' || tag === 'textarea' || target?.isContentEditable;
      if (isTyping) return;
      if (!readOnlyLocalBrowser && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'enter') {
        event.preventDefault();
        dispatchBuilderCommand('verify');
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'l') {
        event.preventDefault();
        dispatchBuilderCommand('layout');
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === '0') {
        event.preventDefault();
        dispatchBuilderCommand('fit-view');
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key === '\\') {
        event.preventDefault();
        setChromeHidden((v) => !v);
        return;
      }
      if (!event.shiftKey && !event.altKey && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'p') {
        event.preventDefault();
        setQuickJumpOpen(true);
        return;
      }
      if (!readOnlyLocalBrowser && !event.metaKey && !event.ctrlKey && !event.altKey && event.shiftKey && event.key.toLowerCase() === 'a') {
        event.preventDefault();
        setPaletteOpen(true);
        return;
      }
      // Shift+C, not bare 'c': TldrawCanvas binds bare 'c' to the context-region
      // tool (kbd: 'c' in its uiOverrides), and Shift+A already sets the
      // precedent for a shifted builder verb.
      if (!readOnlyLocalBrowser && !event.metaKey && !event.ctrlKey && !event.altKey && event.shiftKey && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        dispatchBuilderCommand('toggle-connection');
        return;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [readOnlyLocalBrowser]);

  // Escape closes whichever transient overlay is open. Neither of these had a key
  // path at all: the More menu closed only by clicking its own full-screen
  // backdrop, so pressing Escape left the menu up AND left that invisible
  // `fixed inset-0` div swallowing the next click on anything else — the button
  // you pressed afterwards simply did not respond. The palette drawer had the
  // same gap without even a backdrop: Escape reached the canvas handler, quietly
  // deselected your nodes, and left the drawer open. The diagnostics drawer had
  // no key path either: at compact sizes it owns real screen space, so Escape
  // collapses it from anywhere instead of hunting for its toolbar toggle.
  //
  // Registered after the shortcut handler above on purpose, so the two listeners
  // are independent and the shortcut block keeps its own boundaries. The pattern
  // is CardMenu's, which has had it right all along. Each close hands focus back
  // to the control that opened the surface — a dismissed overlay must not strand
  // focus on BODY.
  useEffect(() => {
    if (!moreOpen && !paletteOpen && logsCollapsed) return;
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      // Quick jump is an input-first dialog with its own Escape (dismiss the
      // dialog, keep every other surface); its handler owns the key while open.
      if (quickJumpOpen) return;
      event.preventDefault();
      setMoreOpen(false);
      setPaletteOpen(false);
      if (moreOpen) moreButtonRef.current?.focus();
      if (!logsCollapsed) {
        setLogsCollapsed(true);
        (document.querySelector('[data-testid="ui2-toggle-logs"]') as HTMLElement | null)?.focus();
      }
    };
    window.addEventListener('keydown', onEscape);
    return () => window.removeEventListener('keydown', onEscape);
  }, [moreOpen, paletteOpen, logsCollapsed, quickJumpOpen]);

  // Measure the trigger while the menu is open. The toolbar is centred on the
  // stage, so its x moves whenever the window or the Inspector resizes; without
  // this the portalled menu would sit where the button used to be.
  useEffect(() => {
    if (!moreOpen) {
      setMoreAnchor(null);
      return;
    }
    const place = () => {
      const button = moreButtonRef.current;
      if (!button) return;
      const rect = button.getBoundingClientRect();
      const above = rect.top > window.innerHeight - rect.bottom;
      setMoreAnchor({ left: Math.max(8, Math.min(rect.left, window.innerWidth - 248)),
        ...(above ? { bottom: window.innerHeight - rect.top + 8 } : { top: rect.bottom + 8 }),
        maxHeight: Math.max(0, (above ? rect.top : window.innerHeight - rect.bottom) - 24) });
    };
    place();
    const dismissOutside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!moreButtonRef.current?.contains(target) && !morePanelRef.current?.contains(target)) setMoreOpen(false);
    };
    window.addEventListener('resize', place);
    document.addEventListener('pointerdown', dismissOutside);
    return () => {
      window.removeEventListener('resize', place);
      document.removeEventListener('pointerdown', dismissOutside);
    };
  }, [moreOpen]);

  const summary = snapshot.verifyResult?.summary;
  // `(n / 1e6).toFixed(1)` reads "0.0M · 0 MB" for anything under a million parameters,
  // which is every model in examples/ — TinyTransformer is ~39K. The design's own
  // helpers already handle the small end, and they are what the rest of the UI uses.
  const paramsDisplay = summary ? formatParams(summary.total_parameters) : '—';


  const memoryDisplay = summary ? formatSize(summary.model_size_mb * 1024 * 1024) : '—';
  const selectedNodeId = snapshot.selectedNode?.id || null;
  // Inside the IDE this asks the host to put the cursor on the selected node's own line —
  // the same path a double-click on the card takes. Outside it, it opens the snippet tab.
  const showSelectedLayerCode = useCallback(() => {
    if (inIde) {
      if (selectedNodeId) reportNodeClick(selectedNodeId);
      return;
    }
    setViewTab('code');
  }, [inIde, selectedNodeId]);

  const shapeRows = useMemo(() => {
    const nodes = (canvasGraph?.nodes || []).slice();
    return nodes
      .sort((a, b) => (a.position?.x || 0) - (b.position?.x || 0))
      .map((node) => ({
        id: node.id,
        name:
          operationDisplayName(node.data) || node.data?.layerType?.displayName
          || node.data?.layerType?.name
          || node.id,
        in: formatShape(node.data?.inputShape),
        out: formatShape(node.data?.outputShape),
      }));
  }, [canvasGraph]);
  const warningRows: DiagnosticRow[] = useMemo(() => {
    if (readOnlyLocalBrowser) {
      // The same static list the strip's warning drawer shows. Re-running
      // topology rules here instead ("isolated node") flags every hierarchical
      // projection's leaves and directly contradicts the backend's diagnostics.
      return snapshot.staticDiagnostics.map((diagnostic) => ({
        nodeId: diagnostic.node ?? null,
        name:
          diagnostic.node
          ?? (diagnostic.line
            ? `${diagnostic.file}:${diagnostic.line}`
            : diagnostic.file || 'workspace'),
        message: `${DIAGNOSTIC_LABELS[diagnostic.code]} — ${diagnostic.detail}`,
        severity: diagnostic.severity,
      }));
    }
    return graphWarnings(canvasGraph?.nodes || [], canvasGraph?.edges || []);
  }, [readOnlyLocalBrowser, snapshot.staticDiagnostics, canvasGraph]);

  // Local page: the only issues that exist are the projection's static
  // diagnostics — runtime error counts would claim a backend this page never runs.
  const issueCount = readOnlyLocalBrowser
    ? warningRows.length
    : snapshot.errorCount + warningRows.length;

  // Situational tools, moved off the toolbar surface into the More menu. Each
  // dispatches exactly the command its old toolbar button did.
  const moreTools = useMemo(
    () => {
      const tools = [
      { label: 'Select tool', icon: MousePointer2, testid: 'ui2-action-select', run: () => dispatchBuilderCommand('select-mode'), hint: 'Return to the select tool' },
      {
        label: 'Refresh shapes', icon: Sparkles, testid: 'ui2-action-refresh-shapes',
        run: () => dispatchBuilderCommand('refresh-shapes'),
        disabled: !snapshot.backendConnected || snapshot.nodeCount === 0 || snapshot.isInferringShapes,
        hint: !snapshot.backendConnected ? 'Backend not connected' : snapshot.nodeCount === 0 ? 'Add layers first' : 'Re-run shape inference',
      },
      { label: 'Blocks view', icon: Boxes, testid: 'ui2-action-view-blocks', run: () => dispatchBuilderCommand('view-overview'), hint: 'Collapse modules to blocks and fit the graph in the canvas' },
      { label: 'Expanded view', icon: Expand, testid: 'ui2-action-view-expanded', run: () => dispatchBuilderCommand('view-expanded'), hint: 'Expand modules to layers' },
      {
        label: svgOverview ? 'Interactive canvas' : 'SVG overview', icon: ImageIcon, testid: 'ui2-action-svg-overview',
        run: () => { const next = !svgOverview; setSvgOverview(next); dispatchBuilderCommand('toggle-expanded-svg', { active: next }); },
        hint: svgOverview ? 'Return to the interactive tldraw canvas'
          : 'Draw the graph as one SVG image — far lighter for a large expanded graph',
      },
      {
        label: 'Focus selected', icon: Crosshair, testid: 'ui2-action-focus',
        run: () => dispatchBuilderCommand('view-focus'),
        disabled: !snapshot.selectedNode,
        hint: !snapshot.selectedNode ? 'Select a module/layer first' : 'Zoom to the selection',
      },
      {
        label: 'Copy canvas as SVG', icon: ImageDown, testid: 'ui2-action-export-svg',
        run: () => dispatchBuilderCommand('export-image', { format: 'svg' }),
        disabled: snapshot.nodeCount === 0,
        hint: snapshot.nodeCount === 0 ? 'Nothing on the canvas yet' : 'Copy the graph as SVG, for a README or an issue',
      },
      {
        label: 'Download SVG', icon: ImageDown, testid: 'ui2-action-download-svg',
        run: () => dispatchBuilderCommand('export-image', { format: 'svg', save: true }),
        disabled: snapshot.nodeCount === 0,
        // In the browser this is a download; a VS Code webview blocks that, so the
        // host writes the file into tensorviz-exports/ instead (hostServices.saveFile).
        hint: snapshot.nodeCount === 0 ? 'Nothing on the canvas yet' : 'Save the graph as an SVG file',
      },
      {
        label: 'Copy canvas as PNG', icon: ImageDown, testid: 'ui2-action-export-png',
        run: () => dispatchBuilderCommand('export-image', { format: 'png' }),
        disabled: snapshot.nodeCount === 0,
        hint: snapshot.nodeCount === 0 ? 'Nothing on the canvas yet' : 'Copy the graph as an image',
      },
      { label: 'Zoom in', icon: Plus, testid: 'ui2-action-zoom-in', run: () => dispatchBuilderCommand('zoom-in'), hint: 'Or scroll / pinch on the canvas' },
      { label: 'Zoom out', icon: Minus, testid: 'ui2-action-zoom-out', run: () => dispatchBuilderCommand('zoom-out'), hint: 'Or scroll / pinch on the canvas' },
      {
        label: isCanvasFullscreen ? 'Exit fullscreen' : 'Fullscreen canvas', icon: Maximize2, testid: 'ui2-action-fullscreen',
        run: () => setIsCanvasFullscreen((v) => !v), hint: 'Fill the window with the canvas',
      },
      {
        label: 'Group as module', icon: GitBranchPlus,
        run: () => dispatchBuilderCommand('group-module'),
        disabled: snapshot.selectedNodeCount < 2,
        hint: snapshot.selectedNodeCount < 2
          ? 'Select two or more layers to group them (G)'
          : 'Group the selected layers into a reusable nn.Module (G)',
      },
      {
        label: 'Ungroup module', icon: Ungroup, testid: 'ui2-action-ungroup',
        run: () => dispatchBuilderCommand('ungroup-module'),
        disabled: snapshot.selectedNode?.nodeType !== 'moduleNode',
        hint: snapshot.selectedNode?.nodeType !== 'moduleNode'
          ? 'Select a module block to break it back into layers'
          : 'Break this module back into its layers, keeping its connections',
      },
      { label: 'Version history', icon: History, run: () => setHistoryOpen(true), hint: 'Version history' },
      {
        label: 'Save as template', icon: Save,
        run: () => {
          const name = hostServices.prompt('Save the current model as a reusable template. Name:');
          if (name && name.trim()) dispatchBuilderCommand('save-as-template', { name: name.trim() });
        },
        disabled: snapshot.nodeCount === 0,
        hint: 'Save this model as a reusable template',
      },
      ];
      if (!readOnlyLocalBrowser) return tools;
      const readOnlyTools = new Set([
        'Select tool',
        'Blocks view',
        'Expanded view',
        'SVG overview',
        'Interactive canvas',
        'Focus selected',
        'Download SVG',
        'Zoom in',
        'Zoom out',
        'Fullscreen canvas',
        'Exit fullscreen',
      ]);
      return [{ label: 'Auto layout', icon: Wrench, testid: 'ui2-action-layout',
        run: () => dispatchBuilderCommand('layout'), hint: 'Arrange the canvas without changing the model (⌘L)' },
        ...tools.filter((tool) => readOnlyTools.has(tool.label))];
    },
    [readOnlyLocalBrowser, snapshot.backendConnected, snapshot.nodeCount, snapshot.isInferringShapes, snapshot.selectedNode, snapshot.selectedNodeCount, isCanvasFullscreen, svgOverview],
  );

  const nodeList = useMemo(() => {
    const nodes = (canvasGraph?.nodes || []).slice();
    return nodes
      .sort((a, b) => (a.position?.x || 0) - (b.position?.x || 0))
      .map((node) => ({
        id: node.id,
        path: node.data.captureRef?.kind === 'invocation' && node.data.captureLabel ? node.data.captureLabel : node.id,
        title: operationDisplayName(node.data) ?? [(!readOnlyLocalBrowser || node.data.captureRef?.kind !== 'invocation') && node.data?.captureLabel,
          node.data?.layerType?.displayName
          || node.data?.layerType?.name
          || node.data?.layerType?.id
          || node.type
          || node.id].filter(Boolean).join(' · '),
        searchText: [node.data.captureLabel, node.data.layerType.name, node.data.layerType.id].filter(Boolean).join(' '),
      }));
  }, [canvasGraph, readOnlyLocalBrowser]);
  const quickJumpMatches = useMemo(
    () => findQuickJumpNodes(nodeList, quickJumpQuery, readOnlyLocalBrowser ? nodeList.length : 20),
    [nodeList, quickJumpQuery, readOnlyLocalBrowser],
  );
  const advancedSearch = useMemo(() => readOnlyLocalBrowser && /^(type|shape)=|^attr:/i.test(quickJumpQuery.trim())
    ? searchGraph(canvasGraph?.nodes || [], quickJumpQuery) : null, [readOnlyLocalBrowser, canvasGraph, quickJumpQuery]);
  const searchMatches = useMemo(() => {
    if (!advancedSearch) return quickJumpMatches.map(node => ({ ...node, detail: '' }));
    const titles = new Map(nodeList.map(node => [node.id, node.title]));
    const paths = new Map(nodeList.map(node => [node.id, node.path]));
    return advancedSearch.groups.map(group => ({ id: group.visibleId,
      path: paths.get(group.visibleId) ?? group.visibleId,
      title: titles.get(group.visibleId) ?? group.visibleId,
      detail: `${group.hits.length} ${group.hits.length === 1 ? 'match' : 'matches'} · ${group.hits[0].matched}` }));
  }, [advancedSearch, nodeList, quickJumpMatches]);
  const currentSearchPage = Math.min(searchPage, Math.max(0, Math.ceil(searchMatches.length / 20) - 1));
  const searchStart = readOnlyLocalBrowser ? currentSearchPage * 20 : 0;
  const visibleSearchMatches = readOnlyLocalBrowser ? searchMatches.slice(searchStart, searchStart + 20) : searchMatches;
  React.useLayoutEffect(() => { searchListRef.current?.scrollTo?.(0, 0); }, [currentSearchPage, quickJumpQuery]);
  useEffect(() => {
    if (!readOnlyLocalBrowser) return;
    graphSearchQuery.set(quickJumpOpen && advancedSearch ? quickJumpQuery : '');
    return () => { graphSearchQuery.set(''); };
  }, [readOnlyLocalBrowser, quickJumpOpen, advancedSearch, quickJumpQuery]);

  const codePreview = useMemo(() => {
    const selected = snapshot.selectedNode;
    if (!selected) {
      return '# Select a layer to inspect generated snippet\n';
    }
    const layerCtor = torchClassForLayer(selected.layerType, selected.layerName);
    if (!layerCtor) {
      // No invented class name. An Input or Output node is a graph endpoint with no
      // torch class, and for anything this file does not recognise the truthful
      // answer is that it does not know.
      return [
        `# ${selected.layerName} (${selected.layerType})`,
        '#',
        '# This preview cannot name a torch class for this layer, so it will not',
        '# guess one. Input and Output are graph endpoints with no torch class at',
        '# all; other layers resolve through the backend op registry, which this',
        '# panel cannot see.',
        '#',
        '# Use Export Code for the real generated module.',
        '',
      ].join('\n');
    }
    const paramEntries = Object.entries(selected.parameters || {});
    // Every parameter, not the first six. The old slice quietly dropped the
    // seventh onward, which is the same class of untruth as the invented class
    // name: a snippet that looks complete and is not.
    const args = paramEntries
      .map(([key, value]) => `${key}=${formatValue(value)}`)
      .join(', ');
    return [
      `class ${selected.layerName}(nn.Module):`,
      '    def __init__(self):',
      '        super().__init__()',
      `        self.layer = ${layerCtor}(${args})`,
      '',
      '    def forward(self, x):',
      '        return self.layer(x)',
    ].join('\n');
  }, [snapshot.selectedNode]);

  const copyCodePreview = async () => {
    try {
      await hostServices.copyText(codePreview);
      setCopiedCode(true);
      window.setTimeout(() => setCopiedCode(false), 1500);
    } catch (error) {
      console.warn('Failed to copy code preview:', error);
    }
  };

  // The Inspector and Properties sidebar is gone (2026-09-15). It opened on a
  // pointer-down that tldraw treats as a selection, so dragging a node across the
  // canvas made a panel appear over the thing being dragged — and everything it
  // read out was already on the card or the toolbar. What was only there moved:
  // Export Package to the toolbar beside Export Code, because a package belongs to
  // the model and not to whichever node is selected. Parameter editing did not
  // move, because it already exists as SourceEditForm, which goes through the
  // native review every other source change uses instead of writing the canvas.

  const moreControl = readOnlyLocalBrowser ? <LocalCanvasMenu actions={moreTools} nodes={canvasGraph?.nodes || []}
    viewMode={snapshot.canvasViewMode} svgOverview={svgOverview} canvasTool={snapshot.canvasToolId} /> : (
    <div className="relative flex-shrink-0">
      <button
        ref={moreButtonRef}
        className={`ui2-tool ${moreOpen ? 'active' : ''}`}
        onClick={() => setMoreOpen((v) => !v)}
        aria-label="More tools"
        aria-haspopup="menu"
        aria-expanded={moreOpen}
        title="More tools"
        data-testid="ui2-action-more"
      >
        <MoreHorizontal size={15} />
      </button>
      {moreOpen && moreAnchor && createPortal(
        <>
          <div className="fixed inset-0" style={{ zIndex: readOnlyLocalBrowser ? 1199 : 49 }} onClick={() => setMoreOpen(false)} />
          <div
            ref={morePanelRef}
            className={`ui2-panel${readOnlyLocalBrowser ? ' local-canvas-popover' : ''}`}
            role="menu"
            style={{
              position: 'fixed',
              ...moreAnchor,
              minWidth: Math.min(232, window.innerWidth - 16),
              maxWidth: 'calc(100vw - 16px)',
              overflowY: 'auto',
              padding: 6,
              zIndex: readOnlyLocalBrowser ? 1200 : 50,
            }}
          >
            <PassThroughToggle menuItem className="ui2-menu-item" nodes={canvasGraph?.nodes || []}
              onToggle={shown => { dispatchBuilderCommand('show-unchanged-calls', { shown }); setMoreOpen(false); }} />
            {moreTools.map((tool) => (
              <button
                key={tool.label}
                role="menuitem"
                className="ui2-menu-item"
                onClick={() => { tool.run(); setMoreOpen(false); }}
                disabled={tool.disabled}
                title={tool.hint}
                data-testid={tool.testid}
              >
                <tool.icon size={14} /> {tool.label}
              </button>
            ))}
          </div>
        </>,
        document.body
      )}
    </div>
  );

  return (
    <WorkbenchShell
      centerScrollable={false}
      chrome={readOnlyLocalBrowser ? 'local' : inIde ? 'ide' : 'minimal'}
      saveState={snapshot.saveState}
      ideStatus={snapshot.ideStatus}
      modelName={snapshot.graphName}
      onRenameModel={readOnlyLocalBrowser ? undefined : (name) => dispatchBuilderCommand('rename-model', { name })}
    >
      <div className={`h-full min-h-0 flex flex-col ui2-builder-stage${isCanvasFullscreen ? ' ui2-builder-stage-fullscreen' : ''}${chromeHidden ? ' ui2-chrome-hidden' : ''}`}>
        {/* Two bottom bars did not fit. When a project is open the toolbar DOCKS
            into the status bar's right end, icon-only — every control keeps its
            title, so hovering says what it does, and the labels come back the
            moment there is no status bar to share with. The standalone builder is
            untouched: no project, no status bar, floating toolbar as before. */}
        {!readOnlyLocalBrowser && (() => {
          const bar = (
        <div
          className="ui2-builder-toolbar"
          style={drawerHeight ? { bottom: drawerHeight + 12 } : undefined}
        >
          <div className="ui2-view-tabs">
            <button
              className={viewTab === 'graph' ? 'active' : ''}
              onClick={() => setViewTab('graph')}
              data-testid="ui2-view-graph"
              title="Graph view"
            >
              <Workflow size={14} /> <span className="ui2-tool-label">Graph View</span>
            </button>
            <button
              className={!inIde && viewTab === 'code' ? 'active' : ''}
              onClick={showSelectedLayerCode}
              disabled={inIde && !selectedNodeId}
              data-testid="ui2-view-code"
              title={
                inIde
                  ? 'Jump to the line that declares the selected layer'
                  : 'Show the generated snippet for the selected layer'
              }
            >
              <FileCode2 size={14} /> <span className="ui2-tool-label">{codeTabLabel}</span>
            </button>
          </div>

          {/* Only the controls used on nearly every edit stay on the surface.
              Everything situational lives behind More — still in the DOM, one
              click away — so the bar is scannable instead of a wall of 19 icons. */}
          <span className="ui2-toolbar-divider-vertical" />
          <GraphDetailToggle level={snapshot.captureDetailLevel} />
          {!readOnlyLocalBrowser && <button
            className={`ui2-tool ${paletteOpen ? 'active' : ''}`}
            onClick={() => setPaletteOpen((value) => !value)}
            aria-label={paletteOpen ? 'Hide layer palette' : 'Show layer palette'}
            title={paletteOpen ? 'Hide layer palette' : 'Add a layer (Shift+A)'}
          >
            <Layers3 size={15} />
          </button>}
          <button className="ui2-tool" onClick={() => dispatchBuilderCommand('layout')} aria-label="Auto layout" title="Auto layout (⌘L)" data-testid="ui2-action-layout"><Wrench size={15} /></button>
          <button className="ui2-tool" onClick={() => dispatchBuilderCommand('fit-view')} aria-label="Fit graph to view" title="Fit to view (⌘0)" data-testid="ui2-action-fit-view"><Scan size={15} /></button>
          {!readOnlyLocalBrowser && <>
          <span className="ui2-toolbar-divider-vertical" />
          <button className="ui2-tool" onClick={() => dispatchBuilderCommand('undo')} disabled={snapshot.historyState.undo_count < 1} aria-label="Undo" title="Undo" data-testid="ui2-action-undo"><RotateCcw size={15} /></button>
          <button className="ui2-tool" onClick={() => dispatchBuilderCommand('redo')} disabled={snapshot.historyState.redo_count < 1} aria-label="Redo" title="Redo" data-testid="ui2-action-redo"><RotateCw size={15} /></button>
          <button
            onClick={() => dispatchBuilderCommand('toggle-connection')}
            className="ui2-tool"
            disabled={snapshot.selectedNodeCount !== 2}
            aria-label="Connect or disconnect the two selected layers"
            title={snapshot.selectedNodeCount !== 2 ? 'Select exactly two layers to connect or disconnect (Shift+C)' : 'Connect / disconnect the two selected layers (Shift+C)'}
            data-testid="ui2-action-toggle-connection"
          >
            <GitBranchPlus size={15} />
          </button>
          <button className="ui2-tool" onClick={() => dispatchBuilderCommand('delete-selected')} disabled={!snapshot.selectedNode} aria-label="Delete selected" title={!snapshot.selectedNode ? 'Select a layer first' : 'Delete selected layer'} data-testid="ui2-action-delete-selected"><Trash2 size={15} /></button>
          </>}

          <span className="ui2-toolbar-divider-vertical" />
          {moreControl}
          <button
            onClick={() => setLogsCollapsed((v) => !v)}
            className={`ui2-tool ${!logsCollapsed ? 'active' : ''}${issueCount !== 0 ? ' ui2-tool-warn' : ''}`}
            aria-label={logsCollapsed
              ? readOnlyLocalBrowser ? 'Show static resolution warnings' : 'Show diagnostics'
              : 'Hide diagnostics'}
            title={logsCollapsed
              ? readOnlyLocalBrowser ? 'Static resolution warnings' : 'Logs, shapes, errors, warnings'
              : 'Hide diagnostics'}
            data-testid="ui2-toggle-logs"
          >
            <ClipboardList size={15} />
            {issueCount > 0 && <span className="ui2-tool-badge">{issueCount}</span>}
          </button>

          {/* Version history had no affordance anywhere in the builder chrome — it
              existed only as a command-palette entry, so a new user following the
              guide could not find it at all. */}
          {!readOnlyLocalBrowser && <button
            className="ui2-tool-button"
            onClick={() => setHistoryOpen(true)}
            aria-label="Version history"
            title="Version history"
            data-testid="ui2-toggle-history"
          >
            <History size={15} />
          </button>}

          <div className="ml-auto flex items-center gap-2">
            <span className="ui2-muted text-xs whitespace-nowrap ui2-tool-label" title="Model summary">
              {snapshot.nodeCount} layer{snapshot.nodeCount === 1 ? '' : 's'}
              {summary && ` · ${paramsDisplay} · ${memoryDisplay}`}
            </span>
            {!readOnlyLocalBrowser && <button className="ui2-primary-action" onClick={() => dispatchBuilderCommand('export-code')} data-testid="ui2-action-export-code" title="Export the model as Python code">
              <Code2 size={14} /> <span className="ui2-tool-label">Export Code</span>
            </button>}
            {/* Re-homed from the Inspector, which is gone. A package is a property of
                the MODEL, not of whichever node happens to be selected, so it belongs
                beside Export Code and not on a card. It had no other trigger: removing
                the panel without this would have made it unreachable. */}
            {!readOnlyLocalBrowser && <button
              className="ui2-primary-action"
              onClick={() => dispatchBuilderCommand('export-package')}
              disabled={!snapshot.backendConnected || snapshot.nodeCount === 0 || snapshot.exporting}
              data-testid="ui2-action-export-package"
              title="Export a runnable package"
            >
              <Package size={14} /> <span className="ui2-tool-label">Export Package</span>
            </button>}
            {!readOnlyLocalBrowser && <button
              className="ui2-primary-action ui2-primary-action-green"
              onClick={() => dispatchBuilderCommand('verify')}
              disabled={!snapshot.backendConnected || snapshot.nodeCount === 0 || snapshot.verifying}
              data-testid="ui2-action-validate"
              title="Validate the model"
            >
              <Play size={14} /> <span className="ui2-tool-label">Validate</span>
            </button>}

          </div>
        </div>
          );
          // Docked = a real flex child of the status bar, via a portal into its slot.
          // Not an overlay: an absolutely-positioned bar sat over the status text on
          // every narrow screen, whatever its z-index said.
          return bar;
        })()}

        {readOnlyLocalBrowser && <LocalCanvasNavigation onFind={() => setQuickJumpOpen(true)}
          findButtonRef={findButtonRef}
          detailLevel={snapshot.captureDetailLevel}
          layout={snapshot.layoutChoice && snapshot.layoutDirection
            ? { choice: snapshot.layoutChoice, direction: snapshot.layoutDirection } : undefined}
          onReveal={showSelectedLayerCode} hasSelection={!!selectedNodeId} nodeCount={snapshot.nodeCount}
          more={moreControl} fullscreen={isCanvasFullscreen} />}

        <div className="ui2-toast-stack" aria-live="polite">
          {toasts.map((toast) => (
            <div key={toast.id} className={`ui2-toast ui2-toast-${toast.status}`}>
              <div className="ui2-toast-label">{toast.label}</div>
              <div className="ui2-toast-detail">{toast.detail}</div>
            </div>
          ))}
        </div>

        {/* Canvas stays mounted across the Graph<->Code toggle — hidden via CSS,
            never unmounted — so tldraw camera/zoom/selection/undo survive a
            Code View peek. */}
        <div
          className="min-h-0 flex-1 relative overflow-hidden"
          style={{ display: viewTab === 'graph' ? undefined : 'none' }}
        >
          {!readOnlyLocalBrowser && paletteOpen && (
            <div className="ui2-palette-drawer">
              <BuilderPaletteV2
                onAddLayer={(layerType, options) => {
                  dispatchBuilderCommand('add-layer', { layerType, options });
                  setPaletteOpen(false);
                }}
              />
            </div>
          )}
          {/* `left` rather than `inset: 0`: an expanded Inspector reserves real
              width, so .tl-container is genuinely narrower and tldraw stops
              putting cards where they cannot be clicked. See CV-11 in
              ui2.tokens.css. */}
          <div
            className="ui2-canvas-frame ui2-reactflow-skin"
            data-testid="builder-canvas"
            style={{
              position: 'absolute',
              top: 0,
              left: 'var(--ui2-canvas-inset-left, 0px)',
              // Reserved, not floated over. The status bar is opaque and 26px
              // tall; leaving the canvas underneath it puts cards in a strip
              // where elementsFromPoint returns the bar and no .tl-shape, which
              // is CV-11 again along the other edge.
              bottom: 'var(--ui2-canvas-inset-bottom, 0px)',
              right: 0,
            }}
          >
            <ModelBuilder
              initialGraph={initialGraph}
              draftStore={draftStore}
              autoOpenImport={autoOpenImport}
              hideSidebar
              compactStatus
              hideStatusIndicators
              onBuilderSnapshot={setSnapshot}
              onGraphCreated={onGraphCreated}
              enableExternalCommands
            />
            {/* An empty canvas used to be a black rectangle with no way in. The
                bottom bar is icon-only, Validate is disabled and silent, and the
                palette is behind an unadvertised Shift+A — so the first and most
                prominent start option on Home ("Blank graph") dropped a newcomer
                into a screen with nothing to act on. Measured on the live app:
                6m50s to escape it, against ~11s via a template.

                Rendered as an overlay rather than inside the canvas so it cannot
                interfere with tldraw's own layers, and pointer-events are off on
                the container so it never intercepts a drag onto the canvas. */}
            {snapshot.nodeCount === 0 && !readOnlyLocalBrowser && (
              <div
                data-testid="builder-empty-state"
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                  pointerEvents: 'none',
                  textAlign: 'center',
                  padding: 24,
                }}
              >
                <p style={{ fontSize: 15, color: 'var(--ui2-text)', margin: 0 }}>
                  This canvas is empty.
                </p>
                <p style={{ fontSize: 13, color: 'var(--ui2-text-dim)', margin: 0, maxWidth: 380 }}>
                  Add layers one at a time, and each one wires itself to the end of the chain.
                </p>
                <div style={{ display: 'flex', gap: 8, pointerEvents: 'auto' }}>
                  <button
                    className="ui2-chip"
                    onClick={() => setPaletteOpen(true)}
                    data-testid="builder-empty-add-layer"
                  >
                    Add a layer
                    <kbd style={{ marginLeft: 6, opacity: 0.6 }}>&#8679;A</kbd>
                  </button>
                  {/* Home is not mounted in the IDE bundle, so this button would
                      navigate nowhere and the line below it would be untrue. */}
                  {!inIde && (
                    <button className="ui2-chip" onClick={() => services?.navigate('/')}>
                      Browse templates
                    </button>
                  )}
                </div>
                {!inIde && (
                  <p style={{ fontSize: 12, color: 'var(--ui2-text-dim)', margin: '2px 0 0' }}>
                    Templates are on Home, and are the quickest way to a working model.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
        {viewTab === 'code' && (
          <div className="ui2-panel min-h-0 flex-1 overflow-auto p-4 mx-7 mb-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg">Selected Layer Code</h3>
              <button className="ui2-chip" onClick={copyCodePreview}>
                {copiedCode ? <Check size={12} className="inline mr-1" /> : <Copy size={12} className="inline mr-1" />}
                {copiedCode ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="ui2-code-block" style={{ margin: 0 }}>{codePreview}</pre>
          </div>
        )}

        {/* Diagnostics are no longer a second permanent bar — the drawer only
            exists while it is open, and is opened from the single toolbar.
            pr-28 keeps the bottom-right clear of tldraw's 96x32 watermark, which
            paints above us (z-index 200) with pointer-events:all and would
            swallow clicks on the right-most controls. It may not be hidden or
            obscured without a licence. */}
        {!readOnlyLocalBrowser && !logsCollapsed && (
        <div ref={drawerRef} className="ui2-diagnostics-drawer p-3 pr-28">
          <div className="flex items-center gap-2 mb-3">
            {readOnlyLocalBrowser && (
              <span className="text-sm font-semibold">
                Static resolution warnings ({warningRows.length})
              </span>
            )}
            {!readOnlyLocalBrowser && <>
            <button
              className={`ui2-chip${diagnosticsTab === 'logs' ? ' ui2-chip-active' : ''}`}
              onClick={() => setDiagnosticsTab('logs')}
            >
              <Layers3 size={14} className="inline mr-1" /> Logs
            </button>
            <button
              className={`ui2-chip${diagnosticsTab === 'shapes' ? ' ui2-chip-active' : ''}`}
              onClick={() => setDiagnosticsTab('shapes')}
            >
              <Cpu size={14} className="inline mr-1" /> Shape Table
            </button>
            <button
              className={`ui2-chip${diagnosticsTab === 'errors' ? ' ui2-chip-active' : ''}`}
              onClick={() => setDiagnosticsTab('errors')}
            >
              <MemoryStick size={14} className="inline mr-1" /> Errors ({snapshot.errorCount})
            </button>
            <button
              className={`ui2-chip${diagnosticsTab === 'warnings' ? ' ui2-chip-active' : ''}`}
              onClick={() => setDiagnosticsTab('warnings')}
            >
              <CheckCircle2 size={14} className="inline mr-1" /> Warnings ({warningRows.length})
            </button>
            </>}
            <div className="ml-auto flex items-center gap-2">
              {!readOnlyLocalBrowser && diagnosticsTab === 'logs' && (
                <button className="ui2-chip" onClick={() => setLogs([])}>Clear Logs</button>
              )}
              <button
                className="ui2-chip"
                onClick={() => setLogsCollapsed(true)}
                aria-label="Hide diagnostics"
                title="Hide diagnostics"
              >
                <ChevronDown size={14} />
              </button>
            </div>
          </div>

          {(
            <>
          {!readOnlyLocalBrowser && diagnosticsTab === 'logs' && (
            <div className="space-y-2 max-h-40 overflow-auto">
              {logs.map((entry) => (
                <div key={entry.id} className="ui2-panel px-3 py-2 text-sm flex items-center gap-3">
                  <span className="ui2-muted text-xs">{entry.at}</span>
                  <span style={{ color: entry.status === 'error' ? 'var(--ui2-red)' : entry.status === 'warn' ? 'var(--ui2-orange)' : entry.status === 'ok' ? 'var(--ui2-green)' : 'var(--ui2-accent)' }}>{entry.label}</span>
                  <span className="ui2-muted truncate">{entry.detail}</span>
                </div>
              ))}
              {logs.length === 0 && <p className="ui2-muted text-sm">No logs yet.</p>}
            </div>
          )}

          {!readOnlyLocalBrowser && diagnosticsTab === 'shapes' && (
            <div className="ui2-panel px-2 py-2 overflow-auto">
              {shapeRows.length === 0 ? (
                <div className="text-sm ui2-muted px-2 py-1">No shape rows available yet.</div>
              ) : (
                <table className="ui2-shape-table">
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th>Input</th>
                      <th>Output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shapeRows.map((row) => (
                      <tr key={row.id}>
                        <td>
                          <button
                            className="ui2-shape-link"
                            onClick={() => { dispatchBuilderCommand('focus-node', { nodeId: row.id }); }}
                          >
                            {row.name}
                          </button>
                        </td>
                        <td>{row.in}</td>
                        <td>{row.out}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {!readOnlyLocalBrowser && diagnosticsTab === 'errors' && (
            <div className="space-y-2 max-h-44 overflow-auto">
              {snapshot.repairCount > 0 && (
                // A complete repair implementation had sat unreferenced behind a
                // comment saying it was "not yet wired to a control", with an
                // eslint-disable holding the build green over it — while a
                // beginner got a raw torch message and no way to act on it.
                <button
                  type="button"
                  className="ui2-primary-action w-full"
                  onClick={() => dispatchBuilderCommand('apply-repairs')}
                  title="Apply the shape corrections the validator suggested"
                >
                  Fix {snapshot.repairCount} shape problem
                  {snapshot.repairCount === 1 ? '' : 's'}
                </button>
              )}
              {snapshot.nodeIssues.length > 0 ? snapshot.nodeIssues.map((issue, index) => (
                <button
                  key={`${issue.nodeId}-${index}`}
                  className="w-full ui2-panel px-3 py-2 text-sm text-left"
                  onClick={() => dispatchBuilderCommand('focus-node', { nodeId: issue.nodeId })}
                  title="Show this layer on the canvas"
                >
                  <div className="flex items-center gap-2">
                    <span className="ui2-status-dot" style={{ background: 'var(--ui2-red)' }} />
                    <span className="font-medium">{issue.layerName}</span>
                  </div>
                  <div className="ui2-muted">{issue.message}</div>
                </button>
              )) : (
                <div className="ui2-panel px-3 py-2 text-sm ui2-muted">No active errors.</div>
              )}
            </div>
          )}

          {(readOnlyLocalBrowser || diagnosticsTab === 'warnings') && (
            <div className="space-y-2 max-h-44 overflow-auto">
              {warningRows.length > 0 ? warningRows.map((warning, index) => {
                const rowBody = (
                  <>
                    <div className="flex items-center gap-2">
                      <span
                        className="ui2-status-dot"
                        style={{
                          background:
                            warning.severity === 'error'
                              ? 'var(--ui2-red)'
                              : warning.severity === 'info'
                                ? 'var(--ui2-text-dim)'
                                : 'var(--ui2-orange)',
                        }}
                      />
                      <span className="font-medium">{warning.name}</span>
                    </div>
                    <div className="ui2-muted">{warning.message}</div>
                  </>
                );
                return warning.nodeId ? (
                  <button
                    key={`${warning.nodeId}-${index}`}
                    className="w-full ui2-panel px-3 py-2 text-sm text-left"
                    onClick={() => dispatchBuilderCommand('focus-node', { nodeId: warning.nodeId as string })}
                    title="Show this layer on the canvas"
                  >
                    {rowBody}
                  </button>
                ) : (
                  <div key={`static-${index}`} className="w-full ui2-panel px-3 py-2 text-sm">
                    {rowBody}
                  </div>
                );
              }) : (
                <div className="ui2-panel px-3 py-2 text-sm ui2-muted">
                  {readOnlyLocalBrowser
                    ? 'No static resolution warnings.'
                    : 'No structural warnings detected.'}
                </div>
              )}
            </div>
          )}
            </>
          )}
        </div>
        )}

        {quickJumpOpen && (() => {
          const search = <div className={`ui2-quickjump-backdrop${readOnlyLocalBrowser ? ' local-canvas-search-backdrop' : ''}`} onClick={() => { setQuickJumpOpen(false); setQuickJumpQuery(''); }}>
            <div className={`ui2-quickjump${readOnlyLocalBrowser ? ' local-canvas-search' : ''}`} role="dialog" aria-label="Find a layer"
              style={readOnlyLocalBrowser ? searchPosition ?? { visibility: 'hidden' } : undefined} onClick={(e) => e.stopPropagation()}
              onKeyDown={event => {
                if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setQuickJumpOpen(false); setQuickJumpQuery(''); }
                if (!readOnlyLocalBrowser || !['ArrowDown', 'ArrowUp'].includes(event.key)) return;
                const buttons = Array.from(searchListRef.current?.querySelectorAll<HTMLButtonElement>('button') ?? []);
                if (!buttons.length) return;
                event.preventDefault();
                const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
                const index = current < 0 ? (event.key === 'ArrowDown' ? 0 : buttons.length - 1)
                  : (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
                buttons[index].focus();
              }}>
              {readOnlyLocalBrowser && <div className="local-canvas-search-heading"><Search size={14} aria-hidden="true" />Find in graph<kbd>esc</kbd></div>}
              <input
                autoFocus
                className="ui2-quickjump-input"
                placeholder={readOnlyLocalBrowser ? 'Find layer · type=Conv2d · shape=256' : 'Jump to node…'}
                value={quickJumpQuery}
                onChange={(e) => setQuickJumpQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    setQuickJumpOpen(false);
                    setQuickJumpQuery('');
                  } else if (e.key === 'Enter' && visibleSearchMatches[0]) {
                    dispatchBuilderCommand('focus-node', { nodeId: visibleSearchMatches[0].id });
                    setQuickJumpOpen(false);
                    setQuickJumpQuery('');
                  }
                }}
                aria-label="Quick jump to node"
              />
              <div className="ui2-quickjump-list" ref={searchListRef}>
                {advancedSearch?.error ? <div className="local-canvas-search-error" role="alert">Invalid pattern: {advancedSearch.error}</div> : searchMatches.length === 0 ? (
                  <div className="ui2-muted text-sm px-3 py-2">No matching nodes.</div>
                ) : (
                  visibleSearchMatches.map((n) => (
                    <button
                      key={n.id}
                      className="ui2-quickjump-item"
                      aria-label={readOnlyLocalBrowser ? `${n.title} — ${n.path ?? n.id}${n.path && n.path !== n.id ? ` — ${n.id}` : ''}${n.detail ? ` — ${n.detail}` : ''}` : `${n.id} — ${n.title}`}
                      onClick={() => {
                        dispatchBuilderCommand('focus-node', { nodeId: n.id });
                        setQuickJumpOpen(false);
                        setQuickJumpQuery('');
                      }}
                    >
                      {readOnlyLocalBrowser ? <><span className="ui2-quickjump-name">{n.title}</span>
                        <span className="local-canvas-search-path">{n.path ?? n.id}</span>
                        {n.detail && <span className="local-canvas-search-context">{n.detail}</span>}</> : <>
                        <span className="ui2-quickjump-id">{n.id}</span>
                        <span className="ui2-quickjump-type ui2-muted">{n.title}</span>
                      </>}
                    </button>
                  ))
                )}
              </div>
              {readOnlyLocalBrowser && <div className="local-canvas-search-footer" aria-live="polite">
                <span>{searchMatches.length ? `${searchStart + 1}–${searchStart + visibleSearchMatches.length} of ${searchMatches.length}` : '0'} results
                  {advancedSearch && !advancedSearch.error ? ` · ${advancedSearch.hits.length} matching nodes` : ''}</span>
                {searchMatches.length > 20 && <div className="local-canvas-search-pages" role="group" aria-label="Search result pages">
                  <button type="button" disabled={currentSearchPage === 0} onClick={() => setSearchPage(currentSearchPage - 1)}>Previous</button>
                  <button type="button" disabled={searchStart + 20 >= searchMatches.length} onClick={() => setSearchPage(currentSearchPage + 1)}>Next</button>
                </div>}
                {searchMatches.length > 0 && <span>↑ ↓ Select · ↵ Jump</span>}</div>}
            </div>
          </div>;
          return readOnlyLocalBrowser ? createPortal(search, document.body) : search;
        })()}
        {/* The local-browser page is source-authoritative and read-only: there is
            no saved graph and its versions panel must never fetch /api/projects. */}
        {!readOnlyLocalBrowser && HistoryPanel && (
          <HistoryPanel open={historyOpen} onClose={() => setHistoryOpen(false)} />
        )}

      </div>
    </WorkbenchShell>
  );
}
