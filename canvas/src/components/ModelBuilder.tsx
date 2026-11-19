import { useWorkbenchServices } from '../workbench/services';
import { NativeLayout, retainedLayoutChoice, useNativeCanvasPresentation } from '../ide/nativeCanvasPresentation';
import { autoLayoutDirection, isLayoutChoice, type LayoutChoice } from '../ide/layoutDirection';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { useValue } from 'tldraw';
import { planCanvasFocus, applyCanvasFocus, queuedEditorFocus } from '../canvas/focusSelection';
import { showPassThroughCalls } from '../services/passThroughPresentation';
import { captureRoot, expandedDetailNodes, graphDetailLevel, type GraphDetailLevel } from '../services/graphDetailLevels';
import { graphNodeOrigin } from '../services/graphOverview';
import { revealOperationGroup } from '../services/operationGroups';
import React, { useState, useCallback, useRef, useEffect, useLayoutEffect, useMemo } from 'react';
import { warnIfUnsaved } from '../utils/unloadGuard';
import * as hostServices from '../ide/hostServices';
import { applyProjection as applyIdeProjection, currentProjection, ideMode, localBrowserMode, reportNodeClick, showingProjectGraph } from '../ide/projection';
import { isCodeFocusOn } from '../ide/codeFocusOn';
import { IdeStatus, ideStatus } from '../ide/status';
import { downloadGraphFile, prepareCurrentGraphFile } from '../ide/graphFileExport';
import { graphFileName } from '../services/graphFile';
import type { IdeProjectionDiagnostic, ProjectionPayload, LocalBrowserTargetIdentity } from '../ide/protocol.generated';

import {
  GraphLayout,
  ModelNode,
  ModelEdge,
  LayerType,
  SavedGraph,
  ModelGraph,
  PendingGraphCommit,
} from '../types';
import { getLayerByType } from '../data/layerTypes';
import {
  GraphDraftStore,
  localDevelopmentDraftStore,
  UNSAVED_DRAFT_ID,
} from '../services/localDraft';
import {
  GraphSaveRequest,
  GraphSaveState,
  RevisionedGraphSaveQueue,
  graphSaveDocument,
  graphSaveFingerprint,
} from '../services/revisionedGraphSaveQueue';

import SaveGraphDialog from './SaveGraphDialog';
import CodeExportDialog from './CodeExportDialog';
import CodeImportDialog from './CodeImportDialog';
import ModernTemplateGallery from './ModernTemplateGallery';
import { hierarchicalToFlat } from '../services/hierarchicalToFlat';
import { buildInferablePayload } from '../services/shapeInferencePayload';
import { verificationMetadata } from '../services/verificationRecord';
import { explainTorchError } from '../utils/explainTorchError';
import { ApiService, SessionHistoryState } from '../services/apiService';
import './StatusIndicators.css';
import { reconcileModuleView, graphNeedsLayout, ReconcileOptions } from '../services/canvasLayout';
import { autoModularize } from '../services/autoModularize';
import { TldrawCanvas } from '../canvas/TldrawCanvas';
import GraphSvgLayer from '../canvas/GraphSvgLayer';
import { useTldrawBridge } from '../canvas/useTldrawBridge';
import { useModuleFocus } from '../canvas/useModuleFocus';
import { ModuleFocusSurface } from '../canvas/ModuleFocusSurface';
import { focusScopeForTargets } from '../services/moduleFocus';
import { exportCanvasImage, saveCanvasImage, type ExportOutcome } from '../canvas/exportImage';
import { graphToSvg, graphSvgBounds } from '../services/graphSvg';
import { getCanvasTheme } from '../styles/canvasTheme';
import { zoomToReadable } from '../canvas/readableZoom';
import { parseInputShape as parseSharedInputShape } from '../utils/inputShape';
import { cameraDuration } from '../canvas/reducedMotion';
import { selectedEdgeId } from '../canvas/edgeSelection';
import { sliceDefaults } from '../canvas/sliceDefaults';
import { descendantsOf } from '../canvas/connectivity';
import {
  ConnectableNode,
  anchorReach,
  connectionRejectionMessage,
  kindFromNodeType,
  validateConnection,
} from '../canvas/connectivity';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { NodeExplanationContext } from '../canvas/NodeExplanationContext';
import { nodeIdToShapeId } from '../canvas/canvasShapeId';
import { visibleExplanationNodeIds } from '../canvas/visibleExplanationScope';
import { ValidationSweepOverlay } from '../canvas/ValidationSweepOverlay';
import { deriveInputShapes, healEdgesAfterDeletion } from '../services/graphTopology';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import { ungroupModuleGraph } from '../services/ungroupModule';



// Auto-induce module structure on load only for graphs big enough that a flat
// view is unreadable; small hand-built graphs are left as-is.
const AUTO_MODULARIZE_MIN_NODES = 30;

/**
 * A canvas node as the connect rules see it, so Shift+C enforces exactly what
 * the drag gesture does. An EXPANDED module frame deliberately resolves to
 * nothing: it is a container around children that carry their own wiring.
 */
const toConnectable = (node: ModelNode): ConnectableNode => {
  const kind = kindFromNodeType(node.type);
  const routable = kind === 'module' && !node.data?.expanded;
  return {
    id: node.id,
    kind,
    moduleEntryNodeId: routable ? node.data?.moduleEntryNodeId : undefined,
    moduleExitNodeId: routable ? node.data?.moduleExitNodeId : undefined,
  };
};

const newGraphCreateKey = (): string => {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  return `create-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
};

const needsCreateKey = (graph?: SavedGraph | null): boolean => (
  !graph || /^(graph-|template[-_]|template$|__unsaved__$)/.test(graph.id)
);

type CanvasViewMode = 'overview' | 'expanded' | 'focus';

type BuilderCommandEventDetail = {
  command: string;
  payload?: {
    shown?: boolean;
    level?: GraphDetailLevel;
    /** Expanded-overview (SVG) mode; omitted toggles the current value. */
    active?: boolean;
    /** `set-layout-direction`: Auto, Horizontal or Vertical. */
    choice?: unknown;
    layerType?: string;
    name?: string;
    nodeId?: string;
    /** Which picture `export-image` should make. See canvas/exportImage.ts. */
    format?: 'svg' | 'png';
    /** `export-image`: save a file instead of copying to the clipboard. */
    save?: boolean;
    nodeIds?: string[];
    edgeIds?: string[];
    /** Space around a focused change, in screen pixels. */
    inset?: number;
    target?: LocalBrowserTargetIdentity;
    finish?: (ok: boolean) => void;
    /** The host's `projection` message, forwarded by src/ide/bridge.ts. */
    projection?: ProjectionPayload;
    parameters?: Record<string, any>;
    options?: {
      parameterOverrides?: Record<string, any>;
      dataOverrides?: Record<string, any>;
    };
  };
};

interface ModelBuilderProps {
  initialGraph?: SavedGraph | null;
  /** Owner-captured crash store; never shared across authenticated users. */
  draftStore?: GraphDraftStore;
  onGraphChange?: () => void;
  onSaveComplete?: (savedGraph: SavedGraph) => void;
  /** Called once when a from-scratch graph is first persisted (currentGraphId
   *  transitions from unset to a real id), so the route can adopt /builder/:id. */
  onGraphCreated?: (id: string) => void;
  autoOpenImport?: boolean;
  onBuilderSnapshot?: (snapshot: ModelBuilderSnapshot) => void;
  /** Fired on a real click on a node — not on the pointer-down that begins a
   *  drag — so the embedding page can open its inspector. */
  onNodeClick?: () => void;
  enableExternalCommands?: boolean;
  hideStatusIndicators?: boolean;
  /** Hide the internal left sidebar. Used when embedding inside the new
   *  IDE shell, which renders the layer palette in its own activity
   *  panel. */
  hideSidebar?: boolean;
  /** Compact the canvas status indicators (icon-only, no labels) to
   *  reclaim vertical space when embedded. */
  compactStatus?: boolean;
}

export type SaveState = GraphSaveState;

const clearInferredNodeData = (node: ModelNode): ModelNode => ({
  ...node,
  data: {
    ...node.data,
    inputShape: undefined,
    outputShape: undefined,
    error: undefined,
  },
});

// Horizontal step between a node and the layer appended after it. 305 = the
// layerNode footprint dagre uses (215, canvasLayout.ts) + ranksep 90, so an
// appended chain lines up with what Auto layout would produce — and it keeps the
// gap above the forward-curve threshold in ModelEdgeShapeUtil, so auto-wired
// edges render as a forward S rather than the bowed back-edge route.
const NODE_STEP_X = 305;

const graphLayoutForSave = (
  graphNodes: ModelNode[],
  mode: CanvasViewMode,
  focusedModuleId: string | null,
  stored: GraphLayout = {},
): GraphLayout => {
  const {
    node_canvas: _nodeCanvas,
    node_data: _nodeData,
    edge_canvas: _edgeCanvas,
    ...documentLayout
  } = stored;
  return {
    ...documentLayout,
    strategy: 'manual',
    canvas_view_mode: mode,
    focused_module_id: focusedModuleId,
    expanded_module_ids: graphNodes.filter((node) => node.data?.expanded).map((node) => node.id),
    custom_module_ids: graphNodes
      .filter((node) => node.data?.moduleLayoutState === 'custom')
      .map((node) => node.id),
  };
};

const recoveredCommitRequest = (commit: PendingGraphCommit): GraphSaveRequest => {
  const description = commit.document.semantic.metadata?.description;
  return {
    graph: graphDocumentV2ToModelGraph(commit.document),
    name: commit.document.semantic.name,
    ...(typeof description === 'string' ? { description } : {}),
    author: commit.author,
    opSummary: commit.opSummary,
    createKey: commit.createKey,
    operations: commit.document.semantic.operations,
    layout: commit.document.view.layout,
  };
};

/** First vertical slot at `x` that no visible node already occupies, so an
 *  appended layer never lands on top of an existing card. */
const findFreeSlot = (nodes: ModelNode[], x: number, y: number): { x: number; y: number } => {
  const taken = (py: number) =>
    nodes.some((n) => !n.hidden
      && Math.abs(n.position.x - x) < 180
      && Math.abs(n.position.y - py) < 110);
  let slotY = y;
  for (let i = 0; i < 24 && taken(slotY); i += 1) slotY += 140;
  return { x, y: slotY };
};

export interface ModelBuilderSnapshot {
  backendConnected: boolean;
  isInferringShapes: boolean;
  shapeInferenceHasRun: boolean;
  verifying: boolean;
  exporting: boolean;
  nodeCount: number;
  edgeCount: number;
  errorCount: number;
  warningCount: number;
  /** The projection's own static resolution diagnostics; populated only in
   *  read-only local-browser mode, where they are the backend's static answer,
   *  not a runtime measurement. Runtime errors stay in nodeIssues. */
  staticDiagnostics: IdeProjectionDiagnostic[];
  /** Shape repairs the backend has suggested and nothing has applied yet. */
  repairCount: number;
  selectedNodeCount: number;
  graphName?: string;
  nodeIssues: Array<{
    nodeId: string;
    layerName: string;
    message: string;
  }>;
  selectedNode?: {
    id: string;
    layerName: string;
    layerType: string;
    nodeType?: string;
    inputShape?: number[];
    outputShape?: number[];
    error?: string;
    parameters: Record<string, any>;
    moduleTemplateId?: string;
    moduleLayerCount?: number;
    modulePreviewNodes?: Array<{ id: string; label?: string }>;
    modulePreviewEdges?: Array<{ source: string; target: string }>;
  } | null;
  historyState: SessionHistoryState;
  lastHistorySummary: string;
  canvasViewMode: CanvasViewMode;
  canvasToolId?: string;
  captureDetailLevel?: GraphDetailLevel;
  /** Live local Canvas only: the Layout choice and the direction it resolves to. */
  layoutChoice?: LayoutChoice;
  layoutDirection?: LayoutDirection;
  verifyResult: import('../services/apiService').VerifyModelResponse | null;
  /** Server persistence state, surfaced to the builder chrome save pill. */
  saveState: SaveState;
  /** What the header says in the IDE, where nothing is saved to a project store
   *  and the save states above describe a path that does not exist. Null in the
   *  web build, which keeps its pill. */
  ideStatus: IdeStatus | null;
}

/** Module ids that are a display FRAME: some other node names them as its parent.
 *
 * A collapsed wrapper around children that are ALSO in the graph is grouping, not
 * computation, and sending both duplicates the path. But a module node with no
 * children present IS the computation — a traced BERT arrives as twelve `moduleNode`
 * children carrying `nn.TransformerEncoderLayer`, and dropping those removed every
 * compute node in the model. */
function frameModuleIds(nodes: Array<any>): Set<string> {
  const parents = new Set<string>();
  for (const node of nodes) {
    const parent = node?.data?.parentId ?? node?.parentNode;
    if (parent) parents.add(String(parent));
  }
  return parents;
}

/** Edges whose endpoints all survived the node filter.
 *
 * Dropping a node without dropping its edges left the verify payload with 15 edges
 * over 4 nodes, and the backend answered "Cycle detected or graph is not connected"
 * — a build failure reported on a model that is wired correctly. */
function edgesWithin<T extends { source: string; target: string }>(edges: T[], ids: Set<string>): T[] {
  return edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target));
}

/** Commands the external page may execute without changing graph semantics,
 * source, persisted model state, or running validation. */
const LOCAL_BROWSER_VIEW_COMMANDS = new Set([
  'layout',
  'zoom-in',
  'zoom-out',
  'zoom-readable',
  'explain-model',
  'select-mode',
  'fit-view',
  'toggle-fullscreen',
  'view-overview',
  'view-expanded',
  'set-graph-detail',
  'toggle-expanded-svg',
  'show-unchanged-calls',
  'set-layout-direction',
  'view-focus',
  'focus-node',
  'focus-selection',
  'report-selection',
  'apply-projection',
  'validation-complete',
  'export-graph-file',
  'share-graph-file',
  'export-image',
]);

const ModelBuilder: React.FC<ModelBuilderProps> = ({
  initialGraph,
  draftStore = localDevelopmentDraftStore,
  onGraphChange,
  onSaveComplete,
  onGraphCreated,
  autoOpenImport = false,
  onBuilderSnapshot,
  onNodeClick,
  enableExternalCommands = false,
  compactStatus = false,
}) => {
  const services = useWorkbenchServices();
  const readOnlyLocalBrowser = localBrowserMode();
  const [nodes, setNodes] = useState<ModelNode[]>([]);
  const [edges, setEdges] = useState<ModelEdge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showCodeExportDialog, setShowCodeExportDialog] = useState(false);
  const [showCodeImportDialog, setShowCodeImportDialog] = useState(false);
  const [showTemplateGallery, setShowTemplateGallery] = useState(false);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [isInferringShapes, setIsInferringShapes] = useState(false);
  const [shapeInferenceHasRun, setShapeInferenceHasRun] = useState(false);
  const [canvasViewMode, setCanvasViewMode] = useState<CanvasViewMode>('overview');
  const [focusedModuleId, setFocusedModuleId] = useState<string | null>(null);
  const localContext = React.useContext(LocalBrowserContext);
  const explanationContext = React.useContext(NodeExplanationContext);
  const explanationContextRef = useRef(explanationContext);
  explanationContextRef.current = explanationContext;
  const localContextRef = useRef(localContext);
  const graphExportPending = useRef(false);
  localContextRef.current = localContext;
  const [projectionViewId, setProjectionViewId] = useState<string>();
  const [projectionSourceKey, setProjectionSourceKey] = useState<string>();
  const rememberProjectionView = (projection: ProjectionPayload) => {
    const page = localContextRef.current?.page;
    setProjectionViewId(JSON.stringify([page?.activeTarget?.workspaceFolder ?? page?.project?.workspaceFolder,
      projection.graphId ?? page?.project?.experimentId]));
    // Capture source IDs are run-scoped. Only the host's full source-set digest
    // proves that invocation context may migrate between capture namespaces.
    setProjectionSourceKey(projection.truncated ? undefined : projection.projectionDigest);
  };
  const [saveState, setSaveState] = useState<SaveState>('idle');
  const [graphIdentity, setGraphIdentity] = useState(() => ({
    name: initialGraph?.name || 'Untitled Model',
    description: initialGraph?.description,
  }));
  const graphIdentityRef = useRef(graphIdentity);
  const autosaveTimerRef = useRef<number | null>(null);
  const adoptGraphIdentity = useCallback((name: string, description?: string) => {
    const nextIdentity = { name, description };
    graphIdentityRef.current = nextIdentity;
    setGraphIdentity((current) =>
      current.name === name && current.description === description
        ? current
        : nextIdentity
    );
  }, []);
  const onGraphCreatedRef = useRef(onGraphCreated);
  const operationHistoryRef = useRef<Array<Record<string, unknown>>>(
    initialGraph?.document.semantic.operations || []
  );
  const storedLayoutRef = useRef<GraphLayout>(initialGraph?.document.view.layout || {});
  /**
   * The graph's semantic metadata, carried through every save.
   *
   * buildSaveRequest used to send `graph: { nodes, edges }` and nothing else, which
   * had two consequences. A verify result had nowhere durable to go, so the home
   * and projects lists — which read status out of the saved document — could only
   * ever say "Never run", however many times you validated. And because
   * graphDocumentAdapter spreads whatever metadata arrives, every user autosave
   * spread `{}` over the stored metadata, so the first save after loading a
   * template-instantiated graph quietly erased its `template_instances`. Seeding
   * this from the loaded document and sending it back fixes both.
   */
  const graphMetadataRef = useRef<Record<string, unknown>>(
    (initialGraph?.graph?.metadata as Record<string, unknown> | undefined) || {}
  );
  const draftAuthorityKeyRef = useRef<string | null>(initialGraph?.id ?? null);
  const suppressAutosaveFingerprintRef = useRef<string | null>(null);
  const createKeyRef = useRef<string | undefined>(
    needsCreateKey(initialGraph)
      ? initialGraph?.recoveredCommits?.find((commit) => commit.createKey)?.createKey
        ?? newGraphCreateKey()
      : undefined
  );
  const saveQueueRef = useRef<RevisionedGraphSaveQueue | null>(null);
  // buildSaveRequest is defined below this point, and the unload guard needs it
  // at event time rather than at mount time.
  const buildSaveRequestRef = useRef<((name: string, description: string | undefined, author: 'user' | 'agent') => GraphSaveRequest) | null>(null);
  onGraphCreatedRef.current = onGraphCreated;
  // Write the crash journal for a set of outstanding requests. Called both by the
  // queue (as requests enter and leave it) and, before the autosave debounce fires,
  // for the edit still waiting to be sent — Guide 11 promises a local draft written
  // BEFORE the server call, and until that pre-debounce write existed a reload
  // inside the 2s window silently reverted the edit.
  const writeCrashJournal = useCallback((
    graphId: string | undefined,
    revision: number,
    requests: GraphSaveRequest[],
  ) => {
    const draftKey = graphId ?? UNSAVED_DRAFT_ID;
    const persisted = draftStore.replaceJournal(
      draftKey,
      revision,
      requests.map((request) => ({
        document: graphSaveDocument(request, draftKey, revision),
        author: request.author,
        ...(request.opSummary ? { opSummary: request.opSummary } : {}),
        ...(request.createKey ? { createKey: request.createKey } : {}),
      }))
    );
    const previousKey = draftAuthorityKeyRef.current;
    if (persisted && previousKey && previousKey !== draftKey) {
      draftStore.clear(previousKey);
    }
    if (persisted) draftAuthorityKeyRef.current = draftKey;
  }, [draftStore]);
  const writeCrashJournalRef = useRef(writeCrashJournal);
  writeCrashJournalRef.current = writeCrashJournal;
  if (!saveQueueRef.current && services) {
    saveQueueRef.current = new RevisionedGraphSaveQueue(services.persistGraph, {
      onState: setSaveState,
      onCreated: (saved) => {

        onGraphCreatedRef.current?.(saved.id);
      },
      onSaved: (saved) => {

      },
      onPendingChanged: ({ graphId, revision, requests }) => {
        writeCrashJournalRef.current(graphId, revision, requests);
      },
    });
  }

  useEffect(() => {
    const queue = saveQueueRef.current;
    queue?.activateCallbacks();
    return () => queue?.retireCallbacks();
  }, []);

  // Warn before the tab closes on an uncommitted edit. A parameter change lands
  // on the server about 4s later (autosave debounce plus round trip), and closing
  // the tab inside that window destroyed the edit silently: no dialog, no
  // message, and nothing in the UI afterwards to say it had happened. Reload and
  // in-app navigation were already safe via the local draft queue, which is what
  // made this the worst version of the bug — the app had taught users that the
  // "Saved" chip could be trusted.
  //
  // A returnValue is all a browser will honour here; the text is the browser's,
  // not ours. Deliberately NOT a sendBeacon flush on pagehide: the commit is an
  // authenticated PATCH with a revision precondition, and beacons cannot carry
  // those headers or read the 409 that tells us the write lost a race — a flush
  // that silently dropped or misapplied the edit would be a worse bug than the
  // dialog. The prompt only appears when work is genuinely unsaved.
  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      // Same precondition the autosave effect uses: an empty canvas that has
      // never been saved is not unsaved work, it is nothing. Without this the
      // guard prompted on a freshly opened blank builder, because needsSave() is
      // trivially true when no snapshot has ever been accepted — and a dialog on
      // the way out of a canvas you never touched is how users learn to dismiss
      // the one that matters.
      if (nodesRef.current.length === 0 && !saveQueueRef.current?.authority().graphId) return;
      // Built here rather than captured, so the comparison is against the canvas
      // as it stands at the moment of the close, not as it was on mount.
      const identity = graphIdentityRef.current;
      const request = buildSaveRequestRef.current?.(identity.name, identity.description, 'user');
      warnIfUnsaved(event, saveQueueRef.current, request);
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  const [undoCount, setUndoCount] = useState(0);
  const [redoCount, setRedoCount] = useState(0);

  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<import('../services/apiService').VerifyModelResponse | null>(null);
  const [sweepRunId, setSweepRunId] = useState(0); // bumped after each verify to (re)play the validation sweep
  const [exporting, setExporting] = useState(false);
  // Expanded overview: render the whole graph as one SVG image instead of one
  // mounted React card per node. tldraw's culling is CSS-only, so a 700-node
  // fully-expanded graph keeps every shape mounted; this does not.
  const [svgOverview, setSvgOverview] = useState(false);
  // Layout direction, live local Canvas only. Auto follows the canvas shape;
  // `layoutDirection` is what the current nodes were laid out with, so ports,
  // skip lanes and SVG always match the geometry on screen.
  const [layoutChoice, setLayoutChoice] = useState<LayoutChoice>(() => retainedLayoutChoice() ?? 'auto');
  const layoutChoiceRef = useRef(layoutChoice);
  layoutChoiceRef.current = layoutChoice;
  const autoDirectionRef = useRef<LayoutDirection>(autoLayoutDirection(window.innerWidth, window.innerHeight) ?? 'horizontal');
  const [autoDirection, setAutoDirection] = useState(autoDirectionRef.current);
  const autoDirectionTimer = useRef<number | undefined>(undefined);
  const [layoutDirection, setLayoutDirection] = useState<LayoutDirection>();
  const layoutDirectionRef = useRef(layoutDirection);
  layoutDirectionRef.current = layoutDirection;
  // Captured reviews overlay the retained live geometry; source reviews own theirs.
  const liveDirectionRef = useRef<LayoutDirection | undefined>(undefined);
  const desiredDirection = readOnlyLocalBrowser ? (layoutChoice === 'auto' ? autoDirection : layoutChoice) : undefined;
  /** The direction a layout of `projection` should use. Review layouts are immutable. */
  const liveLayoutDirection = useCallback((projection = currentProjection()): LayoutDirection | undefined => {
    if (!readOnlyLocalBrowser) return undefined;
    const review = projection as import('../ide/graphReview').ReviewProjection | null;
    // A draft preview is laid out once in the orientation the live canvas uses.
    if (review?.reviewGraph && !review.proposal) return review.capturedContext ? liveDirectionRef.current : undefined;
    return layoutChoiceRef.current === 'auto' ? autoDirectionRef.current : layoutChoiceRef.current;
  }, [readOnlyLocalBrowser]);
  const commitAutoDirection = useCallback((next?: LayoutDirection) => {
    if (!next || next === autoDirectionRef.current) return;
    autoDirectionRef.current = next;
    setAutoDirection(next);
  }, []);
  /** Commit a new Auto direction only after the size has settled past the boundary. */
  const scheduleAutoDirection = useCallback((width: number, height: number) => {
    window.clearTimeout(autoDirectionTimer.current);
    if (autoLayoutDirection(width, height, autoDirectionRef.current) === autoDirectionRef.current) return;
    autoDirectionTimer.current = window.setTimeout(() => {
      const size = reactFlowWrapper.current?.getBoundingClientRect();
      if (size) commitAutoDirection(autoLayoutDirection(size.width, size.height, autoDirectionRef.current));
    }, 200);
  }, [commitAutoDirection]);
  useEffect(() => () => window.clearTimeout(autoDirectionTimer.current), []);
  // Resolve Auto from the mounted canvas before a pending projection is laid out.
  useLayoutEffect(() => {
    const size = reactFlowWrapper.current?.getBoundingClientRect();
    if (readOnlyLocalBrowser && size) commitAutoDirection(autoLayoutDirection(size.width, size.height));
  }, [readOnlyLocalBrowser, commitAutoDirection]);
  const [pendingShapeRepairs, setPendingShapeRepairs] = useState<Array<Record<string, any>>>([]);
  const initialFitGraphIdRef = useRef<string | null>(null);
  const fittedSweepRunIdRef = useRef(0);
  const [blocksFitRequest, setBlocksFitRequest] = useState(0);
  const fittedBlocksRequestRef = useRef(0);
  const lastDropRef = useRef<{ signature: string; at: number } | null>(null);
  // Live mirrors of nodes/edges so the stable expand/layout callbacks can read
  // the current graph without being re-created on every change.
  const nodesRef = useRef<ModelNode[]>([]);
  const edgesRef = useRef<ModelEdge[]>([]);
  // Read by addNode's placement/auto-wire so it can anchor on the live selection
  // without taking selectedNodeId as a dependency (which would re-create the
  // callback on every selection change).
  const selectedNodeIdRef = useRef<string | null>(null);
  const pendingCanvasFocus = useRef<(() => void) | null>(null);
  const pendingRenderedProjection = useRef<unknown>(null);
  useEffect(() => () => pendingCanvasFocus.current?.(), []);

  const handleSelectionChange = useCallback((nodeIds: string[]) => {
    // A selected node takes Delete back off any clicked edge. Without this,
    // edge-click -> node-click -> Delete would delete the edge, not the node.
    if (nodeIds.length > 0) selectedEdgeId.set(null);
    // Touch only the nodes whose selection actually flipped. Rewriting every
    // node's data object on each click invalidated the whole canvas cache, so a
    // single selection re-synced all shapes in a large graph.
    const selected = new Set(nodeIds);
    setNodes(nds => {
      let changed = false;
      const next = nds.map(n => {
        const isSelected = selected.has(n.id);
        if (Boolean(n.selected) === isSelected && Boolean(n.data?.isSelected) === isSelected) return n;
        changed = true;
        return { ...n, selected: isSelected, data: { ...n.data, isSelected } };
      });
      return changed ? next : nds;
    });

    const firstId = nodeIds.length > 0 ? nodeIds[0] : null;
    setSelectedNodeId(firstId);
    
  }, [setNodes]);
  
  const presentationGraphId = projectionViewId ?? initialGraph?.id;
  const moduleFocus = useModuleFocus({ nodes, edges, setNodes, graphId: presentationGraphId, enabled: !showingProjectGraph(),
    sourceKey: projectionSourceKey, svg: svgOverview, onRestoreSelection: handleSelectionChange });
  const moduleFocusRef = useRef(moduleFocus); moduleFocusRef.current = moduleFocus;
  const activateNode = useCallback((id: string) => {
    const node = nodesRef.current.find(node => node.id === id);
    if (!node) return; // Display-only boundary cards never reach source navigation.
    if (node?.type === 'moduleNode' && moduleFocusRef.current.navigate(id)) return;
    // With Code focus off, exploring never opens code; source links and Source still
    // do. Graph review and the copy-only page have no Code focus and keep their route.
    if (localContextRef.current?.codeFocus && !isCodeFocusOn()) return;
    reportNodeClick(id);
  }, []);
  const {
    editorRef,
    editorReady,
    handleEditorReady,
    handleSelectionChange: handleTldrawSelectionChange,
    handleShapeClick,
    handleShapeDoubleClick,
    handleCanvasClick,
  } = useTldrawBridge({ nodes: svgOverview ? [] : moduleFocus.graph.nodes, edges: svgOverview ? [] : moduleFocus.graph.edges,
    direction: layoutDirection, queryNodes: nodes, onNodesSynced: moduleFocus.markSceneCommitted,
    setNodes: moduleFocus.setDisplayNodes, setEdges, onSelectionChange: handleSelectionChange, onNodeClick, onNodeActivate: activateNode });
  const attachFocusEditor = moduleFocus.attachEditor;
  const handleFocusEditorReady = useCallback((editor: Parameters<typeof handleEditorReady>[0]) => {
    attachFocusEditor(editor);
    return handleEditorReady(editor);
  }, [handleEditorReady, attachFocusEditor]);

  const fitGraphKey = presentationGraphId || (nodes.length > 0 ? '__new_graph__' : null);
  const fitView = useCallback((duration = cameraDuration()) => {
    const editor = editorRef.current;
    const container = reactFlowWrapper.current;
    if (!editor || !container) return;
    const { width, height } = container.getBoundingClientRect();
    if (width <= 0 || height <= 0) return;
    // Both Chrome and the native sidebar can render before tldraw's throttled
    // viewport measurement catches up. Every fit uses the space available now.
    editor.updateViewportScreenBounds(container);
    if (!svgOverview) {
      if (!editor.getCurrentPageShapes().length) return;
      const bounds = editor.getCurrentPageBounds();
      const steps = editor.getCameraOptions().zoomSteps;
      const inset = Math.min(editor.options?.zoomToFitPadding ?? 128, width * .28);
      // Large primitive graphs can extend beyond the default 5% zoom floor.
      if (bounds) {
        const minimum = Math.min(steps[0], Math.max(1, width - inset) / Math.max(1, bounds.w),
          Math.max(1, height - inset) / Math.max(1, bounds.h));
        if (minimum < steps[0]) editor.setCameraOptions({ zoomSteps: [minimum, ...steps] });
      }
      editor.zoomToFit({ animation: { duration } });
    } else {
      const explanation = explanationContextRef.current;
      const boxes = graphSvgBounds(moduleFocusRef.current.graphRef.current.nodes, {
        explanationStates: explanation?.annotations ?? explanation?.states, annotationsVisible: explanation?.annotationsVisible,
      });
      if (!boxes.length) return;
      const x = Math.min(...boxes.map(box => box.x)), y = Math.min(...boxes.map(box => box.y));
      editor.zoomToBounds({ x, y, w: Math.max(...boxes.map(box => box.x + box.w)) - x,
        h: Math.max(...boxes.map(box => box.y + box.h)) - y }, { inset: 48, animation: { duration } });
    }
    initialFitGraphIdRef.current = fitGraphKey;
    fittedSweepRunIdRef.current = sweepRunId;
    fittedBlocksRequestRef.current = blocksFitRequest;
  }, [svgOverview, editorRef, fitGraphKey, sweepRunId, blocksFitRequest]);

  useEffect(() => {
    const editor = editorRef.current;
    const container = reactFlowWrapper.current;
    if (!editorReady || !editor || !container || typeof ResizeObserver === 'undefined') return;
    // Seed from the mounted size so the observer's first delivery does not
    // replace a restored camera. Pan/zoom never changes this element's size.
    let { width, height } = container.getBoundingClientRect();
    const surface = readOnlyLocalBrowser ? container.closest('.local-browser-canvas') : null;
    let surfaceSize = surface?.getBoundingClientRect();
    if (readOnlyLocalBrowser) scheduleAutoDirection(width, height);
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const next = container.getBoundingClientRect();
        const nextSurface = surface?.getBoundingClientRect();
        // Header/footer progress can take a row without resizing the host. That
        // is chrome reflow, not a request to reframe or relayout the user's graph.
        const chromeReflow = width === next.width && width > 0 && height > 0 && next.height > 0
          && surfaceSize && nextSurface && nextSurface.width > 0 && nextSurface.height > 0
          && surfaceSize.width === nextSurface.width && surfaceSize.height === nextSurface.height;
        surfaceSize = nextSurface;
        if (next.width === width && next.height === height) return;
        width = next.width; height = next.height;
        if (chromeReflow) return;
        if (readOnlyLocalBrowser) scheduleAutoDirection(width, height);
        if (width <= 0 || height <= 0 || !nodesRef.current.length) return;
        // Follow the drag without restarting a camera animation on every frame.
        if (moduleFocusRef.current.isReady() && !moduleFocusRef.current.consumesResize(next)) {
          if (moduleFocusRef.current.active) moduleFocusRef.current.refit();
          else fitView(0);
        }
      });
    });
    observer.observe(container);
    return () => { observer.disconnect(); cancelAnimationFrame(frame); };
  }, [editorReady, editorRef, fitView, readOnlyLocalBrowser, scheduleAutoDirection]);

  const [canvasToolId, setCanvasToolId] = useState<string>();
  const restoreLayout = useCallback((layout: NativeLayout) => {
    if (layout.choice) { layoutChoiceRef.current = layout.choice; setLayoutChoice(layout.choice); }
    liveDirectionRef.current = layout.direction;
    setLayoutDirection(layout.direction);
  }, []);
  useNativeCanvasPresentation(presentationGraphId, nodes as ModelNode[], setNodes,
    editorReady ? editorRef.current : null, initialFitGraphIdRef, edges as ModelEdge[], setEdges, moduleFocus,
    { choice: layoutChoice, direction: layoutDirection }, restoreLayout);

  useEffect(() => {
    const payload = pendingRenderedProjection.current;
    if (!payload || !editorReady || nodes.length === 0) return;
    let frame = 0;
    const acknowledge = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => { frame = requestAnimationFrame(() => {
        const editor = editorRef.current;
        const bounds = editor?.getContainer?.().getBoundingClientRect();
        if (payload === pendingRenderedProjection.current && payload === currentProjection()
          && document.visibilityState === 'visible' && bounds?.width && bounds.height && (svgOverview ? editor?.getContainer?.().querySelector('[data-testid="graph-svg-layer"]') : editor?.getCurrentPageShapes().length)) {
          window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: payload }));
          pendingRenderedProjection.current = null;
        }
      }); });
    };
    // A projection may arrive behind the measured tab. A resize when its canvas
    // becomes visible retries the render signal without polling or counting hidden UI.
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(acknowledge);
    const container = editorRef.current?.getContainer?.();
    if (container) observer?.observe(container);
    document.addEventListener('visibilitychange', acknowledge);
    acknowledge();
    return () => { cancelAnimationFrame(frame); observer?.disconnect(); document.removeEventListener('visibilitychange', acknowledge); };
  }, [nodes, edges, editorReady, editorRef, svgOverview]);

  useEffect(() => {
    if (!editorRef.current) return;
    const sync = () => {
      setUndoCount(editorRef.current?.getCanUndo() ? 1 : 0);
      setRedoCount(editorRef.current?.getCanRedo() ? 1 : 0);
    };
    sync(); // seed immediately once the editor exists (editorReady flips)
    const cleanup = editorRef.current.store.listen(sync);
    return cleanup;
  }, [editorRef, editorReady]);

  const historyState = useMemo<SessionHistoryState>(() => ({ undo_count: undoCount, redo_count: redoCount, event_count: 0 }), [undoCount, redoCount]);
  const lastHistorySummary = 'No changes yet';

  const applyLogicalGraphRef = useRef<
    ((nodes: ModelNode[], edges: ModelEdge[], opts?: ReconcileOptions) => ModelGraph) | null
  >(null);
  useEffect(() => {
    nodesRef.current = nodes as ModelNode[];
    edgesRef.current = edges as ModelEdge[];
  });

  selectedNodeIdRef.current = selectedNodeId;
  const selectedNode = selectedNodeId ? nodes.find(node => node.id === selectedNodeId) : null;
  const selectedModuleInstanceId = selectedNode?.data?.moduleInstanceId || null;
  const selectedNodes = useMemo(
    () => nodes.filter((node) => Boolean(node.selected) || Boolean(node.data?.isSelected)),
    [nodes]
  );
  const selectedIds = useMemo(() => new Set(selectedNodes.map((node) => node.id)), [selectedNodes]);
  const focusedEdge = useValue('MCP selected edge', () => selectedEdgeId.get(), []);
  const reportSelection = useCallback(() => {
    if (!readOnlyLocalBrowser) return;
    const selectedEdge = moduleFocusRef.current.graphRef.current.edges.find(edge => edge.id === selectedEdgeId.get());
    // A display-only source assignment card is never a host or agent target.
    const selected = nodesRef.current.filter(node => (node.selected || node.data?.isSelected) && !node.data?.sourceAssignment);
    window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: {
      nodeIds: selected.map(node => node.id),
      nodeLabels: Object.fromEntries(selected
        .map(node => [node.id, node.data.captureLabel || node.data.moduleInstanceId || node.data.layerType?.displayName || node.id])),
      edgeIds: selectedEdge?.data?.originalEdgeIds ?? (selectedEdge?.data?.synthetic ? [] : selectedEdge ? [selectedEdge.id] : []),
    } }));
  }, [readOnlyLocalBrowser]);
  useEffect(() => { reportSelection(); }, [selectedNodes, focusedEdge, reportSelection]);

  // Check backend health on mount — never on the read-only local page: the
  // broker owns session truth there, and an unauthenticated direct probe could
  // only manufacture a "Backend not connected" state that is not true.
  useEffect(() => {
    if (localBrowserMode()) return;
    const checkBackend = async () => {
      const isHealthy = await ApiService.checkBackendHealth();
      setBackendConnected(isHealthy);
    };
    checkBackend();
  }, []);

  // Auto-open the import dialog when the caller asks (e.g. from /builder?import=1)
  useEffect(() => {
    if (autoOpenImport) setShowCodeImportDialog(true);
  }, [autoOpenImport]);

  // Notify parent of changes
  useEffect(() => {
    if (onGraphChange && (nodes.length > 0 || edges.length > 0)) {
      onGraphChange();
    }
  }, [nodes, edges, onGraphChange]);

  // Retry a queued (offline) save when the browser regains connectivity.
  useEffect(() => {
    const onOnline = () => saveQueueRef.current?.retry();
    window.addEventListener('online', onOnline);
    return () => window.removeEventListener('online', onOnline);
  }, []);

  const buildSaveRequest = useCallback((
    name: string,
    description: string | undefined,
    author: 'user' | 'agent',
    opSummary?: string,
  ): GraphSaveRequest => ({
    graph: { nodes: nodes as ModelNode[], edges: edges as ModelEdge[], metadata: graphMetadataRef.current },
    name,
    description,
    author,
    opSummary,
    createKey: createKeyRef.current,
    operations: operationHistoryRef.current,
    layout: graphLayoutForSave(nodes, canvasViewMode, focusedModuleId, storedLayoutRef.current),
  }), [nodes, edges, canvasViewMode, focusedModuleId]);

  // Journal the pending edit well before the 2s autosave fires, and stop the pill
  // claiming "Saved" while it is not. Both used to wait for the debounced request:
  // reloading inside that window silently reverted the edit, and the pill read green
  // "Saved" for the full 2s after a delete. Kept on a short debounce of its own
  // rather than written synchronously, because serialising the document on every
  // node change janks dragging on a large graph.
  useEffect(() => {
    if (nodes.length === 0 && !saveQueueRef.current?.authority().graphId) return;
    const draftTimer = window.setTimeout(() => {
      const queue = saveQueueRef.current;
      if (!queue) return;
      // In the IDE there is no server save to journal against, and no pill should
      // claim one is in flight (task.md D1: the .py is the truth).
      if (ideMode()) return;
      const identity = graphIdentityRef.current;
      const request = buildSaveRequest(identity.name, identity.description, 'user');
      if (!queue.needsSave(request)) return;
      const { graphId, revision } = queue.authority();
      writeCrashJournalRef.current(graphId, revision, [request]);
      // Never overwrite a blocked state with "Saving…". Once the queue is
      // blocked, an edit made after it is appended to the frozen backlog and is
      // not sent at all, so a second tab's dropped edit went silent: the
      // conflict pill appeared for the first stale edit and the next edit
      // replaced it with a cyan "Saving…" that never resolved. Only the queue
      // clears these — by draining, or by being reconfigured on reload.
      setSaveState((current) => (
        current === 'conflict' || current === 'offline' ? current : 'saving'
      ));
    }, 300);
    return () => window.clearTimeout(draftTimer);
  }, [nodes, edges, canvasViewMode, focusedModuleId, buildSaveRequest]);

  buildSaveRequestRef.current = buildSaveRequest;

  // Debounced autosave to the server, persisting via graphStorage:
  // create-if-no-id (adopting the returned id) or update. Save failures keep the
  // draft and flip the pill to 'offline'.
  useEffect(() => {
    if (nodes.length === 0 && !saveQueueRef.current?.authority().graphId) return;
    const timer = window.setTimeout(() => {
      autosaveTimerRef.current = null;
      // Current Canvas presentation is checkpointed by its view transport.
      if (ideMode()) return;
      const identity = graphIdentityRef.current;
      const request = buildSaveRequest(
        identity.name,
        identity.description,
        'user',
      );
      const queue = saveQueueRef.current;
      const fingerprint = graphSaveFingerprint(request);
      if (suppressAutosaveFingerprintRef.current === fingerprint) {
        suppressAutosaveFingerprintRef.current = null;
        return;
      }
      suppressAutosaveFingerprintRef.current = null;
      if (!queue || !queue.needsSave(request)) return;
      queue.enqueue(request).catch(() => undefined);
    }, 2000);
    autosaveTimerRef.current = timer;
    return () => {
      window.clearTimeout(timer);
      if (autosaveTimerRef.current === timer) autosaveTimerRef.current = null;
    };
  }, [nodes, edges, canvasViewMode, focusedModuleId, buildSaveRequest]);

  useEffect(() => {
    const inputNode = nodes.find((n) => n.data?.layerType?.id === 'input');
    const inputShape = inputNode?.data?.parameters?.shape || '1,3,224,224';
    const parsedShape = String(inputShape)
      .split(',')
      .map((s: string) => parseInt(s.trim(), 10))
      .filter((v: number) => Number.isFinite(v));
    window.dispatchEvent(
      new CustomEvent('canvas-graph-updated', {
        detail: {
          graph: {
            nodes,
            edges,
            input_shape: parsedShape.length > 0 ? parsedShape : [1, 3, 224, 224],
          },
        },
      })
    );
  }, [nodes, edges]);

  // Track graph structure changes to avoid unnecessary shape inference


  // Delegates to the shared parser in utils/inputShape so there is ONE implementation;
  // kept as a useCallback because it appears in several dependency arrays below.
  const parseInputShape = useCallback(
    (rawShape: unknown): number[] => parseSharedInputShape(rawShape),
    [],
  );

  /** Reconstruct the flat logical graph from the current render state: drop
   *  display frames and synthetic edges, clear parentId/parent flags.
   *  Order-independent, so it works whether or not the canvas is grouped. */
  const flattenToLogical = useCallback((): { nodes: ModelNode[]; edges: ModelEdge[] } => {
    const frames = frameModuleIds(nodes as Array<any>);
    const flatNodes = (nodes as ModelNode[])
      .filter((n) => n.type !== 'moduleNode' || !frames.has(n.id))
      .map((n) => ({
        ...n,
        parentNode: undefined,
        extent: undefined,
        hidden: false,
        data: { ...n.data, parentId: undefined },
      })) as ModelNode[];
    const keptIds = new Set(flatNodes.map((n) => n.id));
    const flatEdges = edgesWithin(
      (edges as ModelEdge[]).filter((e) => !e.data?.synthetic),
      keptIds,
    ).map((e) => ({ ...e, hidden: false })) as ModelEdge[];
    return { nodes: flatNodes, edges: flatEdges };
  }, [nodes, edges]);

  // Shape inference uses real logical edges and layers, never display-only
  // wrappers or synthetic grouping edges that can obscure branch lineage.
  const buildShapeInferencePayload = useCallback(() => {
    const flat = flattenToLogical();
    return buildInferablePayload(flat.nodes, flat.edges);
  }, [flattenToLogical]);

  const collectShapeRepairs = useCallback(
    (response: import('../services/apiService').ShapeInferenceResponse): Array<Record<string, any>> => {
      const out: Array<Record<string, any>> = [];
      for (const err of response.errors || []) {
        for (const repair of err.suggested_repairs || []) {
          if (repair && typeof repair === 'object' && typeof repair.op === 'string') {
            out.push(repair as Record<string, any>);
          }
        }
      }
      return out;
    },
    []
  );
  
  // Cache presentation only. A new capture invalidates both layouts.
  const detailViews = useRef<{ captureId?: string; views: Partial<Record<GraphDetailLevel,
    Map<string, Pick<ModelNode, 'position'> & { expanded?: boolean; operationDetailsShown?: boolean }>>> }>({ views: {} });
  const setDetailLevel = useCallback((level: GraphDetailLevel) => {
    const current = nodesRef.current;
    const root = captureRoot(current);
    if (!root) return;
    const captureId = root.data.captureRef!.captureId;
    if (detailViews.current.captureId !== captureId) detailViews.current = { captureId, views: {} };
    const previousLevel = graphDetailLevel(current);
    const saved = level !== previousLevel ? detailViews.current.views[level] : undefined;
    if (root.data.captureDetailLevel) detailViews.current.views[previousLevel] = new Map(current.map(node =>
      [node.id, { position: { ...node.position }, expanded: node.data.expanded, operationDetailsShown: node.data.operationDetailsShown }]));
    let next = expandedDetailNodes(current, level);
    if (saved) next = next.map(node => {
      const view = saved.get(node.id);
      return view ? { ...node, position: view.position, data: { ...node.data,
        expanded: view.expanded, operationDetailsShown: view.operationDetailsShown } } : node;
    });
    const result = applyLogicalGraphRef.current?.(next, edgesRef.current,
      { relayout: !saved, preserveChildPositions: Boolean(saved) });
    setCanvasViewMode('expanded');
    setFocusedModuleId(null);
    if (moduleFocusRef.current.active) return;
    if (!result) return;
    const selected = selectedNodeIdRef.current;
    const before = selected ? graphNodeOrigin(current, selected) : undefined;
    let target = result.nodes.find(node => node.id === selected);
    const byId = new Map(result.nodes.map(node => [node.id, node]));
    const seen = new Set<string>();
    while (target?.hidden && !seen.has(target.id)) {
      seen.add(target.id);
      target = byId.get(target.data.operationGroupHiddenBy ?? target.data.parentId ?? '');
    }
    const after = target ? graphNodeOrigin(result.nodes, target.id) : undefined;
    if (target && !target.hidden) handleSelectionChange([target.id]);
    if (before && after && editorRef.current) {
      const camera = editorRef.current.getCamera();
      editorRef.current.setCamera({ ...camera, x: camera.x + before.x - after.x, y: camera.y + before.y - after.y },
        { animation: { duration: cameraDuration() } });
    } else if (editorRef.current) {
      // SVG mode deliberately has no tldraw shapes for zoomToFit to measure.
      const explanation = explanationContextRef.current;
      const boxes = graphSvgBounds(result.nodes, {
        explanationStates: explanation?.annotations ?? explanation?.states, annotationsVisible: explanation?.annotationsVisible,
      });
      if (boxes.length) {
        const x = Math.min(...boxes.map(box => box.x)), y = Math.min(...boxes.map(box => box.y));
        editorRef.current.zoomToBounds({ x, y, w: Math.max(...boxes.map(box => box.x + box.w)) - x,
          h: Math.max(...boxes.map(box => box.y + box.h)) - y }, { inset: 48, animation: { duration: cameraDuration() } });
      }
    }
  }, [editorRef, handleSelectionChange]);

  const setViewMode = useCallback((mode: CanvasViewMode) => {
    if (mode !== 'focus') moduleFocusRef.current.navigate(null, { immediate: true });
    if (mode === 'overview') setBlocksFitRequest(request => request + 1);
    if (mode === 'expanded' && captureRoot(nodesRef.current)) {
      setDetailLevel(graphDetailLevel(nodesRef.current));
      return;
    }
    setCanvasViewMode(mode);
    if (mode !== 'focus') {
      setFocusedModuleId(null);
    }
    if (mode !== 'overview' && mode !== 'expanded') return;
    const expanded = mode === 'expanded';
    const current = nodesRef.current;
    const modules = current.filter((n) => n.type === 'moduleNode');
    if (modules.length === 0) return;
    const flipped = current.map((n) =>
      n.type === 'moduleNode' && Boolean(n.data.expanded) !== expanded
        ? { ...n, data: { ...n.data, expanded } }
        : n
    );
    applyLogicalGraphRef.current?.(flipped, edgesRef.current, { relayout: true });
  }, [setDetailLevel]);

  // Focus is a separate display scope, never destructive expansion/relayout.
  const focusSelectedModule = useCallback(() => {
    const id = selectedNodeIdRef.current;
    if (id) moduleFocusRef.current.navigate(id);
  }, []);




  // ── Flat / template / user-built graph rendering ──────────────────────
  // These graphs (from templates, code import, or hand-grouping) live as a
  // single flat node list where module nodes carry `data.expanded` and their
  // children carry `data.parentId`. `applyLogicalGraph` is the one place that
  // turns that logical graph into a laid-out, collapse-aware render graph.

  // Toggle a flat module's expand state. Matches by node id OR moduleInstanceId
  // so the module header button (which passes moduleInstanceId) works too.
  const handleFlatToggleExpand = useCallback((moduleKey: string) => {
    const current = nodesRef.current;
    const target = current.find(n => n.id === moduleKey || n.data.moduleInstanceId === moduleKey);
    if (!target || target.data.expandable === false) return;
    const displayed = moduleFocusRef.current.graphRef.current.nodes.find(node => node.id === target.id);
    if (!displayed?.data.expanded && moduleFocusRef.current.navigate(target.id)) return;
    const flipped = current.map((n) =>
      n.id === moduleKey || n.data.moduleInstanceId === moduleKey
        ? { ...n, data: { ...n.data, expanded: !n.data.expanded } }
        : n
    );
    applyLogicalGraphRef.current?.(flipped, edgesRef.current, { relayout: true, anchorNodeId: target?.id, preserveChildPositions: true });
  }, []);

  const handleRevealOperations = useCallback((nodeId: string) => {
    applyLogicalGraphRef.current?.(revealOperationGroup(nodesRef.current, nodeId), edgesRef.current,
      { relayout: true, anchorNodeId: nodeId, preserveChildPositions: true });
  }, []);

  // Re-run the deterministic layout for the whole flat graph.
  const handleFlatResetLayout = useCallback(() => {
    if (moduleFocusRef.current.active) { moduleFocusRef.current.setOrientation(moduleFocusRef.current.orientation); return; }
    applyLogicalGraphRef.current?.(nodesRef.current, edgesRef.current, { relayout: true });
  }, []);

  /**
   * Ungroup: the inverse of `groupSelectionAsModule`, and the only way back out
   * of a G. Undo does not survive a reload, so before this a module made
   * yesterday was permanent.
   *
   * The transform itself is `ungroupModuleGraph`, which rewires the wrapper's
   * boundary edges onto the same entry/exit children `buildInferablePayload`
   * already rewires them to — so the flat graph this produces is the one the
   * backend has been validating for that module all along.
   */
  const ungroupModule = useCallback((moduleKey: string) => {
    const result = ungroupModuleGraph(nodesRef.current, edgesRef.current, moduleKey);
    if (!result) {
      window.dispatchEvent(new CustomEvent('builder-toast', {
        detail: { label: 'Nothing to ungroup', detail: 'Select a module block first.', status: 'warn' },
      }));
      return;
    }
    applyLogicalGraphRef.current?.(result.nodes, result.edges, { relayout: true });
    setSelectedNodeId(null);
    window.dispatchEvent(new CustomEvent('builder-toast', {
      detail: {
        label: `Ungrouped ${result.restored} layer${result.restored === 1 ? '' : 's'}`,
        detail: 'The block is gone; its layers are back on the canvas.',
        status: 'ok',
      },
    }));
  }, []);

  /**
   * Model-level undo history.
   *
   * Undo used to delegate to editorRef.current.undo() — tldraw's history — but every
   * model edit (delete, add from the palette, parameter change) mutates React state via
   * setNodes/setEdges, which tldraw never sees. So its stack was empty: the Undo button
   * stayed enabled, clicking it did nothing, and a select-all delete survived five undos
   * AND a reload as an empty model.
   *
   * Snapshots are taken on a short debounce so one user action is one entry rather than
   * every intermediate render, and the stack is capped so a long session cannot grow
   * without bound.
   */
  const MODEL_HISTORY_LIMIT = 50;
  type GraphSnapshot = { nodes: ModelNode[]; edges: ModelEdge[] };
  const historyRef = useRef<{
    past: GraphSnapshot[];
    future: GraphSnapshot[];
    applying: boolean;
    last: GraphSnapshot | null;
  }>({ past: [], future: [], applying: false, last: null });

  const snapshotKey = (snap: GraphSnapshot | null) =>
    snap ? `${snap.nodes.map((n) => n.id).join(',')}|${snap.edges.map((e) => e.id).join(',')}` : '';

  const applyLogicalGraph = useCallback(
    (logicalNodes: ModelNode[], logicalEdges: ModelEdge[], opts: ReconcileOptions = {}) => {
      const direction = liveLayoutDirection();
      const reviewing = Boolean((currentProjection() as { reviewGraph?: unknown } | null)?.reviewGraph);
      const reorient = !reviewing && direction !== liveDirectionRef.current
        && (liveDirectionRef.current !== undefined || !currentProjection()?.viewRestored);
      const { nodes: renderNodes, edges: renderEdges } = reconcileModuleView(
        logicalNodes,
        logicalEdges,
        { ...opts, relayout: opts.relayout || reorient, direction }
      );
      const withActions = renderNodes.map((node) =>
        node.type === 'moduleNode'
          ? {
              ...node,
              data: {
                ...node.data,
                onToggleExpand: handleFlatToggleExpand,
                onResetLayout: handleFlatResetLayout,
              },
            }
          : node.data.operationGroup ? { ...node, data: { ...node.data, onRevealOperations: handleRevealOperations } } : node
      );
      setNodes(withActions as ModelNode[]);
      setEdges(renderEdges as ModelEdge[]);
      setLayoutDirection(direction);
      if (reorient && liveDirectionRef.current !== undefined) setBlocksFitRequest(request => request + 1);
      if (!reviewing) liveDirectionRef.current = direction;
      return { nodes: withActions as ModelNode[], edges: renderEdges as ModelEdge[] };
    },
    [handleFlatToggleExpand, handleFlatResetLayout, handleRevealOperations, setNodes, setEdges, liveLayoutDirection]
  );

  useEffect(() => {
    applyLogicalGraphRef.current = applyLogicalGraph;


  }, [applyLogicalGraph]);

  // A new direction (choice, committed Auto resize or restored view) re-lays
  // out the whole graph once, then fits it after the new cards render.
  useEffect(() => {
    if (!desiredDirection || desiredDirection === layoutDirection || !nodesRef.current.length
      || (currentProjection() as { reviewGraph?: unknown } | null)?.reviewGraph) return;
    applyLogicalGraphRef.current?.(nodesRef.current, edgesRef.current, { relayout: true });
    setBlocksFitRequest(request => request + 1);
  }, [desiredDirection, layoutDirection, liveLayoutDirection]);

  // Apply a freshly-loaded logical graph, auto-inducing module structure first
  // when it's a large FLAT graph (e.g. the 267-node F5-TTS template) so it lands
  // as readable collapsed blocks instead of one unreadable row. No-op for
  // hierarchical templates (autoModularize bails when modules/parented children
  // already exist) and for small hand-built graphs (below the node threshold).
  // Assigned just below; the initialGraph effect above runs before this callback
  // is declared, so it reaches it through a ref rather than a TDZ reference.
  const applyLogicalAutoGroupedRef = useRef<
    ((n: ModelNode[], e: ModelEdge[], metadata?: Record<string, unknown>) => void) | null
  >(null);

  const applyLogicalAutoGrouped = useCallback(
    (
      logicalNodes: ModelNode[],
      logicalEdges: ModelEdge[],
      metadata?: Record<string, unknown>,
    ) => {
      const isFlat = !logicalNodes.some((n) => n.type === 'moduleNode' || n.data?.parentId);
      if (isFlat && logicalNodes.length >= AUTO_MODULARIZE_MIN_NODES) {
        // Pass the graph's own metadata so DECLARED blocks win over inferred repeats:
        // on a 107-node build carrying 12 declared full_transformer_block instances,
        // signature inference guessed 3 groups, none of them matching.
        const m = autoModularize(logicalNodes, logicalEdges, { metadata });
        if (m.groups.length > 0) {
          applyLogicalGraph(m.nodes, m.edges, { relayout: true });
          return;
        }
      }
      applyLogicalGraph(logicalNodes, logicalEdges, { relayout: graphNeedsLayout(logicalNodes) });
    },
    [applyLogicalGraph]
  );

  applyLogicalAutoGroupedRef.current = applyLogicalAutoGrouped;

  // Load the initial graph (template / saved canvas / import) through the
  // module-aware reconciler. Auto-runs dagre layout when positions are
  // degenerate (a freshly-flattened template), but preserves a canvas the
  // user arranged by hand.
  useEffect(() => {
    if (!initialGraph) {

      draftAuthorityKeyRef.current = null;
      suppressAutosaveFingerprintRef.current = null;
      adoptGraphIdentity('Untitled Model');
      operationHistoryRef.current = [];
      storedLayoutRef.current = {};
      saveQueueRef.current?.configure(undefined, 0, null);
      setCanvasViewMode('overview');
      setFocusedModuleId(null);
      setNodes([]);
      setEdges([]);
      setShapeInferenceHasRun(false);
      setVerifyResult(null);
      setPendingShapeRepairs([]);
      return;
    }
    draftAuthorityKeyRef.current = initialGraph.id;

    suppressAutosaveFingerprintRef.current = null;
    adoptGraphIdentity(initialGraph.name, initialGraph.description);
    const logicalNodes = initialGraph.graph.nodes as ModelNode[];
    const logicalEdges = initialGraph.graph.edges as ModelEdge[];
    const reconciledBaseline = reconcileModuleView(logicalNodes, logicalEdges, { relayout: false });
    const storedLayout = initialGraph.document.view.layout || {};
    storedLayoutRef.current = storedLayout;
    const storedMode = storedLayout.canvas_view_mode;
    const nextMode: CanvasViewMode = storedMode === 'expanded'
      ? storedMode
      : 'overview';
    const nextFocus = null;
    operationHistoryRef.current = initialGraph.document.semantic.operations || [];
    const baselineRequest: GraphSaveRequest = {
      graph: {
        nodes: reconciledBaseline.nodes as ModelNode[],
        edges: reconciledBaseline.edges as ModelEdge[],
      },
      name: initialGraph.name,
      description: initialGraph.description,
      author: 'user',
      operations: operationHistoryRef.current,
      layout: graphLayoutForSave(
        reconciledBaseline.nodes as ModelNode[],
        nextMode,
        nextFocus,
        storedLayout,
      ),
    };
    saveQueueRef.current?.configure(
      initialGraph.id,
      initialGraph.currentRevision,
      initialGraph.recoveredDraft ? null : graphSaveFingerprint(baselineRequest),
    );
    if (initialGraph.recoveredDraft) {
      const pending = initialGraph.recoveredCommits?.length
        ? initialGraph.recoveredCommits
        : [{ document: initialGraph.document, author: 'user' as const }];
      const requests = pending.map((commit) => ({
        ...recoveredCommitRequest(commit),
        createKey: commit.createKey ?? createKeyRef.current,
      }));
      suppressAutosaveFingerprintRef.current = graphSaveFingerprint(baselineRequest);
      requests.forEach((request) => {
        saveQueueRef.current?.enqueue(request).catch(() => undefined);
      });
    }
    setSaveState('idle');
    setCanvasViewMode(nextMode);
    setFocusedModuleId(nextFocus);
    const hasInferredData = logicalNodes.some((node) =>
      Boolean(node.data?.error)
      || Array.isArray(node.data?.inputShape)
      || Array.isArray(node.data?.outputShape)
    );
    setSelectedNodeId(initialGraph.document.view.selection[0] || null);
    setShapeInferenceHasRun(hasInferredData);
    setVerifyResult(null);
    setPendingShapeRepairs([]);
    // Re-derive module structure on REOPEN, not just on first instantiation from
    // the gallery. applyLogicalAutoGrouped had exactly one caller
    // (handleTemplateFromGallery), so a big flat template like the 267-node
    // F5-TTS landed as readable collapsed blocks the first time and then came
    // back as one unreadable 290-layer row the moment it was saved and reopened
    // — the grouping was induced in memory and never persisted.
    // Safe to re-run: autoModularize bails when modules or parented children
    // already exist, so a hierarchical or hand-grouped graph is untouched, and a
    // graph under the node threshold falls straight through.
    // The `currentRevision > 0` guard that used to sit here contradicted the paragraph
    // above: it skipped grouping for every SAVED graph, which is every graph after the
    // first autosave. Grouping does persist (a grouped F5-TTS round-trips as 294 nodes
    // with its 4 wrappers intact), so the guard was not protecting anything — but if a
    // graph was ever stored FLAT, because the autosave beat the in-memory grouping, it
    // could never recover and came back as a 290-layer row on every reopen.
    applyLogicalAutoGroupedRef.current?.(logicalNodes, logicalEdges, initialGraph?.graph?.metadata);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialGraph]);

  useEffect(() => {
    // Wait until the editor exists before scheduling a pending fit.
    if (!editorReady || !fitGraphKey || nodes.length === 0 || moduleFocus.active || moduleFocus.transition) return;
    const needsFit = () => initialFitGraphIdRef.current !== fitGraphKey
      || fittedSweepRunIdRef.current !== sweepRunId
      || fittedBlocksRequestRef.current !== blocksFitRequest;
    if (!needsFit()) return;
    // Wait for projection/layout effects to populate the editor. A newer layout
    // may cancel this timer; only an actual fit consumes the pending request.
    const timer = window.setTimeout(() => {
      if (needsFit()) fitView();
    }, 250);
    return () => window.clearTimeout(timer);
  }, [fitGraphKey, nodes, edges, fitView, editorReady, sweepRunId, blocksFitRequest, moduleFocus.active, moduleFocus.transition]);



  const addNode = useCallback((
    layerType: string,
    position?: { x: number; y: number },
    options?: {
      parameterOverrides?: Record<string, any>;
      dataOverrides?: Record<string, any>;
      idOverride?: string;
      /** Set by callers that build their own edges (code import,
       *  template instantiation) so addNode does not also splice one in. */
      skipAutoConnect?: boolean;
    }
  ) => {
    const layerDefinition = getLayerByType(layerType);
    if (moduleFocusRef.current.active) return;
    if (!layerDefinition) {
      console.warn(`[ModelBuilder] Unknown layer type: "${layerType}". Could not add node.`);
      return;
    }

    const id = options?.idOverride || `${layerType}-${Date.now()}`;
    
    // Initialize parameters with default values
    const parameters: Record<string, any> = {};
    layerDefinition.parameters.forEach(param => {
      parameters[param.name] = param.defaultValue;
    });
    if (options?.parameterOverrides) {
      Object.assign(parameters, options.parameterOverrides);
    }

    // Determine node type based on layer type
    let nodeType = 'layerNode';
    if (layerType === 'input') nodeType = 'inputNode';
    else if (layerType === 'output') nodeType = 'outputNode';
    else if (layerType === 'module') nodeType = 'moduleNode';

    // Placement: append to the right of the anchor (the single selection, else
    // the graph's last layer), in the first free vertical slot. Previously this
    // was Math.random() * 400 + 100 in PAGE coordinates, so a palette click
    // routinely dropped the layer outside the current viewport.
    const existing = nodesRef.current.filter((n) => !n.hidden);
    const anchor = existing.find((n) => n.id === selectedNodeIdRef.current)
      || existing.filter((n) => n.type !== 'outputNode').slice(-1)[0]
      || null;
    const placement = position
      || (anchor
        ? findFreeSlot(existing, anchor.position.x + NODE_STEP_X, anchor.position.y)
        : { x: 120, y: 160 });

    // A Slice's defaults depend on what flows into it, so they are settled here
    // rather than in the catalogue: the last axis, selected whole. See
    // sliceDefaults.ts for why that is the only default safe to apply unasked.
    // Explicit overrides (code import, templates) still win.
    if (layerType === 'narrow') {
      const supplied = options?.parameterOverrides || {};
      Object.entries(sliceDefaults(anchor?.data?.outputShape)).forEach(([key, value]) => {
        if (!(key in supplied)) parameters[key] = value;
      });
    }

    const newNode: ModelNode = {
      id,
      type: nodeType,
      position: placement,
      data: {
        layerType: layerDefinition,
        parameters,
        isSelected: false,
        // Set properties for module nodes
        ...(layerType === 'module' && {
          expandable: true,
          expanded: false,
          type: 'group',
        }),
        ...(options?.dataOverrides || {}),
      },
      // For module nodes, set styles for grouping
      ...(layerType === 'module' && {
        style: {
          width: 200,
          height: 150,
          backgroundColor: 'rgba(103, 217, 255, 0.08)',
        },
      }),
    };
    setShapeInferenceHasRun(false);
    setVerifyResult(null);
    setPendingShapeRepairs([]);
    setNodes((nds) => [...nds.map(clearInferredNodeData), newNode]);

    // Auto-wire. A hand-built graph used to be N isolated cards that the
    // warnings tab immediately flagged as "Isolated node", because nothing but
    // the reconciler, the initial load, delete-healing, code import ever called setEdges. Now an added layer joins the chain:
    //   - anchor -> new              (append)
    //   - anchor -> new -> oldTarget (splice, when the anchor already fed a node)
    // Input/Output nodes are never auto-wired from here, and no existing node is
    // repositioned — a hand-arranged layout must survive an add.
    // An Output added from the palette used to land isolated, because auto-wiring
    // was gated on layerNode: building Input -> Linear -> Output by hand left the
    // Output unconnected and the warnings tab flagged it. It is a terminal, so it
    // attaches to the chain's tail (the last non-terminal with nothing downstream)
    // and never splices — and it does not become the anchor, or the next added
    // layer would append after the Output.
    if (!options?.skipAutoConnect && nodeType === 'outputNode') {
      const candidates = existing.filter(
        (n) => n.type !== 'outputNode' && !edgesRef.current.some((e) => e.source === n.id),
      );
      const tail = (anchor && candidates.some((n) => n.id === anchor.id))
        ? anchor
        : candidates.slice(-1)[0] || null;
      if (tail) {
        setEdges((eds) => [...eds, { id: `edge-${tail.id}-${id}`, source: tail.id, target: id, data: {} }]);
      }
      return;
    }

    if (!options?.skipAutoConnect && anchor && nodeType === 'layerNode') {
      const downstream = edgesRef.current.find((e) => e.source === anchor.id);
      setEdges((eds) => {
        const kept = downstream ? eds.filter((e) => e.id !== downstream.id) : eds;
        const additions: ModelEdge[] = [{ id: `edge-${anchor.id}-${id}`, source: anchor.id, target: id, data: {} }];
        if (downstream) {
          additions.push({ id: `edge-${id}-${downstream.target}`, source: id, target: downstream.target, data: {} });
        }
        return [...kept, ...additions];
      });
      // Advance the anchor, so clicking the palette repeatedly builds FORWARD
      // instead of splicing every new layer into the same anchor.
      setSelectedNodeId(id);
    }
  }, [setNodes, setEdges]);

  /**
   * Add an Input node and wire it to the head of the chain.
   *
   * Export used to refuse a graph with no Input node — right after Validate had
   * called the same graph fine — and the refusal was a dead end: it named the
   * problem and left the user to work out that the fix was a palette layer the
   * tutorial never mentioned. The export dialog offers this instead.
   *
   * "Head" is any node with no incoming edge, which is what an Input feeds. If
   * there is nothing to feed, the node is still added: an Input node alone is a
   * legitimate starting point, and it is what the recipe begins with.
   */
  const addWiredInputNode = useCallback(() => {
    const currentNodes = nodesRef.current;
    const currentEdges = edgesRef.current;
    if (currentNodes.some((node) => node.data?.layerType?.id === 'input')) return;

    const hasUpstream = new Set(currentEdges.map((edge) => edge.target));
    const heads = currentNodes
      .filter((node) => node.type !== 'outputNode' && !hasUpstream.has(node.id))
      .map((node) => node.id);

    const id = `input-${Date.now()}`;
    addNode('input', undefined, { idOverride: id, skipAutoConnect: true });
    if (heads.length === 0) return;
    setEdges((eds) => [
      ...eds,
      ...heads.map((target) => ({
        id: `edge-${id}-${target}`,
        source: id,
        target,
        data: {},
      })),
    ]);
    setShapeInferenceHasRun(false);
  }, [addNode, setEdges]);

  const addNodeFromPalette = useCallback(
    (
      layerType: string,
      options?: {
        parameterOverrides?: Record<string, any>;
        dataOverrides?: Record<string, any>;
      }
    ) => {
      if (localBrowserMode()) return;
      addNode(layerType, undefined, options);
    },
    [addNode]
  );

  // Instantiate a template block (e.g. ViT) into its REAL layers on the canvas
  // via the canonical registry, instead of dropping a hollow module shell.
  // This is what makes template blocks validatable (shape inference + verify
  // run on real layers). Falls back to a plain module node on failure.
  const instantiateTemplateOnCanvas = useCallback(async (templateId: string) => {
    try {
      const res = await ApiService.instantiateCanonicalTemplate(templateId);
      const mapOp = (op: string) =>
        op.replace(/^nn\./, '').replace(/^torch\.nn\./, '').replace(/^torch\./, '')
          .replace(/^functional\./, '').replace(/^F\./, '').toLowerCase();
      const newNodes: ModelNode[] = res.nodes.map((n) => {
        const ltId = mapOp(n.op);
        const resolved = getLayerByType(ltId);
        const layerType = resolved || ({
          id: ltId, name: n.op, displayName: n.op, description: '',
          icon: 'Box', category: 'misc', parameters: [],
        } as LayerType);
        const type = layerType.id === 'input' ? 'inputNode' : layerType.id === 'output' ? 'outputNode' : 'layerNode';
        return {
          id: n.id,
          type,
          position: { x: 0, y: 0 },
          data: { layerType, parameters: { ...(n.params || {}) } },
        } as ModelNode;
      });
      const newEdges: ModelEdge[] = res.edges.map((e) => ({
        id: e.id, source: e.source, target: e.target, data: {},
      } as ModelEdge));
      applyLogicalGraph(
        [...nodesRef.current, ...newNodes],
        [...edgesRef.current, ...newEdges],
        { relayout: true }
      );
    } catch (err) {
      console.error('Template instantiation failed:', err);
      addNode('module', undefined, { dataOverrides: { moduleTemplateId: templateId } });
    }
  }, [applyLogicalGraph, addNode]);

  // removeNode(nodeId) used to be the Delete handler. deleteSelectedNodes replaced
  // it — it removes the whole selection and heals the edges around it, which
  // removeNode did not — and the last caller went with that change; only a stale
  // dependency array kept it referenced. Nothing calls it now.
  const deleteSelectedNodes = useCallback(() => {
    const selectedIds = nodesRef.current
      .filter((node) => node.selected || node.data?.isSelected)
      .map((node) => node.id);

    if (selectedIds.length === 0) return;
    if (localBrowserMode()) return;

    const selectedIdSet = new Set(selectedIds);
    setShapeInferenceHasRun(false);
    setVerifyResult(null);
    setPendingShapeRepairs([]);
    setEdges((eds) => healEdgesAfterDeletion(nodesRef.current, eds, selectedIdSet));
    setNodes((nds) => nds.filter((node) => !selectedIdSet.has(node.id)).map(clearInferredNodeData));
    setSelectedNodeId(null);
  }, [setNodes, setEdges]);

  /**
   * The editor -> app edge writer for drag-to-connect. Until now nothing in the
   * canvas could write edges at all; the gesture already checked direction,
   * self-loops, duplicates and cycles (canvas/connectivity.ts), so this only
   * appends and invalidates the cached inference the same way toggle-connection
   * does. Tensor shapes are deliberately NOT validated here — /api/validate_shapes
   * runs a real torch forward pass, far too slow for a drag.
   */
  const commitConnection = useCallback((source: string, target: string) => {
    // A drag-to-connect has no single-statement form in M5 — forward decides the
    // order — so the host refuses it and says so, and the canvas does not move.
    if (localBrowserMode() || moduleFocusRef.current.active) return;
    setEdges((eds) => (
      eds.some((edge) => edge.source === source && edge.target === target)
        ? eds
        : [...eds, { id: `edge-${source}-${target}`, source, target, data: {} }]
    ));
    setShapeInferenceHasRun(false);
    setVerifyResult(null);
  }, [setEdges]);

  /** Remove the edge the user clicked on the canvas. Deleting a connection is an
   *  explicit act, so unlike a node delete it does NOT heal the graph around it. */
  const removeSelectedEdge = useCallback((edgeId: string) => {
    setEdges((eds) => eds.filter((edge) => edge.id !== edgeId));
    selectedEdgeId.set(null);
    setShapeInferenceHasRun(false);
    setVerifyResult(null);
  }, [setEdges]);

  /**
   * Insert a new layer INTO an existing connection, targeted by edge id.
   *
   * addNode's auto-wire already splices, but it resolves the downstream edge with
   * `edges.find(e => e.source === anchor.id)` — the FIRST one only — so on a node
   * that already fans out it splices one branch and silently leaves the other
   * wired directly. Pointing at the edge names both endpoints, so there is
   * nothing to guess. addNode still creates the node (defaults, inference reset);
   * only the wiring is done here, explicitly.
   */
  const insertNodeOnEdge = useCallback((edgeId: string, layerTypeId: string) => {
    if (moduleFocusRef.current.active) return;
    const edge = edgesRef.current.find((e) => e.id === edgeId);
    if (!edge) return;
    const upstream = nodesRef.current.find((n) => n.id === edge.source);
    const downstream = nodesRef.current.find((n) => n.id === edge.target);
    const id = `${layerTypeId}-${Date.now()}`;
    // Make room instead of landing on top of the neighbours. The midpoint of two
    // ADJACENT cards is inside both of them — cards are 215 wide and one step
    // apart — so "drop it between them and let Auto layout tidy up" put the new
    // card over the ReLU it was inserted before, every time. Open the chain
    // instead: everything from the downstream node onward moves one step right,
    // and the new node takes the slot that just came free.
    const position = upstream
      ? { x: upstream.position.x + NODE_STEP_X, y: upstream.position.y }
      : downstream?.position;
    if (upstream && downstream) {
      const shifted = descendantsOf(edgesRef.current, edge.target);
      shifted.add(edge.target);
      setNodes((nds) => nds.map((node) => (
        shifted.has(node.id)
          ? { ...node, position: { ...node.position, x: node.position.x + NODE_STEP_X } }
          : node
      )));
    }
    // addNode's own anchor is the selection, which on this path is whatever the
    // user last clicked — not the node feeding this wire. So the Slice defaults
    // are computed from the real upstream here and passed in.
    if (localBrowserMode()) return;
    addNode(layerTypeId, position, {
      idOverride: id,
      skipAutoConnect: true,
      ...(layerTypeId === 'narrow'
        ? { parameterOverrides: sliceDefaults(upstream?.data?.outputShape) }
        : {}),
    });
    setEdges((eds) => [
      ...eds.filter((e) => e.id !== edgeId),
      { id: `edge-${edge.source}-${id}`, source: edge.source, target: id, data: {} },
      { id: `edge-${id}-${edge.target}`, source: id, target: edge.target, data: {} },
    ]);
    // The replaced edge is gone, so a stale selection would arm Delete on nothing.
    selectedEdgeId.set(null);
    setSelectedNodeId(id);
  }, [addNode, setEdges]);

  // What the gesture validates against: a hidden edge is real wiring behind a
  // collapsed module, but a synthetic frame edge is only a drawing.
  const connectEdges = useMemo(
    () => (edges as ModelEdge[]).filter((edge) => !edge.data?.synthetic),
    [edges],
  );

  const updateNodeParameters = useCallback((nodeId: string, parameters: Record<string, any>) => {
    if (localBrowserMode()) return;
    setShapeInferenceHasRun(false);
    setVerifyResult(null);
    setPendingShapeRepairs([]);
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? {
              ...clearInferredNodeData(node),
              data: {
                ...clearInferredNodeData(node).data,
                parameters,
              },
            }
          : clearInferredNodeData(node)
      )
    );
    
    // If this is an input node parameter change, clear the last hash to trigger re-inference
    const node = nodes.find(n => n.id === nodeId);
    if (node?.data.layerType?.id === 'input') {
      // Force re-inference on input parameter changes
    }
  }, [setNodes, nodes]);
  // Manual shape inference trigger
  const triggerShapeInference = useCallback(async () => {
    // A PROJECTION is not a model under construction, on either surface. The
    // external page cannot call editor APIs (it has no launch-token header) and
    // neither page may turn a source-only graph into a misleading field of
    // per-node connection errors — which is exactly what happened once the panel
    // began drawing projects: every class card carried "Node has no input
    // connections", a sentence about a forward pass that a graph of classes and
    // files does not have.
    if (localBrowserMode() || showingProjectGraph()) return;
    // …and again from the nodes themselves, because the flag above is global and
    // set by whichever projection arrived LAST. Opening any `.py` publishes a class
    // projection, which flips it false while a PROJECT is still on the canvas — and
    // then inference stamped every research card with a default `(1, 3, 224, 224)`.
    // A `token_accuracy` function does not have an input tensor, and a graph that
    // says it does is confidently wrong in the one way this canvas must never be.
    if (nodes.some((node) => node.data?.parameters?._kind !== undefined)) return;
    if (!backendConnected || nodes.length === 0 || isInferringShapes) return;

    // No blocking alert when an Input node is missing — fall back to a default
    // input shape so shape inference still runs (e.g. a lone template block).
    const inputNode = nodes.find(n => n.data.layerType?.id === 'input');

    setIsInferringShapes(true);
    setShapeInferenceHasRun(true);

    try {
      const inputShape = inputNode
        ? parseInputShape(inputNode.data.parameters.shape)
        : [1, 3, 224, 224];
      const { inferableNodes, inferableEdges } = buildShapeInferencePayload();

      const request = {
        id: 'temp',
        name: 'temp',
        graph: { nodes: inferableNodes, edges: inferableEdges },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        tags: [],
        input_shape: inputShape,
      };

      const response = await ApiService.inferShapes(request);
      setPendingShapeRepairs(collectShapeRepairs(response));
      const shapeMap =
        response && typeof response.shapes === 'object' && response.shapes !== null
          ? (response.shapes as Record<string, number[]>)
          : {};
      
      // Update nodes with inferred shapes and errors. inputShape was MISSING
      // here: this path set outputShape only, so after "Refresh all shapes" every
      // node's input shape stayed blank — and with it the Shape Table's INPUT
      // column and anything keyed on inputShape, such as the Slice range control.
      // Validate populated it; Refresh did not. Same helper for both now.
      const inputShapeByNode = deriveInputShapes(edgesRef.current, shapeMap);
      setNodes((nds) =>
        nds.map((node) => {
          const nodeError = response.errors?.find(err => err.nodeId === node.id);
          return {
            ...node,
            data: {
              ...node.data,
              outputShape: shapeMap[node.id],
              inputShape: inputShapeByNode.get(node.id) ?? node.data.inputShape,
              error: nodeError ? `${explainTorchError(nodeError.message)}${nodeError.suggestion ? ` Suggestion: ${nodeError.suggestion}` : ''}` : undefined,
            },
          };
        })
      );
      
    } catch (error) {
      console.error('Shape inference failed:', error);
      setNodes((nds) =>
        nds.map((node) => ({
          ...node,
          data: {
            ...node.data,
            error: node.id !== nodes[0]?.id ? 'Unable to infer shape - check backend connection' : undefined,
          },
        }))
      );
    } finally {
      setIsInferringShapes(false);
    }
  }, [backendConnected, buildShapeInferencePayload, collectShapeRepairs, isInferringShapes, nodes, parseInputShape, setNodes]);

  /**
   * Shapes re-infer themselves once the edits stop.
   *
   * Nothing but the Refresh button ever ran inference, while every structural
   * edit and every parameter change cleared the inferred data — so the normal
   * state of a graph being built was "no shapes". That is also the state in which
   * the Slice range control degrades to three numbers typed against a remembered
   * shape, the shape table's columns go blank, and a freshly inserted node cannot
   * show what flows into it. Asking the user to press Refresh after every edit
   * was the bug, not the fix.
   *
   * The trigger is `shapeInferenceHasRun === false`, which every mutation already
   * sets, so this needs no new state. triggerShapeInference sets it true on entry,
   * which is what stops this from looping. It is held in a ref because the
   * callback's identity changes with `nodes`, and depending on it directly would
   * re-arm the timer on every keystroke.
   */
  const inferShapesRef = useRef(triggerShapeInference);
  useEffect(() => {
    inferShapesRef.current = triggerShapeInference;
  }, [triggerShapeInference]);
  useEffect(() => {
    if (localBrowserMode() || showingProjectGraph()) return;
    if (shapeInferenceHasRun || isInferringShapes) return;
    if (!backendConnected || nodes.length === 0) return;
    // Long enough that a drag of the slice range, or a burst of typing, settles
    // into one request instead of one per frame.
    const timer = window.setTimeout(() => { void inferShapesRef.current(); }, 700);
    return () => window.clearTimeout(timer);
  }, [shapeInferenceHasRun, isInferringShapes, backendConnected, nodes.length]);

  /**
   * Apply the repairs the backend suggested. Wired to the Errors tab's
   * "Fix N shape problem(s)" button.
   *
   * This was a finished implementation sitting behind a comment saying it was
   * "not yet wired to a control", with an eslint-disable holding the build green
   * over it. Meanwhile a beginner who put a Conv2d after a Flatten got a raw torch
   * message, no plain-language hint and no fix-it control — while the fix itself
   * was already computed and thrown away.
   */
  const applySuggestedShapeRepairs = useCallback(async () => {
    if (pendingShapeRepairs.length === 0) return;

    const setParamRepairs = pendingShapeRepairs.filter((r) => r.op === 'set_param');
    const removeConnectionRepairs = pendingShapeRepairs.filter((r) => r.op === 'remove_connection');

    if (setParamRepairs.length > 0) {
      setNodes((nds) =>
        nds.map((node) => {
          const nodeRepairs = setParamRepairs.filter((repair) => repair.node_id === node.id);
          if (nodeRepairs.length === 0) return node;
          const nextParams = { ...(node.data.parameters || {}) };
          for (const repair of nodeRepairs) {
            if (typeof repair.param === 'string') {
              nextParams[repair.param] = repair.value;
            }
          }
          return {
            ...node,
            data: {
              ...node.data,
              parameters: nextParams,
              error: undefined,
            },
          };
        })
      );
    }

    if (removeConnectionRepairs.length > 0) {
      setEdges((eds) =>
        eds.filter((edge) => {
          const shouldRemove = removeConnectionRepairs.some((repair) => {
            if (typeof repair.edge_id === 'string' && repair.edge_id.length > 0) {
              return edge.id === repair.edge_id;
            }
            const sourceMatch = repair.source_node_id ? edge.source === repair.source_node_id : true;
            const targetMatch = repair.target_node_id ? edge.target === repair.target_node_id : true;
            return sourceMatch && targetMatch;
          });
          return !shouldRemove;
        })
      );
    }

    // Re-run inference so users immediately see repaired shapes/errors.
    setPendingShapeRepairs([]);
    await triggerShapeInference();
  }, [pendingShapeRepairs, setNodes, setEdges, triggerShapeInference]);

  const buildVerificationPayload = useCallback(() => {
    const inputNode = nodes.find((n) => n.data.layerType?.id === 'input');
    const rawShape = inputNode?.data?.parameters?.shape || '1,3,224,224';
    const inputShape = parseInputShape(rawShape);
    // Same requirement as buildShapeInferencePayload: verification runs on the
    // FLAT graph. It is named `flat` but was mapping every node including module
    // wrappers, and every edge including the synthetic ones the reconciler adds
    // to route around collapsed modules — so a grouped canvas sent the backend a
    // graph with wrapper nodes and duplicated paths.
    const frames = frameModuleIds(nodes as Array<any>);
    const keptNodes = nodes
      .filter((n) => n.type !== 'moduleNode' || !frames.has(n.id))
      .map(n => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: { ...n.data, parentId: undefined },
      }));
    const keptIds = new Set(keptNodes.map((n) => n.id));
    const flat = {
      nodes: keptNodes,
      edges: edgesWithin(edges.filter((e) => !e.data?.synthetic), keptIds)
        .map(e => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle,
          targetHandle: e.targetHandle,
        })),
    };
    return { flat, inputShape };
  }, [nodes, edges, parseInputShape]);

  const runVerification = useCallback(async () => {
    if (verifying || nodes.length === 0 || !backendConnected) return;
    setVerifying(true);
    try {
      const { flat, inputShape } = buildVerificationPayload();
      const result = await ApiService.verifyModel(flat as any, inputShape, true);
      setVerifyResult(result);
      setShapeInferenceHasRun(true);
      // Reflect per-node errors on the canvas so they're not buried in
      // the result panel.
      const errorByNodeId = new Map<string, string>();
      for (const err of result.errors || []) {
        // Same treatment the inference path gives: plain language first, torch's
        // own words kept. Validate used to show the bare message while Refresh
        // shapes showed the suggestion, so the same defect explained itself or
        // not depending on which button you pressed.
        // The verify result carries no `suggestion` field — only the inference
        // response does — so this path gets the plain-language sentence and
        // nothing more. That is still the difference between a beginner reading
        // "Expected 3D … but got input of size: [1, 32]" and reading that a
        // Conv2d needs image-shaped data.
        if (err.node_id) errorByNodeId.set(err.node_id, explainTorchError(err.message));
      }
      // Shared with triggerShapeInference — see deriveInputShapes. These two
      // paths disagreed for a long time: only this one populated inputShape.
      const inputShapeByNode = deriveInputShapes(edgesRef.current, result.shapes as any);
      setNodes(nds => nds.map(n => ({
        ...n,
        data: {
          ...n.data,
          outputShape: Array.isArray(result.shapes?.[n.id]) ? result.shapes[n.id] : n.data.outputShape,
          inputShape: inputShapeByNode.get(n.id) ?? n.data.inputShape,
          error: errorByNodeId.get(n.id),
        },
      })));
      setSweepRunId(id => id + 1); // play the validation sweep over the freshly-painted nodes

      // Write the outcome into the saved document, which is what the home and
      // projects lists read. Without this the status badge could only ever say
      // "Never run": the builder recorded verification in browser localStorage
      // while every reader consulted document metadata that nothing wrote, and
      // the function bridging the two had zero callers.
      //
      // Stamped with the queue's live revision, not the revision the route loaded
      // at — the latter is stale the moment anything autosaves, and would make a
      // fresh result read as "verified, then edited".
      //
      // The save is explicit because a metadata-only change moves none of the
      // dependencies the autosave effect watches, so nothing would carry it. The
      // backend does not bump the revision for a metadata-only update, so this
      // cannot invalidate the record it is writing.
      const verifiedQueue = saveQueueRef.current;
      if (verifiedQueue) {
        const firstResultError = result.errors?.[0];
        graphMetadataRef.current = {
          ...graphMetadataRef.current,
          ...verificationMetadata({
            ok: Boolean(result.ok),
            revision: verifiedQueue.authority().revision,
            errorCount: result.errors?.length ?? 0,
            firstError: firstResultError?.message,
            firstErrorNodeId: firstResultError?.node_id,
            summary: result.summary
              ? {
                  layerCount: result.summary.layer_count,
                  params: result.summary.total_parameters,
                  sizeMb: result.summary.model_size_mb,
                }
              : undefined,
          }),
        };
        const verifiedIdentity = graphIdentityRef.current;
        const verifiedRequest = buildSaveRequest(
          verifiedIdentity.name,
          verifiedIdentity.description,
          'user',
        );
        if (verifiedQueue.needsSave(verifiedRequest)) {
          verifiedQueue.enqueue(verifiedRequest).catch(() => undefined);
        }
      }
    } catch (err) {
      console.error('Verify failed:', err);
      setVerifyResult({
        ok: false,
        stage: 'shape_inference',
        input_shape: [],
        output_shape: null,
        shapes: {},
        errors: [{ message: err instanceof Error ? err.message : 'Verify request failed' }],
        trace: null,
        summary: null,
      });
    } finally {
      setVerifying(false);
    }
  // `edges` was listed but never read here: the payload comes from
  // buildVerificationPayload, which already depends on it, and the fresh edge list
  // is read through edgesRef. Listing it rebuilt this callback — and re-ran the
  // effect that installs the keydown listener downstream — on every edge change.
  }, [verifying, nodes, backendConnected, buildVerificationPayload, buildSaveRequest, setNodes]);

  const runExportPackage = useCallback(async () => {
    if (exporting || nodes.length === 0 || !backendConnected) return;
    setExporting(true);
    try {
      const { flat, inputShape } = buildVerificationPayload();
      const modelName = (graphIdentityRef.current.name || 'GeneratedModel').replace(/[^A-Za-z0-9_]/g, '_');
      const blob = await ApiService.exportPackage(flat as any, inputShape, modelName);
      hostServices.saveFile(`${modelName}.zip`, blob);
    } catch (err) {
      console.error('Export package failed:', err);
      hostServices.alert(`Export failed: ${err instanceof Error ? err.message : 'unknown error'}`);
    } finally {
      setExporting(false);
    }
  }, [exporting, nodes, backendConnected, buildVerificationPayload]);

  useEffect(() => {
    if (!onBuilderSnapshot) return;
    const selected = selectedNodeId ? nodes.find((node) => node.id === selectedNodeId) || null : null;
    const nodeIssues = nodes
      .filter((node) => Boolean(node.data?.error))
      .map((node) => ({
        nodeId: node.id,
        layerName: node.data?.layerType?.displayName || node.data?.layerType?.name || node.id,
        message: String(node.data?.error),
      }));
    const snapshot: ModelBuilderSnapshot = {
      backendConnected,
      isInferringShapes,
      shapeInferenceHasRun,
      verifying,
      exporting,
      graphName: graphIdentityRef.current.name,
      nodeCount: nodes.length,
      edgeCount: edges.length,
      errorCount: nodeIssues.length,
      warningCount: 0,
      repairCount: pendingShapeRepairs.length,
      selectedNodeCount: nodes.filter((node) => Boolean(node.selected) || Boolean(node.data?.isSelected)).length,
      nodeIssues,
      selectedNode: selected
        ? {
            id: selected.id,
            layerName: selected.data?.layerType?.displayName || selected.data?.layerType?.name || selected.id,
            layerType: selected.data?.layerType?.id || selected.type || 'layer',
            nodeType: selected.type || 'layerNode',
            inputShape: selected.data?.inputShape,
            outputShape: selected.data?.outputShape,
            error: selected.data?.error,
            parameters: selected.data?.parameters || {},
            moduleTemplateId: selected.data?.moduleTemplateId,
            moduleLayerCount: selected.data?.moduleLayerCount,
            modulePreviewNodes: Array.isArray(selected.data?.modulePreviewNodes)
              ? selected.data.modulePreviewNodes.map((node: any) => ({
                  id: String(node?.id || ''),
                  label: node?.label ? String(node.label) : undefined,
                }))
              : [],
            modulePreviewEdges: Array.isArray(selected.data?.modulePreviewEdges)
              ? selected.data.modulePreviewEdges.map((edge: any) => ({
                  source: String(edge?.source || ''),
                  target: String(edge?.target || ''),
                }))
              : [],
          }
        : null,
      historyState,
      lastHistorySummary,
      canvasViewMode: moduleFocus.active ? 'focus' : canvasViewMode,
      captureDetailLevel: captureRoot(nodes as ModelNode[]) ? graphDetailLevel(nodes as ModelNode[]) : undefined,
      layoutChoice: readOnlyLocalBrowser ? layoutChoice : undefined,
      layoutDirection: desiredDirection,
      canvasToolId,
      verifyResult,
      saveState,
      // Recomputed per publish rather than stored: the strip, the warnings
      // drawer, and the cards must all show the same static answer, so it is
      // read off the live projection here exactly like ideStatus is.
      staticDiagnostics: readOnlyLocalBrowser ? currentProjection()?.diagnostics ?? [] : [],
      // Recomputed per publish rather than stored: it is a pure reading of the
      // projection, and a copy in state could disagree with the graph on screen.
      ideStatus: ideMode() ? ideStatus(currentProjection()) : null,
    };
    onBuilderSnapshot(snapshot);
  }, [
    onBuilderSnapshot,
    backendConnected,
    isInferringShapes,
    shapeInferenceHasRun,
    verifying,
    exporting,
    nodes,
    edges,
    selectedNodeId,
    historyState,
    lastHistorySummary,
    canvasViewMode,
    moduleFocus.active,
    verifyResult,
    canvasToolId,
    saveState,
    pendingShapeRepairs,
    readOnlyLocalBrowser,
    layoutChoice,
    desiredDirection,
  ]);


  /**
   * Group all currently-selected nodes into a single moduleNode. The
   * grouped nodes become children of the new module (parentId set);
   * edges entirely inside the selection are kept; edges crossing the
   * selection boundary are rewired so they enter/exit the module.
   *
   * Modules created this way are real ReactFlow parent nodes — they
   * collapse / expand by toggling `data.expanded`, and they export as
   * standalone nn.Module classes through the existing code generator.
   */
  const groupSelectionAsModule = useCallback(() => {
    if (moduleFocusRef.current.active) return;
    const selected = nodes.filter(n => n.selected);
    // A module from one node is just a wrapper — require at least 2.
    if (selected.length < 2) {
      window.dispatchEvent(new CustomEvent('builder-toast', {
        detail: { label: 'Select at least two layers', detail: 'Shift-click or drag a box, then group.', status: 'warn' },
      }));
      return;
    }
    // Don't allow grouping moduleNodes themselves (would nest modules,
    // which the current rendering doesn't fully support yet).
    if (selected.some(n => n.type === 'moduleNode')) {
      window.dispatchEvent(new CustomEvent('builder-toast', {
        detail: { label: 'Cannot nest modules yet', detail: 'Select only layer nodes to group.', status: 'warn' },
      }));
      return;
    }
    const name = hostServices.prompt('Module class name (Python identifier):', 'CustomModule');
    if (!name) return;
    const safeName = name.replace(/[^A-Za-z0-9_]/g, '_').replace(/^([0-9])/, '_$1') || 'CustomModule';

    const selectedIds = new Set(selected.map(n => n.id));
    const moduleNodeId = `module-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

    const moduleNode: ModelNode = {
      id: moduleNodeId,
      type: 'moduleNode',
      position: { x: 0, y: 0 },
      data: {
        layerType: {
          id: 'module',
          name: 'Module',
          displayName: safeName,
          description: 'User-defined nn.Module',
          icon: 'Package',
          category: 'module',
          parameters: [],
        },
        parameters: { name: safeName },
        isGroup: true,
        expanded: false, // render collapsed like template modules (double-click to expand)
        moduleInstanceId: moduleNodeId,
        moduleLayerCount: selected.length,
        moduleStatus: 'ok',
        moduleLayoutState: 'deterministic',
      },
    } as ModelNode;

    // Build the new logical graph: wrapper first (renders behind children),
    // each selected node tagged as its child. The reconciler then sizes the
    // wrapper, lays the children out inside it, and re-flows the top level.
    const logicalNodes: ModelNode[] = [
      moduleNode,
      ...nodes.map(n =>
        selectedIds.has(n.id)
          ? ({ ...n, data: { ...n.data, parentId: moduleNodeId, isSelected: false } } as ModelNode)
          : n
      ),
    ];

    // Re-route edges crossing the module boundary onto the wrapper so the
    // module stays connected when collapsed. Edges fully inside are kept
    // (the reconciler hides them while collapsed); fully-outside untouched.
    const logicalEdges: ModelEdge[] = edges.map(e => {
      const srcInside = selectedIds.has(e.source);
      const tgtInside = selectedIds.has(e.target);
      if (srcInside && tgtInside) return e;
      if (srcInside && !tgtInside) return { ...e, source: moduleNodeId };
      if (!srcInside && tgtInside) return { ...e, target: moduleNodeId };
      return e;
    });

    applyLogicalGraph(logicalNodes, logicalEdges, { relayout: true });
    setSelectedNodeId(moduleNodeId);
  }, [nodes, edges, applyLogicalGraph]);

  // Load a gallery template/popular-model straight onto the current canvas
  // (the builder-side equivalent of the Welcome screen's "From template").
  const handleTemplateFromGallery = useCallback((templateData: any) => {
    setShowTemplateGallery(false);
    if (templateData?.type !== 'popular_model' && templateData?.type !== 'template') return;
    try {
      const flat = hierarchicalToFlat(templateData.data.graph);
      applyLogicalAutoGrouped(flat.nodes as ModelNode[], flat.edges as ModelEdge[]);
    } catch (e) {
      console.error('Template load failed', e);
    }
  }, [applyLogicalAutoGrouped]);



  const pendingEditorFocus = useRef<string | null>(null);
  const focusNodeById = useCallback((nodeId: string) => {
    if (!nodeId) return;
    const current = nodesRef.current;
    let plan = planCanvasFocus(current, edgesRef.current, [nodeId], []);
    if (!plan) { pendingEditorFocus.current = nodeId; return; }
    pendingEditorFocus.current = null;
    // Exact external selection needs real selectable shapes. The SVG overview
    // remains available for navigation and can be re-enabled after the jump.
    setSvgOverview(false);
    const scope = focusScopeForTargets(plan.nodes, edgesRef.current, [nodeId]);
    if (scope) plan = planCanvasFocus(current, edgesRef.current, [nodeId], [], false)!;
    moduleFocusRef.current.navigate(scope, { immediate: true, nodes: plan.nodes });
    const navigationRevision = moduleFocusRef.current.navigationRevision();
    if (plan.nodes.some((node, index) => node !== current[index])) {
      applyLogicalGraphRef.current?.(plan.nodes, edgesRef.current, { relayout: !scope });
    }
    pendingCanvasFocus.current?.();
    const deadline = Date.now() + 2000;
    let frame = 0;
    pendingCanvasFocus.current = () => cancelAnimationFrame(frame);
    const attempt = () => {
      if (Date.now() > deadline || moduleFocusRef.current.navigationRevision() !== navigationRevision) { pendingCanvasFocus.current = null; return; }
      if (moduleFocusRef.current.isReady() && editorRef.current && applyCanvasFocus(editorRef.current, plan!)) {
        pendingCanvasFocus.current = null; setSelectedNodeId(nodeId); return;
      }
      frame = requestAnimationFrame(attempt);
    };
    frame = requestAnimationFrame(attempt);
  }, [editorRef]);
  useEffect(() => {
    const queued = queuedEditorFocus(pendingEditorFocus.current, nodesRef.current.map(node => node.id));
    pendingEditorFocus.current = queued.pending;
    if (queued.focus) focusNodeById(queued.focus);
  }, [nodes, focusNodeById]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const layerType = event.dataTransfer.getData('application/canvas-layer');
      const templateId = event.dataTransfer.getData('application/template-id');
      const templateName = event.dataTransfer.getData('application/template-name');

      if (typeof layerType === 'undefined' || !layerType || !editorRef.current) {
        return;
      }

      const dropSignature = `${layerType}:${templateId}:${event.clientX}:${event.clientY}`;
      const now = Date.now();
      if (
        lastDropRef.current?.signature === dropSignature &&
        now - lastDropRef.current.at < 350
      ) {
        return;
      }
      lastDropRef.current = { signature: dropSignature, at: now };

      const position = editorRef.current.screenToPage({
        x: event.clientX,
        y: event.clientY,
      });

      const templateOverrides =
        layerType === 'module' && templateId
          ? {
              parameterOverrides: { name: templateName || templateId },
              dataOverrides: { moduleTemplateId: templateId },
            }
          : undefined;
      if (localBrowserMode()) return;
      addNode(layerType, position, templateOverrides);
    },
    [editorRef, addNode]
  );

  // Record a snapshot after the model settles. Debounced so one user action is one
  // history entry, and skipped while an undo/redo is being applied.
  useEffect(() => {
    const h = historyRef.current;
    if (h.applying) {
      h.applying = false;
      h.last = { nodes: nodesRef.current, edges: edgesRef.current };
      return;
    }
    const timer = window.setTimeout(() => {
      const snap = { nodes: nodesRef.current, edges: edgesRef.current };
      if (snapshotKey(h.last) === snapshotKey(snap)) return;
      if (h.last) {
        h.past.push(h.last);
        if (h.past.length > MODEL_HISTORY_LIMIT) h.past.shift();
        h.future = [];
      }
      h.last = snap;
      setUndoCount(h.past.length);
      setRedoCount(h.future.length);
    }, 350);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges]);

  const undoModelEdit = useCallback((): boolean => {
    const h = historyRef.current;
    if (!h.past.length) return false;
    const previous = h.past.pop() as GraphSnapshot;
    if (h.last) h.future.push(h.last);
    h.applying = true;
    h.last = previous;
    applyLogicalGraph(previous.nodes, previous.edges, { relayout: false });
    setUndoCount(h.past.length);
    setRedoCount(h.future.length);
    return true;
  }, [applyLogicalGraph]);

  const redoModelEdit = useCallback((): boolean => {
    const h = historyRef.current;
    if (!h.future.length) return false;
    const next = h.future.pop() as GraphSnapshot;
    if (h.last) h.past.push(h.last);
    h.applying = true;
    h.last = next;
    applyLogicalGraph(next.nodes, next.edges, { relayout: false });
    setUndoCount(h.past.length);
    setRedoCount(h.future.length);
    return true;
  }, [applyLogicalGraph]);

  // Handle keyboard shortcuts
  const onKeyDown = useCallback((event: KeyboardEvent) => {
    // Check if user is currently typing in an input field
    const activeElement = document.activeElement;
    const isTyping = activeElement && (
      activeElement.tagName === 'INPUT' || 
      activeElement.tagName === 'TEXTAREA' || 
      activeElement.tagName === 'SELECT' ||
      activeElement.hasAttribute('contenteditable')
    );

    // BuilderPageV2 hides and guards its own write shortcuts, but this older
    // document-level handler also owns delete, undo/redo, save and grouping.
    // Refuse them here so a read-only page cannot mutate its local projection.
    const key = event.key.toLowerCase();
    const mutatesReadOnlyProjection = (readOnlyLocalBrowser || moduleFocusRef.current.active) && !isTyping && (
      event.key === 'Delete' ||
      event.key === 'Backspace' ||
      ((event.ctrlKey || event.metaKey) && ['s', 'y', 'z'].includes(key)) ||
      (!event.ctrlKey && !event.metaKey && !event.altKey && key === 'g')
    );
    if (mutatesReadOnlyProjection) {
      event.preventDefault();
      return;
    }

    // A clicked edge owns Delete. The edge-click channel selects an edge without
    // selecting any node, so falling through would delete nodes instead of the
    // connection the user just clicked. Deliberately ABOVE the node handler:
    // deleteRemovesWholeSelection.test.ts slices a fixed 700-char window from the
    // key check below and asserts deleteSelectedNodes() sits inside it.
    const clickedEdgeId = selectedEdgeId.get();
    if ((event.key === 'Delete' || event.key === 'Backspace') && !isTyping && clickedEdgeId) {
      removeSelectedEdge(clickedEdgeId);
      return;
    }

    // Don't handle delete/backspace if user is typing in an input field
    if (event.key === 'Delete' || event.key === 'Backspace') {
      // Delete the WHOLE selection. This used to call removeNode(selectedNodeId) — one
      // id — so shift-selecting two cards and pressing Delete removed exactly one of
      // them (5 layers -> 4, not 3), and Cmd+A then Delete removed a single node. The
      // toolbar's trash button already routed through deleteSelectedNodes, which is why
      // the two disagreed. That helper also heals the edges around what it removes,
      // which removeNode does not.
      if (!isTyping) {
        deleteSelectedNodes();
      }
    }
    
    if (event.key === 'Escape') {
      // Close parameter editor if open, otherwise deselect node
      setSelectedNodeId(null);
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          data: { ...n.data, isSelected: false }
        }))
      );
    }
    
    // Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z: the model stack owns these, because tldraw's
    // history does not contain model edits at all.
    //
    // Cmd/Ctrl+Y redoes too. Every shortcut table in the docs listed Ctrl+Y as
    // THE redo key and none of them mentioned Shift+Z, so a user who undid one
    // step too far pressed the documented key, got nothing, and concluded the
    // work was gone. Binding the alias is the smaller half of that fix — the
    // tables are corrected as well — and it is what Windows users expect anyway.
    if (!isTyping && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      if (event.shiftKey) {
        if (!redoModelEdit() && editorRef.current) editorRef.current.redo();
      } else if (!undoModelEdit() && editorRef.current) {
        editorRef.current.undo();
      }
      return;
    }

    if (!isTyping && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
      event.preventDefault();
      if (!redoModelEdit() && editorRef.current) editorRef.current.redo();
      return;
    }

    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
      event.preventDefault();
      setShowSaveDialog(true);
    }

    if (!isTyping && event.shiftKey && event.key.toLowerCase() === 'f') {
      event.preventDefault();
      focusSelectedModule();
    }

    if (!isTyping && !event.metaKey && !event.ctrlKey && event.key.toLowerCase() === 'g') {
      event.preventDefault();
      groupSelectionAsModule();
    }
    // `selectedNodeId` and `removeNode` are gone from this list because the handler
    // stopped using them: deleting now goes through deleteSelectedNodes, which
    // removes the whole selection and its edges. They were left behind by that
    // change, keeping the handler — and the effect that installs it on document —
    // churning on every selection change.
  }, [
    deleteSelectedNodes,
    removeSelectedEdge,
    undoModelEdit,
    redoModelEdit,
    // Read by the undo/redo fallback above. Every other hook in this file already
    // lists it; this one did not, which is the classic setup for a handler that
    // closes over a stale editor.
    editorRef,
    setNodes,
    focusSelectedModule,
    groupSelectionAsModule,
    readOnlyLocalBrowser,
  ]);

  useEffect(() => {
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onKeyDown]);

  const handleSave = useCallback(async (name: string, description?: string) => {
    // A manual save supersedes the outstanding debounce. Adopt its identity
    // before building the request so later autosaves cannot revert the name or
    // append an identical stale-name revision.
    if (autosaveTimerRef.current !== null) {
      window.clearTimeout(autosaveTimerRef.current);
      autosaveTimerRef.current = null;
    }
    adoptGraphIdentity(name, description);
    const request = buildSaveRequest(name, description, 'user');
    const queue = saveQueueRef.current;
    if (!queue || !queue.needsSave(request)) return;
    try {
      const savedGraph = await queue.enqueue(request);
      if (savedGraph && onSaveComplete) {
        onSaveComplete(savedGraph);
      }
    } catch {
      // The queue owns the truthful offline/conflict state and keeps the draft.
    }
  }, [adoptGraphIdentity, buildSaveRequest, onSaveComplete]);

  const saveDialogGraph = useMemo<SavedGraph | null | undefined>(() => (
    initialGraph ? {
      ...initialGraph,
      name: graphIdentity.name,
      description: graphIdentity.description,
    } : initialGraph
  ), [initialGraph, graphIdentity.name, graphIdentity.description]);

  const generateSuggestedName = useCallback(() => {
    const layerCounts = nodes.reduce((acc, node) => {
      const layerType = node.data.layerType as LayerType;
      if (layerType?.category) {
        acc[layerType.category] = (acc[layerType.category] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    const categories = Object.keys(layerCounts);
    if (categories.length === 0) return 'New Model';
    
    if (categories.includes('conv') && categories.includes('linear')) {
      return 'CNN Classifier';
    } else if (categories.includes('conv')) {
      return 'Convolutional Network';
    } else if (categories.includes('linear')) {
      return 'Neural Network';
    } else {
      return 'Custom Model';
    }
  }, [nodes]);

  const autoLayoutCanvas = useCallback(() => {
    if (moduleFocusRef.current.active) { moduleFocusRef.current.setOrientation(moduleFocusRef.current.orientation); return; }
    if (!editorRef.current || nodes.length === 0) return;
    // Size-aware dagre layout of the top level and expanded contents, in the
    // current direction. Fit once the new cards have rendered.
    applyLogicalGraph(nodes as ModelNode[], edges as ModelEdge[], { relayout: true });
    setBlocksFitRequest(request => request + 1);
  }, [editorRef, nodes, edges, applyLogicalGraph]);

  const handleImport = useCallback((graph: ModelGraph, className: string) => {
    // Clear existing graph
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    setShapeInferenceHasRun(false);
    

    // Auto-induce module structure, exactly as the template and reopen paths do.
    // Imported models are the LARGEST flat graphs the app ever sees — a real
    // PyTorch file routinely parses to hundreds of layers — so this is the path
    // that needed grouping most, and it was the only whole-graph load that
    // bypassed it entirely (it set nodes/edges directly, without even going
    // through applyLogicalGraph).
    // The rule this follows: auto-group whenever a whole graph arrives from an
    // EXTERNAL source (template, reopen, import); never on an incremental edit
    // (adding a layer), where regrouping would fight the user.
    applyLogicalAutoGroupedRef.current?.(
      graph.nodes as ModelNode[],
      graph.edges as ModelEdge[],
      graph.metadata,
    );

    console.log(`Successfully imported ${className} model`);
  }, [setNodes, setEdges]);

  useEffect(() => {
    if (!enableExternalCommands) return;
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent<BuilderCommandEventDetail>;
      const command = customEvent.detail?.command;
      const payload = customEvent.detail?.payload;
      if (!command) return;
      if (readOnlyLocalBrowser && !LOCAL_BROWSER_VIEW_COMMANDS.has(command)) return;
      if (moduleFocusRef.current.active && !LOCAL_BROWSER_VIEW_COMMANDS.has(command)) {
        window.dispatchEvent(new CustomEvent('builder-toast', {
          detail: { label: 'Return to overview', detail: 'Open Overview before using this model command.', status: 'info' },
        }));
        return;
      }
      if (command === 'export-graph-file' || command === 'share-graph-file') {
        const sharing = command === 'share-graph-file';
        if (graphExportPending.current) return;
        // A plugin view publishes through its host, which reports each step beside the button.
        const publishing = sharing ? localContextRef.current?.publishing ?? null : null;
        if (publishing && !publishing.start()) return;
        graphExportPending.current = true;
        void (async () => {
          try {
            const file = await prepareCurrentGraphFile(() => {
              const context = localContextRef.current;
              if (!context) throw new Error('Open a local model before exporting its graph.');
              return { graph: { nodes: nodesRef.current, edges: edgesRef.current }, page: context.page, displayed: currentProjection(),
                direction: layoutDirectionRef.current, connected: context.status?.phase === 'connected' && context.status.hostLink === 'connected' };
            }, localContextRef.current?.portableExports);
            if (publishing) {
              // The same file Export graph saves; the plugin host signs in and uploads it.
              publishing.publish(JSON.stringify(file));
              return;
            }
            if (sharing) {
              // The same file Export graph saves; VS Code uploads it and reports back.
              const json = new Blob([JSON.stringify(file)], { type: 'application/json' });
              if (!hostServices.shareGraph(graphFileName(file.document.semantic.name), json)) throw new Error('Add to tensorViz works in the VS Code Canvas.');
              window.dispatchEvent(new CustomEvent('builder-toast', {
                detail: { label: 'Adding to tensorViz', detail: 'Confirm the browser prompt if you are not signed in. tensorViz reports the result here.', status: 'info' },
              }));
              return;
            }
            downloadGraphFile(file);
            window.dispatchEvent(new CustomEvent('builder-toast', {
              detail: { label: 'Graph exported', detail: 'Open the file in the graph viewer to explore or share it.', status: 'ok' },
            }));
          } catch (error) {
            const detail = error instanceof Error ? error.message : 'Try opening the model again.';
            if (publishing) publishing.fail(detail);
            else window.dispatchEvent(new CustomEvent('builder-toast', {
              detail: { label: sharing ? 'Could not add graph to tensorViz' : 'Could not export graph', detail, status: 'error' },
            }));
          } finally { graphExportPending.current = false; }
        })();
        return;
      }
      if (command === 'add-layer') {
        const layerType = payload?.layerType;
        if (!layerType) return;
        const templateId = (payload?.options as any)?.dataOverrides?.moduleTemplateId;
        if (layerType === 'module' && typeof templateId === 'string') {
          void instantiateTemplateOnCanvas(templateId);
          return;
        }
        addNodeFromPalette(layerType, payload?.options);
        return;
      }
      if (command === 'verify') {
        void runVerification();
        return;
      }
      if (command === 'rename-model') {
        const name = typeof payload?.name === 'string' ? payload.name.trim() : '';
        if (!name || name === graphIdentityRef.current.name) return;
        void handleSave(name, graphIdentityRef.current.description);
        return;
      }
      if (command === 'save-as-template') {
        const name = (payload as any)?.name;
        if (!name || !services) return;
        const tplNodes = nodesRef.current.map((n) => ({ id: n.id, type: n.type || 'layerNode', position: n.position, data: n.data }));
        const tplEdges = edgesRef.current.map((e) => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle || undefined, targetHandle: e.targetHandle || undefined, data: e.data }));
        services.saveUserTemplate(name, { nodes: tplNodes, edges: tplEdges } as any)
          .then(() => window.dispatchEvent(new CustomEvent('builder-toast', { detail: { label: 'Saved as template', detail: name, status: 'ok' } })))
          .catch((err) => console.error('save-as-template failed', err));
        return;
      }
      if (command === 'export-package') {
        void runExportPackage();
        return;
      }
      if (command === 'export-code') {
        setShowCodeExportDialog(true);
        return;
      }
      if (command === 'refresh-shapes') {
        void triggerShapeInference();
        return;
      }
      if (command === 'apply-repairs') {
        void applySuggestedShapeRepairs();
        return;
      }
      if (command === 'layout') {
        void autoLayoutCanvas();
        return;
      }
      if (command === 'zoom-in') {
        if (editorRef.current) editorRef.current.zoomIn(editorRef.current.getViewportScreenCenter(), { animation: { duration: cameraDuration() } });
        return;
      }
      if (command === 'zoom-out') {
        if (editorRef.current) editorRef.current.zoomOut(editorRef.current.getViewportScreenCenter(), { animation: { duration: cameraDuration() } });
        return;
      }
      if (command === 'zoom-readable') {
        const explanation = explanationContextRef.current;
        if (editorRef.current) zoomToReadable(editorRef.current, moduleFocusRef.current.graphRef.current.nodes, {
          explanationStates: explanation?.annotations ?? explanation?.states, annotationsVisible: explanation?.annotationsVisible,
        });
        return;
      }
      if (command === 'explain-model') {
        reportSelection();
        const editor = editorRef.current;
        if (!editor) return;
        const graph = moduleFocusRef.current.graphRef.current;
        const explanation = explanationContextRef.current;
        const boxes = new Map(graphSvgBounds(graph.nodes, {
          explanationStates: explanation?.annotations ?? explanation?.states, annotationsVisible: explanation?.annotationsVisible,
        }).map(box => [box.id, box]));
        const ids = visibleExplanationNodeIds(graph.nodes, editor.getViewportPageBounds(), id =>
          editor.getShapePageBounds(nodeIdToShapeId(id)) ?? boxes.get(id));
        explanation?.model?.start(ids);
        return;
      }
      if (command === 'select-mode') {
        if (editorRef.current) editorRef.current.setCurrentTool('select');
        return;
      }
      if (command === 'delete-selected') {
        deleteSelectedNodes();
        return;
      }
      if (command === 'toggle-connection') {
        // The missing build verbs. Until now NOTHING in the UI called setEdges —
        // only the reconciler, initial load, delete-healing, code import and the
        // AI path did — so a hand-built graph could never be connected.
        // Reads from refs and decides BEFORE calling setEdges: React StrictMode
        // double-invokes updaters in dev, so deciding connect-vs-disconnect
        // inside the updater would create an edge on pass 1 and delete it on
        // pass 2, making the whole feature look broken.
        const picked = nodesRef.current.filter((n) => !n.hidden && (n.selected || n.data?.isSelected));
        if (picked.length !== 2) {
          window.dispatchEvent(new CustomEvent('builder-toast', {
            detail: { label: 'Select two layers', detail: 'Pick exactly two layers to connect or disconnect.', status: 'warn' },
          }));
          return;
        }
        // Direction follows the canvas: the left-hand node feeds the right-hand
        // one, which is the same left-to-right reading order the layout uses.
        const [a, b] = picked.slice().sort((m, n) => m.position.x - n.position.x);
        const existingEdge = edgesRef.current.find(
          (e) => (e.source === a.id && e.target === b.id) || (e.source === b.id && e.target === a.id),
        );
        if (existingEdge) {
          setEdges((eds) => eds.filter((e) => e.id !== existingEdge.id));
          window.dispatchEvent(new CustomEvent('builder-toast', {
            detail: { label: 'Disconnected', detail: 'Removed the connection between the two layers.', status: 'ok' },
          }));
        } else {
          // The same structural rules the drag gesture enforces. This path had
          // none at all: it would wire a layer INTO an Input, or close a cycle
          // that the shape engine's topological walk cannot then order.
          const anchor = toConnectable(a);
          const logical = edgesRef.current.filter((e) => !e.data?.synthetic);
          const check = validateConnection(
            { node: anchor, side: 'source', reach: anchorReach(logical, anchor, 'source') },
            toConnectable(b),
            logical,
          );
          if (!check.ok) {
            window.dispatchEvent(new CustomEvent('builder-toast', {
              detail: { label: 'Cannot connect', detail: connectionRejectionMessage(check.reason), status: 'warn' },
            }));
            return;
          }
          setEdges((eds) => [...eds, { id: `edge-${check.source}-${check.target}`, source: check.source, target: check.target, data: {} }]);
          window.dispatchEvent(new CustomEvent('builder-toast', {
            detail: { label: 'Connected', detail: 'Added a connection between the two layers.', status: 'ok' },
          }));
        }
        setShapeInferenceHasRun(false);
        setVerifyResult(null);
        return;
      }
      if (command === 'undo') {
        // Model history first; fall back to tldraw for pure canvas moves.
        if (!undoModelEdit() && editorRef.current) editorRef.current.undo();
        return;
      }
      if (command === 'redo') {
        if (!redoModelEdit() && editorRef.current) editorRef.current.redo();
        return;
      }
      if (command === 'fit-view') {
        fitView();
        return;
      }
      if (command === 'toggle-fullscreen') {
        if (!document.fullscreenElement) {
          reactFlowWrapper.current?.requestFullscreen?.().catch((err) => {
            console.error(`Error attempting to enable fullscreen: ${err.message}`);
          });
        } else {
          document.exitFullscreen?.();
        }
        return;
      }
      if (command === 'view-overview') {
        setViewMode('overview');
        return;
      }
      if (command === 'view-expanded') {
        setViewMode('expanded');
        return;
      }
      if (command === 'set-graph-detail') {
        if (payload?.level === 'layers' || payload?.level === 'operations') setDetailLevel(payload.level);
        return;
      }
      if (command === 'toggle-expanded-svg') {
        setSvgOverview(payload?.active === undefined ? (value: boolean) => !value : Boolean(payload.active));
        return;
      }
      if (command === 'set-layout-direction') {
        const choice = payload?.choice;
        if (isLayoutChoice(choice)) setLayoutChoice(choice);
        return;
      }
      if (command === 'show-unchanged-calls') {
        applyLogicalGraphRef.current?.(showPassThroughCalls(nodesRef.current, payload?.shown === true),
          edgesRef.current, { relayout: true });
        return;
      }
      if (command === 'view-focus') {
        focusSelectedModule();
        return;
      }
      if (command === 'report-selection') { reportSelection(); return; }
      if (command === 'focus-selection') {
        pendingCanvasFocus.current?.();
        let plan = planCanvasFocus(nodesRef.current, edgesRef.current, payload?.nodeIds ?? [], payload?.edgeIds ?? []);
        const target = payload?.target;
        const stillCurrent = () => {
          const projection = currentProjection();
          return !!projection && !!target && projection.file === target.file && projection.cls === target.cls &&
            projection.sourceDigest === target.sourceDigest && projection.revision === target.backendRevision;
        };
        if (!plan || !stillCurrent()) { payload?.finish?.(false); return; }
        setSvgOverview(false);
        const targets = plan.edge ? [plan.edge.source, plan.edge.target] : plan.nodeIds;
        const scope = focusScopeForTargets(plan.nodes, edgesRef.current, targets);
        if (scope) plan = planCanvasFocus(nodesRef.current, edgesRef.current, payload?.nodeIds ?? [], payload?.edgeIds ?? [], false)!;
        moduleFocusRef.current.navigate(scope, { immediate: true, nodes: plan.nodes });
        const navigationRevision = moduleFocusRef.current.navigationRevision();
        if (plan.nodes.some((node, index) => node !== nodesRef.current[index])) {
          applyLogicalGraphRef.current?.(plan.nodes, edgesRef.current, { relayout: !scope });
        }
        const deadline = Date.now() + 2000;
        let frame = 0, settled = false;
        const finish = (ok: boolean) => {
          if (settled) return;
          settled = true;
          cancelAnimationFrame(frame);
          pendingCanvasFocus.current = null;
          if (ok) reportSelection();
          payload?.finish?.(ok);
        };
        pendingCanvasFocus.current = () => finish(false);
        const attempt = () => {
          if (!stillCurrent() || !editorRef.current || Date.now() > deadline || moduleFocusRef.current.navigationRevision() !== navigationRevision) { finish(false); return; }
          const displayedEdge = plan!.edge && scope ? moduleFocusRef.current.graph.edges.find(edge => edge.id === plan!.edge!.id) : plan!.edge;
          if (moduleFocusRef.current.isReady() && (!plan!.edge || displayedEdge)
            && applyCanvasFocus(editorRef.current, { ...plan!, edge: displayedEdge }, payload?.inset)) {
            // Explicit focus owns the camera; a queued initial fit must not
            // zoom back out to the entire graph after the change is revealed.
            initialFitGraphIdRef.current = fitGraphKey;
            fittedSweepRunIdRef.current = sweepRunId;
            fittedBlocksRequestRef.current = blocksFitRequest;
            finish(true); return;
          }
          frame = requestAnimationFrame(attempt);
        };
        frame = requestAnimationFrame(attempt);
        return;
      }
      if (command === 'focus-node') {
        const nodeId = payload?.nodeId;
        if (typeof nodeId === 'string') {
          focusNodeById(nodeId);
        }
        return;
      }
      if (command === 'validation-complete') {
        const completed = payload?.projection;
        const current = currentProjection();
        if (readOnlyLocalBrowser && current?.traced && !current.greyed && completed
          && current.graphId === completed.graphId && current.revision === completed.revision
          && current.sourceDigest === completed.sourceDigest) setSweepRunId(id => id + 1);
        return;
      }
      if (command === 'apply-projection') {
        // The IDE re-projects the file on every settled keystroke, so this
        // merges by node id and never relayouts: applyProjection keeps the
        // position of every card that is already on the canvas and places only
        // the ones the edit created (task.md M3).
        const projection = payload?.projection;
        if (!projection) return;
        const merged = applyIdeProjection(projection, nodesRef.current, liveLayoutDirection(projection));
        if (merged) {
          pendingRenderedProjection.current = projection;
          graphMetadataRef.current = merged.metadata ?? {};
          rememberProjectionView(projection);
          // A source draft is laid out fresh; a draft drawn in the retained capture keeps its arrangement.
          const review = projection as import('../ide/graphReview').ReviewProjection;
          applyLogicalGraph(merged.nodes, merged.edges, { relayout: review.proposal === true && !review.capturedContext,
            preserveChildPositions: 'reviewGraph' in projection });
        }
        return;
      }
      if (command === 'export-image') {
        // Copy (clipboard) or Save (file). See canvas/exportImage.ts.
        const format = payload?.format === 'png' ? 'png' : 'svg';
        const save = payload?.save === true;
        const report = (outcome: ExportOutcome) =>
          window.dispatchEvent(new CustomEvent('builder-toast', {
            detail: {
              label: outcome.ok ? `${save ? 'Saved' : 'Canvas copied'} as ${format.toUpperCase()}` : 'Export failed',
              detail: outcome.message,
              status: outcome.ok ? 'ok' : 'warn',
            },
          }));
        // SVG is the logical graph as <text>, including outside the overview.
        // tldraw's image export is foreignObject HTML and is kept for PNG.
        if (format === 'svg') {
          const displayed = moduleFocusRef.current.graphRef.current;
          if (displayed.nodes.length) {
            const svg = graphToSvg(displayed.nodes, displayed.edges, { theme: getCanvasTheme(), direction: layoutDirection,
              explanationStates: explanationContextRef.current?.annotations ?? explanationContextRef.current?.states, annotationsVisible: explanationContextRef.current?.annotationsVisible,
              viewExplanation: explanationContextRef.current?.model?.state });
            if (save) hostServices.saveFile('graph.svg', new Blob([svg], { type: 'image/svg+xml' }));
            else void hostServices.copyText(svg);
            report({ ok: true, message: save ? 'Saved the SVG.' : 'Copied the SVG.' });
            return;
          }
        }
        void (save ? saveCanvasImage(editorRef.current, format) : exportCanvasImage(editorRef.current, format)).then(report);
        return;
      }
      if (command === 'set-parameters') {
        const nodeId = payload?.nodeId;
        const parameters = payload?.parameters;
        if (typeof nodeId === 'string' && parameters && typeof parameters === 'object') {
          updateNodeParameters(nodeId, parameters);
        }
        return;
      }
      if (command === 'group-module') {
        void groupSelectionAsModule();
        return;
      }
      if (command === 'ungroup-module') {
        ungroupModule(payload?.nodeId || selectedModuleInstanceId || selectedNodeId || '');
        return;
      }
    };
    window.addEventListener('builder-command', handler);
    return () => window.removeEventListener('builder-command', handler);
  }, [
    handleSave,
    undoModelEdit,
    redoModelEdit,
    enableExternalCommands,
    runVerification,
    runExportPackage,
    triggerShapeInference,
    applySuggestedShapeRepairs,
    autoLayoutCanvas,
    editorRef,
    fitView,
    fitGraphKey,
    sweepRunId,
    blocksFitRequest,
    svgOverview,
    setViewMode,
    setDetailLevel,
    focusSelectedModule,
    groupSelectionAsModule,
    addNodeFromPalette,
    instantiateTemplateOnCanvas,
    focusNodeById,
    reportSelection,
    updateNodeParameters,
    deleteSelectedNodes,
    ungroupModule,
    selectedModuleInstanceId,
    selectedNodeId,
    applyLogicalGraph,
    readOnlyLocalBrowser,
    services,
    layoutDirection,
    liveLayoutDirection,
  ]);

  // Drain a projection that arrived before this component existed.
  //
  // In the IDE the bridge posts `ready` from module scope, so the host answers
  // with the projection while React is still mounting and the `builder-command`
  // listener above does not exist yet — the first "Open as Graph" of a session
  // hits that window every time. The bridge keeps the last projection; this
  // takes it. Registered after the listener and in the same commit, so no
  // message can slip between the two.
  useEffect(() => {
    if (!enableExternalCommands) return;
    const pending = currentProjection();
    if (!pending) return;
    const merged = applyIdeProjection(pending, nodesRef.current, liveLayoutDirection(pending));
    if (merged) {
      pendingRenderedProjection.current = pending;
      graphMetadataRef.current = merged.metadata ?? {};
      rememberProjectionView(pending);
      applyLogicalGraph(merged.nodes, merged.edges, { relayout: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enableExternalCommands]);

  // Convert nodes/edges to ModelGraph format for ModelSummary
  const modelGraphNodes: ModelNode[] = nodes.map(node => ({
    id: node.id,
    type: node.type || 'layerNode',
    position: node.position,
    data: node.data,
  }));

  const modelGraphEdges: ModelEdge[] = edges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle || undefined,
    targetHandle: edge.targetHandle || undefined,
    data: edge.data,
  }));

  return (
    <>
      <div className="model-builder">
        <ModuleFocusSurface focus={moduleFocus} nodes={nodes} title={currentProjection()?.cls || graphIdentity.name} selectedId={selectedNodeId}>
        <div className="canvas-container" ref={reactFlowWrapper}>

            <TldrawCanvas
              sidePanelOpen={false}
              readOnly={readOnlyLocalBrowser || moduleFocus.active}
              onEditorReady={handleFocusEditorReady}
              onSelectionChange={handleTldrawSelectionChange}
              onToolChange={setCanvasToolId}
              onShapeClick={handleShapeClick}
              onShapeDoubleClick={handleShapeDoubleClick}
              onCanvasClick={handleCanvasClick}
              onExternalDrop={onDrop}
              onExternalDragOver={onDragOver}
              connectEdges={connectEdges}
              onConnect={commitConnection}
              onInsertOnEdge={insertNodeOnEdge}
            >
              <ValidationSweepOverlay
                editor={editorRef.current}
                nodes={moduleFocus.graph.nodes}
                edges={moduleFocus.graph.edges}
                runId={sweepRunId}
                measuredOnly={readOnlyLocalBrowser}
              />
            </TldrawCanvas>
          {svgOverview && <GraphSvgLayer nodes={moduleFocus.graph.nodes} edges={moduleFocus.graph.edges} direction={layoutDirection}
            onActivate={id => {
              const node = nodesRef.current.find(n => n.id === id);
              if (node?.data.operationGroup) handleRevealOperations(id);
              else if (node?.type === 'moduleNode') handleFlatToggleExpand(id);
            }}
            selectedIds={selectedIds} editor={editorRef.current}
            onSelect={id => handleSelectionChange(id && nodesRef.current.some(node => node.id === id) ? [id] : [])} />}
        </div>
        </ModuleFocusSurface>
        <SaveGraphDialog
          isOpen={showSaveDialog}
          onClose={() => setShowSaveDialog(false)}
          onSave={handleSave}
          existingGraph={saveDialogGraph}
          suggestedName={!initialGraph ? generateSuggestedName() : undefined}
        />

        <CodeExportDialog
          isOpen={showCodeExportDialog}
          onClose={() => setShowCodeExportDialog(false)}
          graph={{ nodes: modelGraphNodes, edges: modelGraphEdges }}
          graphName={graphIdentity.name}
          onAddInputNode={addWiredInputNode}
        />

        <CodeImportDialog
          isOpen={showCodeImportDialog}
          onClose={() => setShowCodeImportDialog(false)}
          onImport={handleImport}
        />

        {showTemplateGallery && (
          <ModernTemplateGallery
            onTemplateSelected={handleTemplateFromGallery}
            onClose={() => setShowTemplateGallery(false)}
          />
        )}
      </div>
    </>
  );
};

export default ModelBuilder; 
