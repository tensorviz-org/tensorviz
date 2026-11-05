import { dropoutInsertionReveal } from './dropoutInsertion';
import { wrapRepeatBlocks } from './repeatBlocks';
/**
 * The app's half of "Open as Graph" (task.md M3).
 *
 * The host owns the file and the parse; this owns what the canvas does with the
 * result. Three things happen here and nowhere else:
 *
 *  1. a `projection` from the host becomes a `builder-command`, so the canvas is
 *     driven through the same window bus every other external caller uses
 *     (BuilderPageV2's `dispatchBuilderCommand`) rather than a second, private
 *     channel into ModelBuilder;
 *  2. the merge result is reported back as `projectionApplied` — counted off the
 *     merged graph, so it is what the user can see and not what the host hoped
 *     for;
 *  3. the projection payload is remembered, because the IDE save path needs to
 *     know which graph id the canvas is showing and whether its structure has
 *     drifted from the file.
 */
import { ModelEdge, ModelGraph, ModelNode } from '../types';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import { GraphDocumentV2 } from '../types/graphDocument';
import { postToHost } from './bridge';
import { mergeProjection } from './applyProjection';
import { reconcileModuleView } from '../services/canvasLayout';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { ProjectionPayload } from './protocol.generated';
import { tensorShapes } from '../canvas/graphSemantics';
import { graphDebugPresentation } from './graphDebugPresentation';
import { assignmentContext } from './assignmentContext';

/** The last projection the host sent, or null in web mode / before the first. */
let current: ProjectionPayload | null = null;
let appliedGraphIdentity: string | undefined;
let pendingInitialLayout: { graphId: string; positions: Map<string, { x: number; y: number }> } | undefined;
let reviewReturn: { file: string; cls: string; nodes: readonly ModelNode[]; context?: string; sourceDigest?: string } | undefined;
/** The view of a closed captured-context draft, kept for its accepted source only. */
let acceptedView: { file: string; cls: string; context: string; sourceDigest?: string; nodes: readonly ModelNode[] } | undefined;

/** The browser page's answer to "the host is a WebSocket away, not a frame
 *  away": the external local-browser page has no `window.parent` shell for
 *  postToHost to reach, so LocalBrowserApp registers its view transport here
 *  for the life of its session. In the framed IDE pane this stays null and
 *  reportNodeClick behaves exactly as it always has. */
let browserProjectExpand: ((nodeId: string, expanded: boolean) => void) | null = null;

export function setBrowserProjectExpandSink(sink: typeof browserProjectExpand): void {
  browserProjectExpand = sink;
}

let browserRevealNode: ((nodeId: string) => void) | null = null;

export function setBrowserRevealNodeSink(sink: ((nodeId: string) => void) | null): void {
  browserRevealNode = sink;
}

export function currentProjection(): ProjectionPayload | null {
  return current;
}

/** True when this document is running inside the tensorViz IDE. Set by
 *  src/ide/main.tsx at startup; the web bundle never evaluates that file. */
export function ideMode(): boolean {
  return typeof window !== 'undefined' && window.__TENSORVIZ_IDE__ === true;
}

/** True only for the authenticated external page opened by the extension.
 * It uses the IDE bundle, but unlike the framed canvas it is a read-only view:
 * its broker capability set permits navigation, never source or graph writes. */
export function localBrowserMode(): boolean {
  return typeof window !== 'undefined' && window.__TENSORVIZ_LOCAL_BROWSER__ === true;
}

/** Hand a host projection to the canvas. */
export function receiveProjection(payload: ProjectionPayload): void {
  current = payload;
  window.dispatchEvent(
    new CustomEvent('builder-command', {
      detail: { command: 'apply-projection', payload: { projection: payload } },
    }),
  );
}

/** A PROJECT document, if that is what this is.
 *
 * The project plane sends an `ExperimentDocument` — `{experiment_id, nodes, edges,
 * spanmap}`. The class plane sends a `GraphDocumentV2`, whose drawable graph is
 * nested under `.document` in `semantic`/`view` halves. Both arrive on the same
 * message, and the class adapter reads `document.semantic.metadata` first — so
 * handing it an experiment document threw before it could look at a single node.
 * The canvas logged the error and kept its previous contents, which on a fresh
 * page is nothing: an empty canvas, for a document full of nodes.
 *
 * The conversion is almost identity, because the backend already emits card-shaped
 * nodes — `{id, type, position, data}` with `data.layerType.id = "custom_code"` —
 * which is exactly what `ModelNode extends Node` is. What was missing was
 * recognising them.
 */
let showingProject = false;

/** True when the canvas is currently drawing a PROJECT, on either surface.
 *
 * A project graph is a projection of source, like the external page's: its nodes
 * are classes and files, not layers wired into a forward pass. Anything that
 * reasons about the canvas as a runnable model — shape inference above all — must
 * refuse on it, and `localBrowserMode()` is the wrong question to ask, because
 * the panel draws the same projection without being that page. */
export function showingProjectGraph(): boolean {
  return showingProject;
}

function experimentGraph(payload: ProjectionPayload): ModelGraph | null {
  const candidate = payload as unknown as {
    experiment_id?: unknown;
    nodes?: unknown;
    edges?: unknown;
  };
  if (typeof candidate.experiment_id !== 'string' || !Array.isArray(candidate.nodes)) {
    return null;
  }
  // The one thing that is NOT identity. The document says which frame a child
  // belongs to with `parentNode`, the field the wire contract has always used. The
  // canvas says it with `data.parentId` — every render path does: the reconciler
  // that hides a collapsed frame's children and sizes an open one from them, the
  // repeat-block wrapper, the grouping code. Nothing translated between the two,
  // so a frame that said "9 layers" drew an empty box: its children were in the
  // document with the right parent, and the reconciler looked at a field they
  // did not have and saw nine top-level cards.
  const nodes = (candidate.nodes as ModelNode[]).map((node) =>
    node.parentNode && !node.data?.parentId
      ? { ...node, data: { ...node.data, parentId: node.parentNode } }
      : node,
  );
  return {
    nodes,
    edges: Array.isArray(candidate.edges) ? (candidate.edges as ModelEdge[]) : [],
  };
}

/**
 * Merge a projection into the canvas and report what the canvas ended up with.
 *
 * Called by ModelBuilder from its `builder-command` handler; kept here so the
 * merge, the report and the remembered payload cannot drift apart.
 */
export function applyProjection(
  payload: ProjectionPayload,
  existingNodes: readonly ModelNode[],
  direction?: LayoutDirection,
): ModelGraph | null {
  const review = payload as import('./graphReview').ReviewProjection;
  if (review.reviewGraph) {
    reviewReturn ??= { file: payload.file, cls: payload.cls, nodes: existingNodes };
    reviewReturn.context = review.contextKey;
    reviewReturn.sourceDigest = payload.sourceDigest;
    acceptedView = undefined;
    current = payload; showingProject = false;
    const existing = review.capturedContext
      ? (existingNodes.some(node => node.data.captureRef) ? existingNodes : reviewReturn.nodes)
      : existingNodes.filter(node => node.id.startsWith('review:'));
    return mergeProjection(review.reviewGraph, existing, { viewRestored: true });
  }
  if (reviewReturn) {
    const same = reviewReturn.file === payload.file && reviewReturn.cls === payload.cls;
    if (same && reviewReturn.context) acceptedView = { file: payload.file, cls: payload.cls, context: reviewReturn.context,
      sourceDigest: reviewReturn.sourceDigest, nodes: existingNodes };
    existingNodes = same ? reviewReturn.nodes : [];
    reviewReturn = undefined;
  }
  // Discard, refusal or any other source restores the live view. Only the accepted
  // source, proven against the same capture and structure, keeps the draft's view;
  // the host may redraw the old capture between closing the draft and that source.
  if (acceptedView) {
    const sameModel = acceptedView.file === payload.file && acceptedView.cls === payload.cls;
    const context = sameModel && !payload.traced && payload.debug?.previousDocument
      ? assignmentContext(payload.debug.previousDocument, payload.document)?.key : undefined;
    const run = (payload.document as GraphDocumentV2 | undefined)?.semantic?.metadata?.capture as { id?: string } | undefined;
    if (context === acceptedView.context && payload.sourceDigest === acceptedView.sourceDigest) existingNodes = acceptedView.nodes;
    if (context === acceptedView.context || !sameModel || !payload.traced || !acceptedView.context.startsWith(`${run?.id}:`)) acceptedView = undefined;
  }
  // Node ids (especially Input/Output) can be shared by different models.
  // Preserve presentation on refresh only within the same graph. Otherwise an
  // Output can inherit the previous model's y coordinate after the new layout.
  if (appliedGraphIdentity && appliedGraphIdentity !== payload.graphId) existingNodes = [];
  let graph: ModelGraph;
  const experiment = experimentGraph(payload);
  showingProject = experiment !== null;
  if (experiment) {
    graph = experiment;
  } else {
    try {
      graph = graphDocumentV2ToModelGraph(payload.document as GraphDocumentV2);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('tensorViz IDE: could not read the projection document', error);
      return null;
    }
  }
  if (localBrowserMode() && payload.traced) {
    graph = { ...graph, nodes: graph.nodes.map(node => {
      const params = node.data.parameters;
      return { ...node, data: { ...node.data,
        ...tensorShapes({ parameters: params }),
        error: payload.traceError?.nodeId === node.id ? payload.traceError.message : node.data.captureRef ? node.data.error : undefined,
        shapeEvidence: payload.greyed ? 'stale' : params._shape_provenance === 'inferred' ? 'inferred'
          : ['observed', 'state'].includes(params._shape_provenance) ? 'measured' : 'unknown',
      } };
    }) };
  }
  // A folded repeat arrives as one bare card carrying `_repeat_count`. Wrapping it
  // before the merge makes it the module frame the rest of the product already uses,
  // so a ModuleList of four opens and closes like any other block instead of reading
  // as three missing layers.
  const canonical = graph.nodes.some(node => node.data.captureRef);
  const initialCapture = canonical && !payload.viewRestored
    && !existingNodes.some(node => node.data.captureRef?.captureId === graph.nodes[0]?.data.captureRef?.captureId);
  const debugging = localBrowserMode() && !experiment ? graphDebugPresentation(payload, graph, existingNodes) : undefined;
  if (debugging) graph = debugging.graph;
  let retainedNodes = debugging?.existing ?? existingNodes;
  // A first capture can arrive before the opening panel leaves "capturing".
  // Its temporary stale badges change card heights. Settle that automatic
  // layout once badges clear, but never override a card the user has moved.
  const settleInitial = pendingInitialLayout && pendingInitialLayout.graphId === payload.graphId && !payload.greyed && !payload.viewRestored
    && Array.from(pendingInitialLayout.positions).every(([id, position]) => {
      const node = retainedNodes.find(node => node.id === id);
      return node && Math.abs(node.position.x - position.x) < 1 && Math.abs(node.position.y - position.y) < 1;
    });
  if (settleInitial) retainedNodes = [];
  if (initialCapture || settleInitial) {
    // Diagnostic/source presentation affects dimensions, so it precedes layout.
    graph = { ...graph, ...reconcileModuleView(graph.nodes, graph.edges, { relayout: true, direction }) };
  }
  const merged = mergeProjection(canonical ? graph : wrapRepeatBlocks(graph), retainedNodes, { viewRestored: payload.viewRestored, direction });
  merged.nodes = dropoutInsertionReveal.apply(payload, merged.nodes);
  if (canonical && retainedNodes.length === 0 && !payload.viewRestored && payload.greyed) {
    pendingInitialLayout = { graphId: payload.graphId,
      positions: new Map(merged.nodes.filter(node => !node.hidden).map(node => [node.id, { ...node.position }])) };
  } else if (!payload.greyed || payload.viewRestored || pendingInitialLayout?.graphId !== payload.graphId) {
    pendingInitialLayout = undefined;
  }
  appliedGraphIdentity = payload.graphId;
  current = payload;
  postToHost({
    type: 'projectionApplied',
    nodeCount: merged.nodes.length,
    traced: payload.traced,
    greyed: payload.greyed,
    sourceDigest: payload.sourceDigest,
  });
  return { nodes: merged.nodes, edges: merged.edges, metadata: graph.metadata };
}

/** Open or close a project frame (step 3.3). Static: the host reads source and
 *  re-draws, and nothing is executed — which is why this is not `expandNode`. */
export function reportProjectExpand(nodeId: string, expanded: boolean): void {
  browserProjectExpand?.(nodeId, expanded);
}

/** A node was activated on the canvas; ask the host to reveal its line. The
 *  framed pane posts to the shell; the external browser page answers through
 *  the registered broker sink — one canvas seam, one outward route per surface. */
export function reportNodeClick(nodeId: string): void {
  if (!current) return;
  browserRevealNode?.(nodeId);
}
