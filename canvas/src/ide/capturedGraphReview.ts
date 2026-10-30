import type { GraphReviewState, ProjectionPayload } from './protocol.generated';
import type { GraphDocumentV2 } from '../types/graphDocument';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import type { GraphReviewPresentation, ReviewNode, ReviewProjection } from './graphReview';
import { assignmentContext } from './assignmentContext';

/** A pending nested-assignment draft shown on the calls it replaces, inside the
 * retained capture (or the run an accepted source view still carries). Only when
 * every draft change is such an assignment; anything else keeps the source draft. */
export function draftAssignmentReview(state: GraphReviewState, source: GraphReviewPresentation,
  capture: ProjectionPayload | null): GraphReviewPresentation | null {
  if (!capture || capture.greyed || capture.file !== state.liveContext?.file || capture.cls !== state.liveContext?.cls
    || !source.changes.length) return null;
  const context = assignmentContext(capture.traced ? capture.document : capture.debug?.previousDocument,
    state.after.document, state.before.document);
  if (!context) return null;
  const nodes = new Map(source.nodes), overlays = new Map<string, ReviewNode>();
  const focusNodes = new Map<string, string[]>();
  for (const [key, calls] of Array.from(context.calls)) {
    nodes.set(key, { status: 'unchanged', before: key, after: key, params: {} });
    for (const id of calls) nodes.set(id, nodes.get(key)!);
  }
  const changes: GraphReviewPresentation['changes'] = [];
  for (const change of source.changes) {
    const record = source.nodes.get(change.id)!;
    const calls = record.after && change.id === `review:after:${record.after}` ? context.calls.get(record.after) : undefined;
    if (!calls) return null;
    const detail = `${(record.summary ?? change.detail).replace(/ · not run$/, '')} · ${calls.length} previous call${calls.length === 1 ? '' : 's'} · not run`;
    const review = { ...record, params: {}, summary: detail };
    nodes.set(record.after!, review);
    for (const id of calls) { overlays.set(id, review); nodes.set(id, review); }
    calls.forEach((call, index) => {
      const id = `${change.id}:call:${index}`;
      nodes.set(id, review);
      focusNodes.set(id, [call]);
      changes.push({ id, label: `${change.label} · previous call ${index + 1}/${calls.length}`, detail });
    });
  }
  const reviewGraph = { ...context.graph, nodes: context.graph.nodes.map(node => overlays.has(node.id)
    ? { ...node, data: { ...node.data, review: overlays.get(node.id), diffStatus: 'modified' as const } } : node) };
  // The proposed source stays the authority for identity, navigation and focus;
  // the capture supplies only the displayed hierarchy.
  const projection: ReviewProjection = { ...state.after, graphId: capture.graphId, reviewId: state.id, proposal: true,
    capturedContext: true, viewRestored: true, reviewGraph, contextKey: context.key };
  return { ...source, inheritedBase: false, nodes, changes, focusNodes, projections: { before: projection, diff: projection, after: projection } };
}

/** Keep the last capture as context. Only exact-snapshot correspondence may attach
 * a source edit to a captured call; never construct runtime Before/After graphs. */
export function capturedGraphReview(state: GraphReviewState, source: GraphReviewPresentation,
  capture: ProjectionPayload | null): GraphReviewPresentation | null {
  if (!capture?.traced || capture.file !== state.liveContext?.file || capture.cls !== state.liveContext?.cls) return null;
  const document = capture.document as GraphDocumentV2;
  const graph = graphDocumentV2ToModelGraph(document);
  const metadata = document.semantic.metadata;
  const rootInvocation = (metadata.capture as { rootInvocationId?: string } | undefined)?.rootInvocationId;
  const root = graph.nodes.find(node => node.data.captureRef?.kind === 'invocation' && node.data.captureRef.semanticId === rootInvocation);
  if (!root) return null;
  // The extension traces a nested project from that project's root; review names
  // files from the repository root. Move the capture's names into review's frame.
  const reviewed = state.models?.find(model => model.id === state.selectedModel)?.file ?? state.after.file;
  const prefix = reviewed.endsWith(`/${capture.file}`) ? reviewed.slice(0, -capture.file.length) : '';
  const reframe = (key: string) => prefix ? key.replace(/^@declaration:\[("(?:[^"\\]|\\.)*")/,
    (_, file: string) => `@declaration:[${JSON.stringify(prefix + JSON.parse(file))}`) : key;
  const side = (['after', 'before'] as const).find(side => {
    const revision = state[side];
    const dependencies = new Map(revision.dependencies?.map(entry => [entry.file, entry.digest]));
    return capture.sourceDigest === revision.sourceDigest && !!capture.dependencies?.length
      && capture.dependencies.every(entry => dependencies.get(prefix + entry.file) === entry.digest);
  });
  const correspondence = (metadata.sourceCorrespondence ?? []) as {
    nodeId: string; captureId: string; basis: string; invocationIds: string[];
  }[];
  const constructorCalls = Object.fromEntries(Object.entries((metadata.sourceDeclarationCalls ?? {}) as Record<string, string[]>)
    .map(([key, calls]) => [reframe(key), calls]));
  const nodes = new Map(source.nodes), focusNodes = new Map<string, string>();
  const overlays = new Map<string, ReviewNode>();
  const calls = new Map(graph.nodes.filter(node => node.data.captureRef?.kind === 'invocation')
    .map(node => [node.data.captureRef!.semanticId, node]));
  const unmapped = new Set<string>();
  const changes: GraphReviewPresentation['changes'] = [];
  const locatedDeclarations = new Set<string>();
  for (const change of source.changes) {
    const record = source.nodes.get(change.id)!;
    if (change.id === 'review:root') {
      overlays.set(root.id, record); focusNodes.set(change.id, root.id); changes.push(change); continue;
    }
    const fact = side && !record.ambiguousDeclaration && correspondence.find(fact => fact.nodeId === record[side]
      && fact.captureId === root.data.captureRef!.captureId && fact.basis === 'registered_path');
    const declaration = side && !record.ambiguousDeclaration && record.sourceDeclaration?.[side];
    const registeredCalls = fact ? fact.invocationIds : undefined;
    const invocationIds = registeredCalls?.length ? registeredCalls : declaration ? constructorCalls[declaration] : undefined;
    const mapped = Array.from(new Set(invocationIds ?? [])).map(id => calls.get(id)).filter(node => node !== undefined);
    if (!mapped.length) {
      unmapped.add(change.id); focusNodes.set(change.id, root.id);
      changes.push({ ...change, detail: `${change.detail} · Not located in this capture` }); continue;
    }
    // Several lowered copies can describe the same constructor expression.
    // Show each captured call once when that expression supplies the mapping.
    const declarationKey = declaration && !registeredCalls?.length
      ? JSON.stringify([declaration, record.status, change.detail, record.params]) : undefined;
    if (declarationKey && locatedDeclarations.has(declarationKey)) continue;
    if (declarationKey) locatedDeclarations.add(declarationKey);
    mapped.forEach((node, index) => {
      const id = mapped.length === 1 ? change.id : `${change.id}:call:${index}`;
      overlays.set(node!.id, { ...record, params: {}, summary: change.detail });
      nodes.set(id, record);
      focusNodes.set(id, node!.id);
      changes.push({ ...change, id, label: mapped.length === 1 ? change.label : `${change.label} · call ${index + 1}/${mapped.length}` });
    });
  }
  const note = `${unmapped.size} source change${unmapped.size === 1 ? '' : 's'} cannot be located in this capture`;
  if (unmapped.size && !overlays.has(root.id)) overlays.set(root.id, { status: 'unresolved', params: {},
    before: '@class', after: '@class', summary: note });
  const contextGraph = { ...graph, nodes: graph.nodes.map(node => {
    const review = overlays.get(node.id);
    if (review) nodes.set(node.id, review);
    return { ...node, data: { ...node.data, review,
      ...(capture.greyed ? { shapeEvidence: 'stale' as const } : {}),
      // Colour the source change on existing captured calls; never add/remove
      // runtime nodes or connections based on source comparison alone.
      diffStatus: !review ? undefined : review.status === 'added' && side === 'after' ? 'added' as const
        : review.status === 'removed' && side === 'before' ? 'removed' as const
          : review.status === 'rewired' ? 'rewired' as const : 'modified' as const,
    } };
  }) };
  const projection: ReviewProjection = { ...capture, reviewId: state.id,
    capturedContext: true, viewRestored: true, reviewGraph: contextGraph };
  return { ...source, nodes, focusNodes, projections: { before: projection, diff: projection, after: projection },
    initialChangeId: changes.find(change => !unmapped.has(change.id)
      && nodes.get(change.id)?.status !== 'unresolved')?.id, changes };
}
