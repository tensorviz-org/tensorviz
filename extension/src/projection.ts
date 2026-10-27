/** Pure projection helpers shared by the current Canvas and source editing. */

export const TYPING_DEBOUNCE_MS = 300;
export const SAVE_COALESCE_MS = 1_500;
export const TOKEN_HEADER = 'X-tensorViz-Token';

export interface BackendEndpoint {
  origin(): string | undefined;
  headers(): Record<string, string>;
}

interface Span {
  file: string;
  start: [number, number];
  end: [number, number];
  kind: string;
  digest: string;
}

/** One observed tensor output shape, anchored to the end of its expression. */
export interface ShapeHint {
  nodeId: string;
  /** 1-based, as the spanmap counts. */
  line: number;
  /** 0-based UTF-16 column, as the captured span and VS Code count. */
  column: number;
  shape: string;
  provenance: string;
}

/** Measured tensor outputs belong to executed expressions, including imported
 * files. A declaration alone cannot identify which invocation produced a shape. */
export function shapeHintsByFile(parsed: {
  traced: boolean;
  document: unknown;
  spanmap: Record<string, unknown>;
}): Map<string, ShapeHint[]> {
  const byFile = new Map<string, ShapeHint[]>();
  if (!parsed.traced) {
    return byFile;
  }
  const semantic = (parsed.document as { semantic?: { nodes?: unknown[] } } | undefined)?.semantic;
  type MeasuredNode = { node_id?: string; params?: Record<string, unknown>; metadata?: {
      captureRef?: { kind?: string }; parentId?: string;
      capturePorts?: Array<{ direction: string; valueId?: string | null; shape?: number[] | null }>;
    } };
  const nodes = (semantic?.nodes ?? []) as MeasuredNode[];
  const operations = new Map<string, MeasuredNode[]>();
  for (const node of nodes) {
    if (node.metadata?.captureRef?.kind !== 'operation') { continue; }
    const span = parsed.spanmap[node.node_id ?? ''] as Span | undefined;
    if (!span || span.kind !== 'call') { continue; }
    const key = JSON.stringify([node.metadata.parentId, span.file, span.start, span.end]);
    const group = operations.get(key) ?? [];
    group.push(node);
    operations.set(key, group);
  }
  const expressionResults = new Set<MeasuredNode>();
  for (const group of operations.values()) {
    // One authored slice can decompose into several ATen operations. Follow
    // their captured values, never their record order or an intermediate shape.
    if (group.some(node => node.params?._shape_provenance !== 'observed'
      || !node.metadata?.capturePorts?.some(port => port.direction === 'out'))) { continue; }
    const consumed = new Set(group.flatMap(node => (node.metadata?.capturePorts ?? [])
      .filter(port => port.direction === 'in' && port.valueId).map(port => port.valueId)));
    const results = group.filter(node => node.metadata?.capturePorts?.some(port =>
      port.direction === 'out' && (!port.valueId || !consumed.has(port.valueId))));
    // Multiple escaping results do not prove which tensor the expression returns.
    if (results.length === 1) { expressionResults.add(results[0]); }
  }
  for (const node of nodes) {
    const nodeId = String(node.node_id ?? '');
    const params = node.params ?? {};
    const measured = params._out_shape;
    const shape = typeof measured === 'string' ? measured
      : Array.isArray(measured) && measured.every(dim => Number.isInteger(dim) && dim >= 0)
        ? (measured.length ? measured.join('x') : 'scalar') : undefined;
    if (!nodeId || shape === undefined || params._shape_provenance !== 'observed') {
      continue;
    }
    if (node.metadata?.captureRef && node.metadata.captureRef.kind !== 'invocation'
      && !expressionResults.has(node)) {
      continue;
    }
    const span = parsed.spanmap[nodeId] as Span | undefined;
    if (!span || span.kind !== 'call' || typeof span.file !== 'string' || !span.end) {
      continue;
    }
    // Torch internals inherit their outer call's source binding. Their temporary
    // shapes are not the result of the expression the user wrote, even when the
    // outer module has no single tensor output to display.
    const parent = node.metadata?.parentId ? parsed.spanmap[node.metadata.parentId] as Span | undefined : undefined;
    if (parent?.file === span.file && parent.start?.[0] === span.start?.[0]
      && parent.start?.[1] === span.start?.[1] && parent.end?.[0] === span.end[0]
      && parent.end?.[1] === span.end[1]) {
      continue;
    }
    const hints = byFile.get(span.file) ?? [];
    hints.push({
      nodeId,
      line: span.end[0],
      column: span.end[1],
      shape,
      provenance: String(params._shape_provenance),
    });
    byFile.set(span.file, hints);
  }
  for (const [file, hints] of byFile) {
    // Nested expressions can end at the same character. Put the inner result
    // first and share a hint when both results have the same measured shape.
    hints.sort((a, b) => a.line - b.line || a.column - b.column
      || (parsed.spanmap[b.nodeId] as Span).start[0] - (parsed.spanmap[a.nodeId] as Span).start[0]
      || (parsed.spanmap[b.nodeId] as Span).start[1] - (parsed.spanmap[a.nodeId] as Span).start[1]);
    const seen = new Set<string>();
    byFile.set(file, hints.filter(hint => {
      const key = JSON.stringify([hint.line, hint.column, hint.shape]);
      if (seen.has(key)) { return false; }
      seen.add(key);
      return true;
    }));
  }
  return byFile;
}

/**
 * Merge a saved sidecar layout into a projection document, and say whether any of
 * it actually landed.
 *
 * Pure and exported so the answer can be asserted directly: it decides whether
 * opening the graph arranges it, and the difference between "the user placed
 * these cards" and "this is the order the tracer emitted them in" is invisible in
 * the document itself — both are just `view.node_positions`.
 *
 * A sidecar whose ids have all gone is NOT a restored layout. The class was
 * rewritten under it, so what is on screen is a graph nobody has arranged, and
 * treating it as arranged is what leaves the user with crossing edges.
 */
export function mergeSavedPositions(
  document: unknown,
  saved: { graph_id?: string; node_positions?: Record<string, unknown> } | undefined,
  graphId: string,
): { document: unknown; restored: boolean } {
  if (!document || typeof document !== 'object' || !saved) {
    return { document, restored: false };
  }
  if (saved.graph_id && saved.graph_id !== graphId) {
    // The file was renamed or the class was; the old layout is not this graph's.
    return { document, restored: false };
  }
  const positions = saved.node_positions;
  if (!positions) {
    return { document, restored: false };
  }
  const copy = document as { view?: { node_positions?: Record<string, unknown> } };
  const view = copy.view ?? {};
  // Only ids the projection still has: a position for a node that no longer
  // exists is not a position, it is a leak.
  const current = view.node_positions ?? {};
  const merged: Record<string, unknown> = { ...current };
  let restored = false;
  for (const [nodeId, position] of Object.entries(positions)) {
    if (nodeId in current) {
      merged[nodeId] = position;
      restored = true;
    }
  }
  return { document: { ...copy, view: { ...view, node_positions: merged } }, restored };
}
