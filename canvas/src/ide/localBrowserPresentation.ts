/** Read-only presentation helpers for a multi-file local-browser projection.
 *
 * The backend owns resolution and its diagnostic codes. This module only turns
 * that structured answer into canvas metadata: node-scoped limitations remain
 * visible on the affected card while the full diagnostic stays available in
 * LocalBrowserApp's warning drawer. No source, graph semantics, or backend
 * state is changed here.
 */

import {
  IdeProjectionDiagnostic,
  IdeProjectionDiagnosticCode,
  ProjectionPayload,
} from './protocol.generated';
import { LocalBrowserPageState } from './localBrowserState';
import { GraphDocumentV2 } from '../types/graphDocument';
import { sourceAssignmentsOf } from './sourceAssignmentCards';
import { assignmentContext } from './assignmentContext';
import { failingLayerLabel } from './GraphRunNotice';

export function graphOpeningPresentation(page: LocalBrowserPageState) {
  const opening = page.opening, graph = page.projection;
  const visible = !!opening && opening.phase !== 'source_only';
  const captured = !!(opening && graph?.traced && graph.cls === opening.target.cls && graph.file === opening.target.file);
  const current = captured && graph!.sourceDigest === opening!.target.sourceDigest
    && graph!.revision === opening!.target.backendRevision;
  const available = !!(opening && graph && graph.cls === opening.target.cls && graph.file === opening.target.file);
  const environmentPending = !!opening?.environment && opening.environment.reason !== 'ready' && opening.phase !== 'cancelled';
  const compact = visible && available && !environmentPending && opening!.phase !== 'needs_setup'
    && (opening!.phase !== 'ready' || (current && opening!.hasCapture))
    && !(opening!.phase === 'failed' && opening!.failure?.kind === 'setup');
  return { visible, compact, waiting: visible && !available, retained: visible && captured && (!current || opening!.phase !== 'ready') };
}

export const DIAGNOSTIC_LABELS: Record<IdeProjectionDiagnosticCode, string> = {
  UNRESOLVED_IMPORT: 'Unresolved import',
  UNRESOLVED_DYNAMIC: 'Dynamic module',
  AMBIGUOUS_BINDING: 'Ambiguous binding',
  CYCLE_BOUNDARY: 'Cycle boundary',
  EXTERNAL_PACKAGE: 'External package',
  DEPENDENCY_SYNTAX_ERROR: 'Dependency syntax error',
  SOURCE_SET_TRUNCATED: 'Source set truncated',
  UNSUPPORTED_IMPORT_FORM: 'Unsupported import',
  UNSUPPORTED_INHERITANCE: 'Unsupported inheritance',
  RESOURCE_LIMIT: 'Resource limit',
};

/** The one sentence about how true the canvas is right now, with its tone.
 *  The header strip and the Ask rail both wear it, so both are computed here:
 *  two surfaces can never disagree about what truth the canvas carries. */
export function localBrowserSyncLabel(
  page: LocalBrowserPageState,
): { tone: 'ok' | 'warn' | 'muted'; text: string } | null {
  // An edit after a capture cancels the opening, but a current source graph is
  // on the canvas: say that it was not run rather than only that source changed.
  const sourceShown = page.opening?.phase === 'cancelled'
    && page.syncState === 'source-only' && !!page.projection && !page.projection.traced && !page.greyed;
  if (page.opening && !sourceShown && ['resolving', 'saving', 'capturing', 'failed', 'cancelled'].includes(page.opening.phase)) {
    if (page.opening.phase === 'failed') {
      const layer = failingLayerLabel(page.projection);
      const detail = page.opening.detail?.trim();
      return { tone: 'warn', text: layer ? `Capture stopped at ${layer}${detail ? ` · ${detail}` : ''}` : (detail || 'Capture stopped · Saved inputs unchanged') };
    }
    return { tone: 'muted', text: page.opening.detail };
  }
  if (page.syncState === 'stale' && page.reason !== null) {
    return { tone: 'warn', text: `${page.projection ? 'Stale' : 'Graph unavailable'} — ${page.reason}` };
  }
  if (page.syncState === 'stale') {
    return { tone: 'warn', text: page.projection ? 'Stale — showing the last available graph' : 'Graph unavailable — reopen from VS Code to retry' };
  }
  if (page.syncState === 'runtime-traced' && page.projection?.traceError) {
    return { tone: 'warn', text: 'Forward failed — partial shapes measured' };
  }
  if (page.syncState === 'runtime-traced') {
    const capture = (page.projection?.document as Partial<GraphDocumentV2> | undefined)
      ?.semantic?.metadata?.capture as { backend?: string; status?: string; omissions?: string[] } | undefined;
    if (capture?.status === 'partial') {
      if (capture.backend === 'jaxpr') return { tone: 'warn', text: 'Partial JAX program — shapes inferred' };
      return { tone: 'warn', text: capture.omissions?.includes('byte_limit')
        ? 'Partial capture — recording size limit reached'
        : 'Partial capture — some execution details are missing' };
    }
    if (capture?.backend === 'jaxpr') {
      return { tone: 'ok', text: 'JAX traced program — shapes inferred' };
    }
    return { tone: 'ok', text: 'Traced — shapes measured' };
  }
  if (page.syncState === 'source-only') {
    const before = sourceAssignmentsOf(page.projection?.debug?.previousDocument);
    const after = sourceAssignmentsOf(page.projection?.document);
    if (page.projection?.debug?.previousDocument
      && [...Object.keys(before), ...Object.keys(after)].some(key => before[key]?.digest !== after[key]?.digest)) {
      const placed = assignmentContext(page.projection?.debug?.previousDocument, page.projection?.document);
      return { tone: placed ? 'muted' : 'warn', text: placed ? 'Source changes · Not run · Previous run context'
        : 'Source edits not located in graph · Validate to update' };
    }
    return { tone: 'muted', text: 'Not run — shapes read from the source' };
  }
  if (page.activeTarget !== null) {
    return { tone: 'muted', text: `Projecting ${page.activeTarget.cls}…` };
  }
  return null;
}

const severityRank: Record<IdeProjectionDiagnostic['severity'], number> = {
  error: 0,
  warning: 1,
  info: 2,
};

/** Stable ordering keeps the warning list from jumping between projections. */
export function orderedDiagnostics(
  diagnostics: readonly IdeProjectionDiagnostic[] | undefined,
): IdeProjectionDiagnostic[] {
  return [...(diagnostics ?? [])].sort((left, right) =>
    severityRank[left.severity] - severityRank[right.severity]
      || left.file.localeCompare(right.file)
      || left.line - right.line
      || left.code.localeCompare(right.code)
      || (left.node ?? '').localeCompare(right.node ?? ''),
  );
}

export function diagnosticSummary(diagnostics: readonly IdeProjectionDiagnostic[] | undefined): string {
  const items = diagnostics ?? [];
  const errors = items.filter((item) => item.severity === 'error').length;
  const warnings = items.filter((item) => item.severity === 'warning').length;
  const information = items.length - errors - warnings;
  const parts = [
    errors ? `${errors} error${errors === 1 ? '' : 's'}` : '',
    warnings ? `${warnings} warning${warnings === 1 ? '' : 's'}` : '',
    information ? `${information} note${information === 1 ? '' : 's'}` : '',
  ].filter(Boolean);
  return parts.join(' · ');
}

type ResolutionPresentation = {
  resolutionCode: IdeProjectionDiagnosticCode;
  resolutionSeverity: IdeProjectionDiagnostic['severity'];
  resolutionDetail: string;
};

function strongestByNode(diagnostics: readonly IdeProjectionDiagnostic[]): Map<string, IdeProjectionDiagnostic> {
  const result = new Map<string, IdeProjectionDiagnostic>();
  for (const diagnostic of orderedDiagnostics(diagnostics)) {
    if (diagnostic.node && !result.has(diagnostic.node)) {
      result.set(diagnostic.node, diagnostic);
    }
  }
  return result;
}

/** Attach each node's strongest resolution diagnostic to its canvas data.
 * The semantic node and view layout are copied, never mutated in place. */
export function projectionForCanvas(projection: ProjectionPayload): ProjectionPayload {
  const diagnostics = projection.diagnostics ?? [];

  const document = projection.document as Partial<GraphDocumentV2> | null;
  const nodes = document?.semantic?.nodes;
  if (!Array.isArray(nodes) || !document?.view) return projection;

  const byNode = strongestByNode(diagnostics);
  const sourceLocations = new Map<string, { file: string; line: number }>();
  for (const [id, value] of Object.entries(projection.spanmap ?? {})) {
    const span = value as { file?: unknown; start?: unknown };
    if (typeof span?.file === 'string' && Array.isArray(span.start) && Number.isSafeInteger(span.start[0]) && span.start[0] > 0) {
      sourceLocations.set(id, { file: span.file, line: span.start[0] });
    }
  }
  if (byNode.size === 0 && sourceLocations.size === 0) return projection;

  const presentation = (diagnostic: IdeProjectionDiagnostic): ResolutionPresentation => ({
    resolutionCode: diagnostic.code,
    resolutionSeverity: diagnostic.severity,
    resolutionDetail: diagnostic.detail,
  });
  const semantic = {
    ...document.semantic,
    nodes: nodes.map((node) => {
      const diagnostic = byNode.get(node.node_id);
      const sourceLocation = sourceLocations.get(node.node_id);
      return diagnostic || sourceLocation
        ? { ...node, metadata: { ...(node.metadata ?? {}), ...(diagnostic ? presentation(diagnostic) : {}), ...(sourceLocation ? { sourceLocation } : {}) } }
        : node;
    }),
  };
  const layout = document.view.layout ?? {};
  const nodeData = { ...(layout.node_data ?? {}) };
  byNode.forEach((diagnostic, nodeId) => {
    nodeData[nodeId] = {
      ...(nodeData[nodeId] ?? {}),
      ...presentation(diagnostic),
      ...(diagnostic.severity === 'error'
        ? { moduleStatus: 'error' }
        : diagnostic.severity === 'warning'
          ? { moduleStatus: 'warning' }
          : {}),
    };
  });

  return {
    ...projection,
    document: {
      ...document,
      semantic,
      view: {
        ...document.view,
        layout: { ...layout, node_data: nodeData },
      },
    },
  };
}
