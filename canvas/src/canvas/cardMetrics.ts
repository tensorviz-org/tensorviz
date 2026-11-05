/**
 * Card geometry shared by three layers that must agree on one height: the shape
 * box (useTldrawBridge), the module child layout (services/canvasLayout) and the
 * rendered card (LayerShapeUtil).
 *
 * Deliberately free of tldraw imports. canvasLayout is exercised by Jest suites
 * that cannot transform tldraw's ESM-only dependency, so importing this from
 * LayerShapeUtil.tsx instead broke four test suites at once.
 */
import { customCardModel } from './customCard';
import { cardParams, inputProvenanceNote } from '../ide/cardParams';
import { opSummary } from './opSummary';
import { portCardHeight } from './capturePorts';
import type { CapturePort, ModelNode, ModelNodeData } from '../types';
import { moduleBoundaryLayout, moduleContents, moduleEvidenceLabel, moduleParameterRows } from '../services/modulePresentation';
import { ownGraphIssues, nodeFailureLines } from './graphIssues';
import { operationTensorRows } from './operationTensorData';
import { nodePresentation } from './nodePresentation';
import { cardHeaderLayout, wrapCardText } from './cardIdentity';
import { tensorShapes } from './graphSemantics';
import { operationGroupCard } from '../services/operationGroups';
import { boundaryTextLayout, definitionLinkLayout, ioValueLayout, moduleRouteLayout, sourceLinkLayout } from './cardContentLayout';
import { portLabel, portType } from './capturePorts';
import { displayInputShape } from '../utils/inputShape';
import { tensorSummaryData } from './tensorSummaryData';
import { tensorRowLayout } from './cardContentLayout';
/** Ordinary cards reserve wrapped identity and visible tensor/status content.
 * Parameters, source excerpts and runtime statistics belong in selection details.
 * Review/assignment cards retain the measured row geometry below for exact changes.
 * Neither camera zoom nor generated explanations participate in card sizing.
 */
// An assignment keeps its draft width once accepted, and the width is what keeps
// its exact target (`sets layer3[0].relu · conditional`) readable.
export const layerCardWidth = (data: Partial<ModelNodeData>): number => {
  if (data.review || data.sourceAssignment) return 280;
  const title = customCardModel(data.layerType?.id, data.parameters)?.title || data.layerType?.displayName || data.layerType?.name || '';
  return Math.min(320, 200 + Math.max(0, title.length - 24) * 6);
};

export const moduleCardWidth = (data: ModelNodeData): number => {
  const title = String(data.moduleTemplateId || data.parameters?.name || data.parameters?._class || data.layerType?.name || 'Module').split('::').pop()!;
  return Math.min(340, 250 + Math.max(0, title.length - 24) * 6);
};

/** Layout and export must reserve the same footprint as the live card. */
export function nodeCardSize(node: ModelNode, width?: number): { w: number; h: number } {
  if (node.type === 'moduleNode') {
    const w = width ?? (typeof node.style?.width === 'number' ? node.style.width : moduleCardWidth(node.data));
    if (node.data.expanded && typeof node.style?.height === 'number') return { w, h: node.style.height };
    const measured = moduleCardHeight(node.data, w);
    return { w, h: typeof node.style?.height === 'number'
      ? Math.max(node.style.height, measured) : measured };
  }
  const w = width ?? layerCardWidth(node.data);
  return {
    w,
    h: node.type === 'inputNode' || node.type === 'outputNode'
      ? ioCardHeight(node.data, node.type === 'outputNode' ? 'output' : 'input', w) : layerCardHeight(node.data, w),
  };
}

/** An assignment's "Set in source · not run" row: live and unchanged-context cards.
 * A changed draft card says added/modified/removed in its badge instead. */
export const assignmentEvidenceShown = (data: { sourceAssignment?: unknown; review?: ModelNodeData['review'] }): boolean =>
  Boolean(data.sourceAssignment) && (!data.review || data.review.status === 'unchanged');

export const STAT_KEYS =['_act_stats', '_grad_stats'] as const;

/** Technical identity stays stable; generated annotations are measured separately. */
export function layerCardHeader(data: Partial<ModelNodeData>, width = layerCardWidth(data)) {
  const normalized = operationGroupCard({ ...data, parameters: data.parameters ?? {}, layerType: {
    id: '', name: '', displayName: '', category: 'misc', ...data.layerType,
  } } as ModelNodeData);
  const identity = nodePresentation(normalized, '');
  const subtitle = data.sourceAssignment
    ? `sets ${data.sourceAssignment.target}${data.sourceAssignment.conditional ? ' · conditional' : ''}` : identity.subtitle;
  return { ...identity, subtitle, ...cardHeaderLayout(identity.title, subtitle, width) };
}

export const layerSummaryShown = (data: Partial<ModelNodeData>): boolean => Boolean(data.inputShape || data.outputShape || data.error
  || operationTensorRows(data) !== undefined
  || (/^(?:linear|conv[123]d)$/.test(data.layerType?.id ?? '')
    && typeof data.parameters?.[data.layerType?.id === 'linear' ? 'in_features' : 'in_channels'] === 'number'
    && typeof data.parameters?.[data.layerType?.id === 'linear' ? 'out_features' : 'out_channels'] === 'number'));

export function moduleCardHeader(data: ModelNodeData, width = moduleCardWidth(data)) {
  const title = String(data.moduleTemplateId || data.parameters?.name || data.parameters?._class || data.layerType?.name || 'Module').split('::').pop()!;
  const subtitle = (data.sourceReplacement ? data.captureLabel : data.review?.summary) ?? data.captureLabel ?? data.moduleInstanceId;
  const layout = cardHeaderLayout(title, subtitle, width, moduleContents(data).canExpand);
  return { title, subtitle, ...layout, height: Math.max(56, layout.height) };
}

/** Expanded frames reserve their author/source row before positioning children. */
export function moduleExpandedHeaderHeight(data: ModelNodeData, width: number): number {
  return moduleCardHeader(data, width).height + (typeof data.parameters?._defined_in === 'string'
    ? definitionLinkLayout(data.parameters._defined_in, width).height : 0);
}

/** Module tiles reserve only their visible content, including every boundary row. */
export function moduleCardHeight(data: ModelNodeData, width = moduleCardWidth(data)): number {
  const header = moduleCardHeader(data, width);
  if (data.capturePassThrough) return header.height + 32;
  const boundary = moduleBoundaryLayout(data, width);
  const rowsHeight = boundary.reduce((sum, row) => sum + row.height + 4, 0);
  // 56px header, 2px border, 10px + 12px body padding. Each section below adds its
  // height plus the body's 8px gap above it (6px above the source line); the first
  // section has no gap above it. Sections are listed in their DOM order.
  const issues = data.moduleDiagnostics ?? ownGraphIssues(data.moduleInstanceId || '', data);
  const status = data.moduleDiagnosticStatus || data.moduleStatus || 'ok';
  const paramRows = data.review ? moduleParameterRows(data) : 0;
  const sections: Array<[height: number, gap: number]> = [];
  if (data.debugChanges?.length && data.debugEvidence !== 'previous') sections.push([29, 8]);
  // Constructor-argument grid: 20px per two-cell row.
  if (paramRows) sections.push([paramRows * 20, 8]);
  // Each boundary row carries its own 4px trailing gap; the last one does not show.
  if (boundary.length) sections.push([rowsHeight - 4, 8]);
  const evidence = moduleEvidenceLabel(data);
  // Empty source cards carry a quiet header icon, not an empty body row.
  if (moduleContents(data).label || (data.repeatCount || 0) > 1 || evidence || issues.length || status !== 'ok') sections.push([27, 8]);
  if (data.sourceReplacement) sections.push([data.sourceReplacement.coverage === 'partial' ? 28 : 14, 8]);
  if (data.sourceLocation) sections.push([sourceLinkLayout(data.sourceLocation, width).height, 6]);
  if (data.parameters?._defined_in) sections.push([definitionLinkLayout(String(data.parameters._defined_in), width).height, 8]);
  if (data.moduleEntryNodeId && data.moduleExitNodeId) sections.push([moduleRouteLayout(data.moduleEntryNodeId, data.moduleExitNodeId, width).height, 8]);
  if (data.modulePreviewNodes?.length) sections.push([42, 8]);
  const content = sections.reduce((sum, [height, gap], index) => sum + height + (index ? gap : 0), 0);
  return portCardHeight(data, header.height + (sections.length ? 24 + content : 2));
}

export function layerCardHeight(rawData: {
  layerType?: Partial<ModelNodeData['layerType']>;
  parameters?: Record<string, unknown>;
  inputShape?: unknown;
  outputShape?: unknown;
  error?: unknown;
  captureDiagnostic?: ModelNodeData['captureDiagnostic'];
  captureRef?: ModelNodeData['captureRef'];
  debugChanges?: string[];
  review?: ModelNodeData['review'];
  sourceLocation?: unknown;
  capturePorts?: CapturePort[];
  captureDisplayPorts?: CapturePort[];
  operationGroup?: ModelNodeData['operationGroup'];
  sourceAssignment?: ModelNodeData['sourceAssignment'];
  captureLabel?: string;
  shapeEvidence?: ModelNodeData['shapeEvidence'];
  debugEvidence?: ModelNodeData['debugEvidence'];
  outputTensor?: ModelNodeData['outputTensor'];
}, width?: number): number {
  const display = operationGroupCard(rawData as ModelNodeData);
  const data = { ...display, ...tensorShapes({ ...display, parameters: display.parameters ?? {} } as ModelNodeData) };
  const header = layerCardHeader(data as Partial<ModelNodeData>, width);
  if (!data.review && !data.sourceAssignment) {
    const summary = layerSummaryShown(data as Partial<ModelNodeData>);
    let height = header.height + 2;
    if (summary) height += tensorSummaryData(data as ModelNodeData, width ?? layerCardWidth(data)).height;
    if (data.debugChanges?.length) height += 38;
    if (data.operationGroup && data.operationGroup.kind !== 'slice') height += 30;
    if (customCardModel(data.layerType?.id, data.parameters)?.projectExpandable) height += 28;
    return portCardHeight(data, Math.ceil(height));
  }
  if (data.operationGroup && data.operationGroup.kind !== 'slice') return portCardHeight(data, 166 + header.height - 51);
  const operationRows = operationTensorRows(data);
  if (operationRows !== undefined) {
    // 51px header, borders/body padding, 21px tensor rows with 6px gaps.
    const rowsHeight = operationRows.reduce((sum, row) => sum + tensorRowLayout(row.label, row.value, width ?? layerCardWidth(data)).height, 0)
      + Math.max(0, operationRows.length - 1) * 6;
    let height = header.height + 26 + Math.max(21, rowsHeight);
    if (data.debugChanges?.length) height += 38;
    if (data.error) height += 7 + Math.max(30, 12 + nodeFailureLines({ ...data, error: String(data.error) }).length * 16);
    return portCardHeight(data, Math.max(120, height));
  }
  /**
   * A `custom_code` / `python` card is a different card in the same box
   * (task.md M6): its provenance keys are not parameter rows, it adds a snippet
   * row, and it adds a "defined in" row — and an expand row when the IDE can
   * actually run the worker. Omitting this entry is what makes custom nodes
   * overlap, because every other layer here would then be measured against a
   * card the component does not draw.
   */
  const custom = customCardModel(data.layerType?.id, data.parameters);
  const HEADER = header.height;
  const BORDERS = 2;
  const BODY_PAD = 16;
  const BODY_GAP = 7;
  const ROW = 16.5;
  const ROW_GAP = 4;
  const SHAPES_LEAD = 7; // 6px padding over a 1px top border

  // Same rule as the card itself, from the same function: a height computed from a
  // different filter draws a box that does not fit its own rows.
  const card = cardParams(data.parameters);
  const params = Array.from(new Set([...card.entries.map(([key]) => key), ...Object.keys(data.review?.params ?? {}).filter(key => !key.startsWith('_'))]));
  const shown = Math.min(params.length, 6);
  // the "+N more params" hint is a row, and so is the custom card's snippet, and so is
  // the container summary ("4 × Linear") that stands in for a member list too big to draw
  const listRows =
    shown + (params.length > 6 ? 1 : 0) + (custom?.snippet ? 1 : 0) + (card.contents ? 1 : 0);
  const statRows = STAT_KEYS.filter((key) => data.parameters?.[key]).length;
  const shapeRows =
    (data.inputShape ? 1 : 0) +
    (data.outputShape ? 1 : 0) +
    statRows +
    (custom?.location && !data.sourceLocation ? 1 : 0) +
    (custom?.projectExpandable ? 1 : 0) +
    // The Validate row (step 5.9). `unsupported` draws nothing, so it measures
    // nothing: a height counted here and not rendered there is a selection box
    // taller than its card, which is the same desynchronisation this file's whole
    // comment is about, pointing the other way.
    (custom?.validatable && custom.validatable.state !== 'unsupported' ? 1 : 0);

  const block = (rows: number) => (rows > 0 ? rows * ROW + (rows - 1) * ROW_GAP : 0);

  let h = HEADER + BORDERS + BODY_PAD;
  if (data.sourceLocation) h += ROW + BODY_GAP;
  if (assignmentEvidenceShown(data)) h += ROW + BODY_GAP;
  // The expression chip precedes the parameter list (26px plus its body gap).
  // It used to grow outside the shape, clipping the final measurement row.
  if (opSummary(data.layerType?.id ?? '', data.parameters)) h += 33;
  h += block(listRows);
  if (shapeRows > 0) {
    if (listRows > 0) h += BODY_GAP;
    h += SHAPES_LEAD + block(shapeRows);
  }
  if (data.error) {
    h += BODY_GAP + Math.max(30, 12 + nodeFailureLines({ ...data, error: String(data.error) }).length * 16);
  }
  if (data.debugChanges?.length) h += 38;
  // The refusal's own sentence, which wraps to roughly two lines at card width.
  if (custom?.validatable?.state === 'needs' && custom.validatable.detail) {
    h += BODY_GAP + 34;
  }

  // Never shorter than the old constant, so a bare node keeps its familiar size.
  return portCardHeight(data, Math.max(120, Math.round(h)));
}


type IoCardData = Pick<Partial<ModelNodeData>, 'layerType' | 'parameters' | 'inputShape' | 'outputShape' | 'captureLabel' | 'capturePorts' | 'captureDisplayPorts' | 'error'>;

/** Root interfaces reserve only the rows they actually draw. An absent output
 *  has no body; an explicitly empty recorded interface keeps its evidence row. */
export function ioCardContent(data: IoCardData | undefined, kind: 'input' | 'output' = data?.layerType?.id === 'output' ? 'output' : 'input', width = 200) {
  const captured = data?.capturePorts !== undefined;
  const name = !captured && typeof data?.parameters?.name === 'string' && data.parameters.name.trim()
    ? data.parameters.name : undefined;
  const provenance = !captured && kind === 'input' ? inputProvenanceNote(data?.parameters) : undefined;
  const shape = !captured && (kind === 'input' || Array.isArray(data?.outputShape));
  const rowCount = captured ? Math.max(1, data?.capturePorts?.length ?? 0) : Number(shape) + Number(Boolean(name)) + Number(Boolean(provenance));
  const shapeValue = shape ? (kind === 'input' ? displayInputShape(data?.parameters?.shape ?? data?.inputShape) ?? 'Not recorded'
    : `(${data?.outputShape?.join(', ')})`) : undefined;
  const values = { ...(shapeValue ? { shape: ioValueLayout(shapeValue, width) } : {}),
    ...(name ? { name: ioValueLayout(name, width) } : {}),
    ...(provenance ? { provenance: ioValueLayout(provenance.label, width) } : {}) };
  const rowsHeight = captured && data?.capturePorts?.length ? data.capturePorts.reduce((sum, port, index) =>
    sum + boundaryTextLayout(portLabel(port, index), portType(port), width, false, true).height, 0) + (rowCount - 1) * 4
    : rowCount ? (captured ? 16.5 : Object.values(values).reduce((sum, row) => sum + row.height, 0)) + (rowCount - 1) * 4 : 0;
  const errorLines = data?.error ? wrapCardText(data.error, width - 54, 11) : [];
  const errorHeight = errorLines.length ? Math.max(30, 12 + errorLines.length * 16) : 0;
  const bodyHeight = rowsHeight + (errorHeight ? (rowCount ? 7 : 0) + errorHeight : 0);
  return { captured, name, provenance, shape, rowCount, values, errorLines, errorHeight, bodyHeight: bodyHeight ? bodyHeight + 16 : 0 };
}

/** A bare input's truthful unknown-shape row, with header and borders. */
export const IO_CARD_HEIGHT = Math.ceil(cardHeaderLayout('Input').height + 2 + 16 + 16.5);

export function ioCardHeight(data: IoCardData | undefined, kind: 'input' | 'output' = data?.layerType?.id === 'output' ? 'output' : 'input', width = 200): number {
  const title = kind === 'output' ? 'Output' : 'Input';
  return portCardHeight(data, Math.ceil(cardHeaderLayout(title, data?.captureLabel, width).height + 2 + ioCardContent(data, kind, width).bodyHeight));
}
