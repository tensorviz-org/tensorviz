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
import { BOUNDARY_ROW_COMPACT, BOUNDARY_ROW_FULL, compactBoundaryLabel, moduleBoundaryRows,
  moduleEvidenceLabel, moduleParameterRows } from '../services/modulePresentation';
import { ownGraphIssues, nodeFailureLines } from './graphIssues';
import { operationTensorRows } from './operationTensorData';
/**
 * The rendered height of a layer card, derived from the same data the component
 * renders. The shape's props.h used to be a hardcoded 120 while the card is
 * auto-height and grows with its parameter rows — so tldraw's selection box, its
 * resize handles and getGeometry()'s hit area only covered the TOP of a tall card
 * ("the resize box comes only in the upper half"). Both must agree.
 *
 * The constants below were MEASURED off the rendered DOM rather than estimated:
 * .ui2-node-header 50.8 (the title+subtitle stack, not the 22px icon, sets it),
 * a 1px border top and bottom, .ui2-node-body 8px padding each side with a 7px
 * row-gap between the param list and the shapes block, .ui2-node-row and
 * .ui2-node-shape-row 16.5 each with a 4px gap, and .ui2-node-shapes adding 6px
 * padding over a 1px top border. A 2-param, 2-shape-row card computes to 157,
 * which is what it measures.
 */
/**
 * Runtime statistics ride in `parameters` under `_`-prefixed keys (task.md §6.1)
 * but are NOT parameter rows: the card renders each as one line in the shapes
 * block, beside In and Out. Counting them in both places — or in neither —
 * desynchronises this height from the DOM, and this file is what tldraw's
 * selection box, resize handles and hit area are computed from.
 */
// An assignment keeps its draft width once accepted, and the width is what keeps
// its exact target (`sets layer3[0].relu · conditional`) readable.
export const layerCardWidth = (data: { review?: unknown; sourceAssignment?: unknown }): number => data.review || data.sourceAssignment ? 280 : 200;

/** Layout and export must reserve the same footprint as the live card. */
export function nodeCardSize(node: ModelNode): { w: number; h: number } {
  if (node.type === 'moduleNode') return {
    w: typeof node.style?.width === 'number' ? node.style.width : 250,
    h: typeof node.style?.height === 'number' ? node.style.height : moduleCardHeight(node.data),
  };
  return {
    w: layerCardWidth(node.data),
    h: node.type === 'inputNode' || node.type === 'outputNode'
      ? ioCardHeight(node.data) : layerCardHeight(node.data),
  };
}

/** An assignment's "Set in source · not run" row: live and unchanged-context cards.
 * A changed draft card says added/modified/removed in its badge instead. */
export const assignmentEvidenceShown = (data: { sourceAssignment?: unknown; review?: ModelNodeData['review'] }): boolean =>
  Boolean(data.sourceAssignment) && (!data.review || data.review.status === 'unchanged');

export const STAT_KEYS =['_act_stats', '_grad_stats'] as const;

/** Module tiles reserve only their visible content, including every boundary row. */
export function moduleCardHeight(data: ModelNodeData): number {
  if (data.capturePassThrough) return 88;
  const boundary = moduleBoundaryRows(data);
  const rowsHeight = boundary.reduce((sum, row) => sum + (compactBoundaryLabel(row.label) ? BOUNDARY_ROW_COMPACT : BOUNDARY_ROW_FULL), 0);
  const captured = Boolean(data.captureRef);
  // 56px header, 2px border, 10px + 12px body padding. Each section below adds its
  // height plus the body's 8px gap above it (6px above the source line); the first
  // section has no gap above it. Sections are listed in their DOM order.
  const issues = data.moduleDiagnostics ?? ownGraphIssues(data.moduleInstanceId || '', data);
  const status = data.moduleDiagnosticStatus || data.moduleStatus || 'ok';
  const paramRows = moduleParameterRows(data);
  const sections: Array<[height: number, gap: number]> = [];
  if (data.debugChanges?.length && data.debugEvidence !== 'previous') sections.push([29, 8]);
  // Constructor-argument grid: 20px per two-cell row.
  if (paramRows) sections.push([paramRows * 20, 8]);
  // Each boundary row carries its own 4px trailing gap; the last one does not show.
  if (boundary.length) sections.push([rowsHeight - 4, 8]);
  // A source module always draws its pill row (layer count, repeat and status); a
  // captured tile draws one only for evidence or a problem, and repeats separately.
  if (!captured || moduleEvidenceLabel(data) !== '' || issues.length || status !== 'ok') sections.push([27, 8]);
  if (captured && (data.repeatCount || 0) > 1) sections.push([21, 8]);
  if (data.sourceReplacement) sections.push([data.sourceReplacement.coverage === 'partial' ? 28 : 14, 8]);
  if (data.sourceLocation) sections.push([18, 6]);
  if (data.parameters?._defined_in) sections.push([15, 8]);
  if (data.moduleEntryNodeId && data.moduleExitNodeId) sections.push([14, 8]);
  if (data.modulePreviewNodes?.length) sections.push([42, 8]);
  const content = sections.reduce((sum, [height, gap], index) => sum + height + (index ? gap : 0), 0);
  return portCardHeight(data, 80 + content);
}

export function layerCardHeight(data: {
  layerType?: { id?: string };
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
}): number {
  if (data.operationGroup && data.operationGroup.kind !== 'slice') return portCardHeight(data, 166);
  const operationRows = operationTensorRows(data);
  if (operationRows !== undefined) {
    // 51px header, borders/body padding, 21px tensor rows with 6px gaps.
    let height = 77 + Math.max(21, operationRows.length * 27 - 6);
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
  const HEADER = 51;
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


/** Input/Output cards render a fixed two rows (Shape, Name). */
export const IO_CARD_HEIGHT = 106;

/**
 * The same card, plus the third row the Input draws when nobody chose its tensor.
 *
 * Not folded into the constant: an Output card and a declared Input still render
 * exactly two rows, and a box taller than its own content is the same desync in
 * the other direction — tldraw's selection box, resize handles and hit area are
 * all computed from this number.
 */
export function ioCardHeight(data: { parameters?: Record<string, unknown>; capturePorts?: CapturePort[]; captureDisplayPorts?: CapturePort[] } | undefined): number {
  // One more `.ui2-node-shape-row`: 16.5 over the 4px gap from the row above it.
  const base = data?.capturePorts ? 80 + data.capturePorts.length * 40
    : inputProvenanceNote(data?.parameters) ? IO_CARD_HEIGHT + 21 : IO_CARD_HEIGHT;
  return portCardHeight(data, base);
}
