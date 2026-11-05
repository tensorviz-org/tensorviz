/**
 * graphSvg — serialize a reconciled graph (the same nodes/edges the tldraw
 * canvas draws) into ONE standalone SVG document.
 *
 * This is the cheap answer to the memory problem: instead of mounting a React
 * card per node (700 cards + 187 frames, all kept mounted because tldraw only
 * CSS-culls), the whole diagram becomes a single `<img>` whose vector source is
 * this string. Nothing here imports React or tldraw, so it is cheap and testable.
 *
 * Cards reuse the interactive card's compact identity and tensor evidence.
 * Source parameters stay in selection details rather than enlarging the graph.
 *
 * Geometry mirrors the bridge/layout: a module child's position is relative to
 * its module, so absolute positions are resolved first; card heights come from the
 * same `layerCardHeight` / `ioCardHeight` the shape box uses, and rows stop before
 * the card's own reserved height so nothing overflows.
 */
import { nodeCardSize, layerCardHeader, layerSummaryShown, moduleCardHeader, STAT_KEYS } from '../canvas/cardMetrics';
import { cardHeaderLayout } from '../canvas/cardIdentity';
import { cardParams } from '../ide/cardParams';
import { opSummary } from '../canvas/opSummary';
import { customCardModel } from '../canvas/customCard';
import { isFlowEdge } from './canvasLayout';
import { ModelEdge, ModelNode } from '../types';
import { operationGroupCard } from './operationGroups';
import { nodeEvidenceLabel } from '../canvas/nodePresentation';
import { tensorShapes } from '../canvas/graphSemantics';
import { operationTensorRows } from '../canvas/operationTensorData';
import { moduleBoundaryRows, moduleContents, moduleEvidenceLabel } from './modulePresentation';
import { isFailingPort, nodeFailureLines } from '../canvas/graphIssues';
import { AnchorSide, LayoutDirection, NodePortSides, portAnchorPoint } from '../canvas/edgeAnchors';
import { displayedPorts, portLabel, portType } from '../canvas/capturePorts';
import { displayInputShape } from '../utils/inputShape';
import { inputProvenanceNote } from '../ide/cardParams';
import { createEdgeRouter } from '../canvas/edgeRouting';
import { edgeControlPoints } from '../canvas/edgeCurve';
import { graphReviewLabel } from '../ide/graphReviewLabels';
import { BlockKind, classifyNode, isBlockFamily } from '../canvas/blockFamily';
import { blockGlyphSvg } from '../canvas/blockGlyphs';

export interface GraphSvgOptions {
  padding?: number;
  background?: string;
  theme?: 'light' | 'dark';
  /** The direction the nodes were laid out with; ports and skip lanes follow it. */
  direction?: LayoutDirection;
}

const ROW_PITCH = 20.5;

const INK = {
  background: '#171817',
  moduleFill: '#1c1d1c',
  moduleStroke: '#383b38',
  moduleHeader: '#242624',
  cardFill: '#2a2c2a',
  cardStroke: '#3f433f',
  edge: '#636b64',
  text: '#e3e8e2',
  muted: '#a4ada5',
  exprFill: 'rgba(227, 232, 226, 0.05)',
  // The card's glyph tile (--tv-row over the card) and its family tint (--tv-glyph-tint).
  glyphTile: '#303230', glyphTint: 0.16,
  error: '#ff9298',
  warning: '#f5c778',
  added: '#83dca3',
  modified: '#f5c778',
  rewired: '#8bbcff',
  removed: '#ff9298',
  residual: '#636b64',
  portIn: '#8f998f', portOut: '#d5dcd4',
};

type Palette = typeof INK;
const PAPER: Palette = {
  background: '#f3f2ee', moduleFill: '#f8f8fb', moduleStroke: '#d5d5db', moduleHeader: '#ffffff',
  cardFill: '#ffffff', cardStroke: '#d9d9df', edge: '#b2b5bd', text: '#16171b', muted: '#555a66',
  exprFill: 'rgba(20, 22, 30, 0.04)', glyphTile: '#f6f6f4', glyphTint: 0, error: '#c62f45', warning: '#8f5600', added: '#1a8a4a', modified: '#b06a00',
  rewired: '#2563d9', removed: '#c62f45', residual: '#b2b5bd', portIn: '#7f8490', portOut: '#17181c',
};

/** Mirrors the --tv-family-* tokens in canvas.tokens.css; an <img> cannot read CSS. */
const FAMILY_INK: Record<string, string> = {
  convolution: '#a8b9c7', linear: '#b5bacb', normalization: '#aabdae', activation: '#c6bda4', pooling: '#baafc5',
  dropout: '#c2adb5', recurrent: '#b5a5c1', attention: '#aba7c7', embedding: '#beaf96', padding: '#b6b8a5',
  resampling: '#a5bbb6', shape: '#aab8be', distance: '#b9a9ae', loss: '#c2aa9f', arithmetic: '#bfb1a4',
  io: '#c9cfc8', neutral: '#848d85',
};
const FAMILY_PAPER: Record<string, string> = {
  convolution: '#6c91af', linear: '#808bb1', normalization: '#6d9775', activation: '#9f8b54', pooling: '#9a84b0',
  dropout: '#ab8291', recurrent: '#9e82b4', attention: '#8e87be', embedding: '#a78956', padding: '#8b8f68',
  resampling: '#67968b', shape: '#73919e', distance: '#a3858e', loss: '#b1836e', arithmetic: '#a5886d',
  io: '#555a66', neutral: '#83888f',
};
export const svgFamilyPalettes = { dark: FAMILY_INK, light: FAMILY_PAPER };
const familyToken = (kind: BlockKind): string => isBlockFamily(kind) ? kind : kind === 'input' || kind === 'output' ? 'io' : 'neutral';
const mixHex = (color: string, share: number, base: string): string => {
  const channel = (hex: string, index: number) => parseInt(hex.slice(1 + index * 2, 3 + index * 2), 16);
  return `#${[0, 1, 2].map(index => Math.round(channel(color, index) * share + channel(base, index) * (1 - share))
    .toString(16).padStart(2, '0')).join('')}`;
};

const FONT = 'font-family="Inter, ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif"';
const MONO = 'font-family="JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"';

const escapeXml = (value: string): string => value
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&apos;');

const clip = (value: string, length: number): string =>
  value.length > length ? `${value.slice(0, length - 1)}…` : value;

const text = (x: number, y: number, value: string, fill: string, size: number, weight = 400, mono = false): string =>
  `<text x="${x}" y="${y}" fill="${fill}" ${mono ? MONO : FONT} font-size="${size}" font-weight="${weight}">${escapeXml(value)}</text>`;

/** Explicit shared lines preserve the entire name in both paint and textContent. */
function identityText(box: Box, layout: ReturnType<typeof cardHeaderLayout>, COLOR: Palette): string {
  const x = box.x + 52;
  const lines = (values: string[], y: number, step: number, size: number, mono = false) => values.length
    ? `<text data-card-${mono ? 'subtitle' : 'title'}="true" fill="${mono ? COLOR.muted : COLOR.text}" ${mono ? MONO : FONT} font-size="${size}" font-weight="${mono ? 400 : 600}">${values.map((line, index) => `<tspan x="${x}" y="${y + index * step}">${escapeXml(line)}</tspan>`).join('')}</text>` : '';
  return lines(layout.titleLines, box.y + 24, 18, 14)
    + lines(layout.subtitleLines, box.y + 24 + (layout.titleLines.length - 1) * 18 + 14, 14, 11, true);
}

/** Keep evidence beside the identity without consuming its wrapped text column. */
function sourceEvidence(node: ModelNode, box: Box, COLOR: Palette): string {
  const previous = node.data.debugEvidence === 'previous';
  if (!previous && node.data.debugEvidence !== 'source') return '';
  const label = nodeEvidenceLabel(node.data);
  const x = box.x + box.w - 23, y = box.y + 14;
  const path = previous
    ? 'M2 5a5 5 0 1 1 0 4M2 1v4h4M7 3v4l2 1'
    : 'M3 1h5l3 3v8H3zM8 1v4h3M5 8h4M5 10h3';
  return `<g data-source-evidence="${previous ? 'previous' : 'source'}" role="img" aria-label="${escapeXml(label)}"><title>${escapeXml(label)}</title><path d="${path}" transform="translate(${x} ${y})" fill="none" stroke="${COLOR.muted}" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></g>`;
}

interface Box { x: number; y: number; w: number; h: number }

function absolutePositions(nodes: ModelNode[]): Map<string, { x: number; y: number }> {
  const byId = new Map(nodes.map((node) => [node.id, node] as const));
  const cache = new Map<string, { x: number; y: number }>();
  const resolving = new Set<string>();
  const resolve = (node: ModelNode): { x: number; y: number } => {
    const cached = cache.get(node.id);
    if (cached) return cached;
    if (resolving.has(node.id)) return node.position;
    resolving.add(node.id);
    let result = node.position;
    if (node.parentNode) {
      const parent = byId.get(node.parentNode as string);
      if (parent) {
        const origin = resolve(parent);
        result = { x: origin.x + node.position.x, y: origin.y + node.position.y };
      }
    }
    resolving.delete(node.id);
    cache.set(node.id, result);
    return result;
  };
  nodes.forEach(resolve);
  return cache;
}

function boxOf(node: ModelNode, origin: { x: number; y: number }): Box {
  return { ...origin, ...nodeCardSize(node) };
}

type Tone = 'plain' | 'muted' | 'hint' | 'shape' | 'expr' | 'error' | 'warning';
interface Row { left: string; right?: string; tone: Tone; mono?: boolean }

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  return String(value);
};

const formatShape = (shape: unknown): string =>
  Array.isArray(shape) ? `(${shape.join(', ')})` : '';

function stats(value: unknown): { nan: number; inf: number; mean?: number; std?: number } | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const record = value as Record<string, unknown>;
  const nan = typeof record.nan === 'number' ? record.nan : 0;
  const inf = typeof record.inf === 'number' ? record.inf : 0;
  if (nan === 0 && inf === 0 && typeof record.mean !== 'number') return undefined;
  return { nan, inf, mean: typeof record.mean === 'number' ? record.mean : undefined,
    std: typeof record.std === 'number' ? record.std : undefined };
}

const statsText = (value: { nan: number; inf: number; mean?: number; std?: number }): string =>
  value.nan > 0 || value.inf > 0
    ? [value.nan > 0 ? `${value.nan} NaN` : '', value.inf > 0 ? `${value.inf} Inf` : ''].filter(Boolean).join(' · ')
    : `μ ${value.mean ?? 0} σ ${value.std ?? 0}`;

/** Explicit source-review cards retain their change evidence at every scale. */
function cardRows(node: ModelNode): Row[] {
  const data = operationGroupCard(node.data);
  const layerType = data.layerType || ({} as ModelNode['data']['layerType']);
  const custom = customCardModel(layerType?.id, data.parameters);
  const { entries, contents } = cardParams(data.parameters);
  const rows: Row[] = [];
  if (data.operationGroup && data.operationGroup.kind !== 'slice') rows.push({ left: 'Double-click to show operations', tone: 'hint' });
  const summary = opSummary(layerType?.id ?? '', data.parameters);
  if (summary) rows.push({ left: summary, tone: 'expr', mono: true });
  if (custom?.snippet) rows.push({ left: 'call', right: custom.snippet, tone: 'muted', mono: true });
  entries.slice(0, 6).forEach(([key, value]) => rows.push({ left: String(key), right: formatValue(value), tone: 'plain' }));
  if (entries.length > 6) rows.push({ left: `+${entries.length - 6} more params`, tone: 'hint' });
  if (contents) rows.push({ left: contents, tone: 'hint' });
  const act = stats(data.parameters?.[STAT_KEYS[0]]);
  const grad = stats(data.parameters?.[STAT_KEYS[1]]);
  if (data.inputShape) rows.push({ left: 'In', right: formatShape(data.inputShape), tone: 'shape' });
  if (data.outputShape) rows.push({ left: 'Out', right: formatShape(data.outputShape), tone: 'shape' });
  if (act) rows.push({ left: 'Act', right: statsText(act), tone: act.nan || act.inf ? 'warning' : 'shape' });
  if (grad) rows.push({ left: 'Grad', right: statsText(grad), tone: grad.nan || grad.inf ? 'warning' : 'shape' });
  if (custom?.location) rows.push({ left: 'defined in', right: custom.location, tone: 'muted' });
  if (custom?.validatable && custom.validatable.state !== 'unsupported') {
    rows.push({ left: custom.validatable.state === 'ready' ? 'Ready to validate' : 'Cannot validate', tone: 'muted' });
  }
  if (custom?.projectExpandable) rows.push({ left: 'Expand', tone: 'muted' });
  if (data.sourceAssignment && (!data.review || data.review.status === 'unchanged')) rows.push({ left: 'Set in source · not run', tone: 'muted' });
  if (data.debugChanges?.length) rows.push({ left: `Changed · ${data.debugChanges[0]}`, tone: 'warning' });
  for (const line of nodeFailureLines(data)) rows.push({ left: line, tone: 'error' });
  return rows;
}

/** Match the ordinary card's stable overview; unknown shapes never become zeros. */
function compactRows(node: ModelNode): Row[] {
  const data = operationGroupCard(node.data);
  const rows: Row[] = [];
  if (node.type === 'inputNode' || node.type === 'outputNode') {
    if (data.capturePorts) {
      data.capturePorts.forEach((port, index) => rows.push({ left: portLabel(port, index), right: portType(port), tone: 'shape' }));
      if (!data.capturePorts.length) rows.push({ left: 'No boundary value recorded', tone: 'muted' });
    } else {
      if (node.type === 'inputNode') rows.push({ left: 'Shape', right: displayInputShape(data.parameters.shape ?? data.inputShape) ?? 'Not recorded', tone: 'shape' });
      else if (data.outputShape) rows.push({ left: 'Shape', right: formatShape(data.outputShape), tone: 'shape' });
      if (data.parameters.name) rows.push({ left: 'Name', right: String(data.parameters.name), tone: 'muted' });
      const provenance = node.type === 'inputNode' && inputProvenanceNote(data.parameters);
      if (provenance) rows.push({ left: 'Source', right: provenance.label, tone: 'warning' });
    }
  } else {
    if (data.operationGroup && data.operationGroup.kind !== 'slice') rows.push({ left: `Show ${data.operationGroup.members.length} operations`, tone: 'hint' });
    const measured = { ...data, ...tensorShapes(data) };
    const operation = operationTensorRows(measured);
    if (operation !== undefined) {
      operation.forEach(row => rows.push({ left: row.label, right: row.value, tone: 'shape' }));
      if (!operation.length) rows.push({ left: 'No tensor shapes recorded', tone: 'muted' });
    } else if (measured.inputShape || measured.outputShape) {
      if (measured.inputShape) rows.push({ left: 'In', right: formatShape(measured.inputShape), tone: 'shape' });
      if (measured.outputShape) rows.push({ left: 'Out', right: formatShape(measured.outputShape), tone: 'shape' });
    } else if (layerSummaryShown(measured)) {
      const feature = data.layerType.id === 'linear';
      const before = data.parameters[feature ? 'in_features' : 'in_channels'];
      const after = data.parameters[feature ? 'out_features' : 'out_channels'];
      if (typeof before === 'number' && typeof after === 'number') rows.push({ left: feature ? 'Features' : 'Channels', right: `${before} → ${after}`, tone: 'shape' });
    }
    if (customCardModel(data.layerType.id, data.parameters)?.projectExpandable) rows.push({ left: 'Expand', tone: 'muted' });
    if (data.debugChanges?.length) rows.push({ left: `Changed · ${data.debugChanges[0]}`, tone: 'warning' });
  }
  for (const line of nodeFailureLines(data)) rows.push({ left: line, tone: 'error' });
  return rows;
}

function renderCardRows(box: Box, rows: Row[], COLOR: Palette, headerHeight: number): string {
  const parts: string[] = [];
  let y = box.y + headerHeight + 10;
  const limit = box.y + box.h - 4;
  for (const row of rows) {
    if (y + 14 > limit) break;
    if (row.tone === 'expr') {
      parts.push(`<rect x="${box.x + 10}" y="${y + 1}" width="${box.w - 20}" height="24" rx="5" fill="${COLOR.exprFill}"/>`);
      parts.push(text(box.x + 16, y + 17, clip(row.left, 30), COLOR.text, 10.5, 500, true));
      y += 33;
      continue;
    }
    const fill = row.tone === 'error' ? COLOR.error
      : row.tone === 'warning' ? COLOR.warning
        : row.tone === 'hint' ? COLOR.muted
          : row.tone === 'muted' ? COLOR.muted : COLOR.text;
    if (row.right !== undefined && row.right !== '') {
      parts.push(text(box.x + 12, y + 12, clip(row.left, 14), COLOR.muted, 10));
      parts.push(`<text x="${box.x + box.w - 12}" y="${y + 12}" fill="${fill}" ${row.mono ? MONO : FONT} font-size="10" text-anchor="end">${escapeXml(clip(row.right, 24))}</text>`);
    } else {
      parts.push(text(box.x + 12, y + 12, clip(row.left, 34), fill, 10));
    }
    y += ROW_PITCH;
  }
  return parts.join('');
}

function renderPorts(node: ModelNode, box: Box, sides: NodePortSides, COLOR: Palette): string {
  const ports = displayedPorts(node.data);
  const captured = Boolean(node.data.captureDisplayPorts || node.data.capturePorts);
  const parts: string[] = [];
  const draw = (ids: string[], side: AnchorSide) => {
    ids.forEach((id, index) => {
      const point = portAnchorPoint(box, sides.ports?.[id] ?? side, index, ids.length);
      const port = ports.find(port => port.id === id);
      const failed = port && isFailingPort(node.data, port);
      parts.push(`<circle cx="${point.x}" cy="${point.y}" r="${failed ? 4 : 3.25}" fill="${failed ? COLOR.error : port?.direction === 'out' || node.data.outputPorts?.includes(id) ? COLOR.portOut : COLOR.portIn}" opacity="0.75"/>`);
    });
  };
  draw(captured ? ports.filter(p => p.direction === 'in').map(p => p.id) : node.data.inputPorts ?? [], sides.in);
  draw(captured ? ports.filter(p => p.direction === 'out').map(p => p.id) : node.data.outputPorts ?? [], sides.out);
  return parts.join('');
}

export function graphToSvg(nodes: ModelNode[], edges: ModelEdge[], options: GraphSvgOptions = {}): string {
  const COLOR = options.theme === 'light' ? PAPER : INK;
  const FAMILY = options.theme === 'light' ? FAMILY_PAPER : FAMILY_INK;
  // The card's glyph tile: the family glyph on a tile tinted like the card's (Ink only).
  const glyph = (node: ModelNode, x: number, y: number) => {
    const kind = classifyNode(node).kind;
    const color = FAMILY[familyToken(kind)];
    const tile = isBlockFamily(kind) && COLOR.glyphTint ? mixHex(color, COLOR.glyphTint, COLOR.glyphTile) : COLOR.glyphTile;
    return `<g data-block="${kind}"><rect x="${x}" y="${y}" width="24" height="24" rx="5" fill="${tile}"/>${blockGlyphSvg(kind, x + 5, y + 5, 14, color)}</g>`;
  };
  const reviewInk = (status: string | undefined, fallback: string) => options.theme === 'light'
    ? status === 'modified' ? '#824e00' : status === 'rewired' ? '#1f52b4' : fallback : fallback;
  const padding = options.padding ?? 24;
  const background = options.background ?? COLOR.background;
  const positions = absolutePositions(nodes);
  const visible = nodes.filter((node) => !node.hidden && positions.has(node.id));
  const boxes = new Map(visible.map((node) => [node.id, boxOf(node, positions.get(node.id)!)] as const));
  const router = createEdgeRouter(nodes, edges, options.direction);
  const routes = edges.flatMap(edge => {
    if (edge.hidden) return [];
    const route = router.route(edge, id => boxes.get(id));
    return route ? [{ edge, route }] : [];
  });

  const bounds = { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  if (visible.length > 0) {
    bounds.minX = Math.min(...visible.map((node) => boxes.get(node.id)!.x));
    bounds.minY = Math.min(...visible.map((node) => boxes.get(node.id)!.y));
    bounds.maxX = Math.max(...visible.map((node) => boxes.get(node.id)!.x + boxes.get(node.id)!.w));
    bounds.maxY = Math.max(...visible.map((node) => boxes.get(node.id)!.y + boxes.get(node.id)!.h));
  }
  for (const { route } of routes) {
    bounds.minX = Math.min(bounds.minX, route.bounds.minX);
    bounds.minY = Math.min(bounds.minY, route.bounds.minY);
    bounds.maxX = Math.max(bounds.maxX, route.bounds.maxX);
    bounds.maxY = Math.max(bounds.maxY, route.bounds.maxY);
  }
  const width = Math.max(1, bounds.maxX - bounds.minX + padding * 2);
  const height = Math.max(1, bounds.maxY - bounds.minY + padding * 2);
  const viewBox = `${bounds.minX - padding} ${bounds.minY - padding} ${width} ${height}`;

  const parts: string[] = [];
  parts.push(`<defs><marker id="tv-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 7 4 L 0 8 z" fill="context-stroke"/></marker></defs>`);
  parts.push(`<rect x="${bounds.minX - padding}" y="${bounds.minY - padding}" width="${width}" height="${height}" fill="${background}"/>`);

  // Edges first, so cards and frames draw over them.
  const edgePaths: string[] = [];
  for (const { edge, route } of routes) {
    const { start, end, startSide, endSide, curveOffset } = route;
    const { c1x, c1y, c2x, c2y } = edgeControlPoints(start.x, start.y, end.x, end.y, curveOffset, startSide, endSide);
    const stroke = edge.data?.diffStatus === 'added' ? COLOR.added
      : edge.data?.diffStatus === 'removed' ? COLOR.removed
        : edge.data?.diffStatus === 'modified' ? COLOR.modified
          : edge.data?.role === 'residual' ? COLOR.residual : COLOR.edge;
    const dashed = edge.data?.role === 'residual' ? ' stroke-dasharray="1 6" stroke-linecap="round"' : edge.data?.debugPrevious || (edge.data?._relationship && !isFlowEdge(edge)) ? ' stroke-dasharray="6 5"' : '';
    edgePaths.push(`<path d="M ${start.x} ${start.y} C ${c1x} ${c1y} ${c2x} ${c2y} ${end.x} ${end.y}" fill="none" stroke="${stroke}" stroke-width="1.4" opacity="0.85" marker-end="url(#tv-arrow)"${dashed}/>`);
  }
  parts.push(`<g>${edgePaths.join('')}</g>`);

  // Frames (outer before inner), then leaf cards on top.
  const modules = visible.filter((node) => node.type === 'moduleNode')
    .sort((a, b) => {
      const left = boxes.get(a.id)!, right = boxes.get(b.id)!;
      return right.w * right.h - left.w * left.h;
    });
  const moduleRects: string[] = [];
  for (const node of modules) {
    const box = boxes.get(node.id)!;
    const header = moduleCardHeader(node.data, box.w);
    const count = node.data?.moduleLayerCount;
    const collapsed = !node.data?.expanded;
    const status = node.data?.moduleDiagnosticStatus ?? (node.data?.error ? 'error' : node.data?.moduleStatus);
    const diffColor = node.data.diffStatus && COLOR[node.data.diffStatus];
    const stroke = status === 'error' ? COLOR.error : diffColor || COLOR.moduleStroke;
    const statusDot = status === 'error' ? COLOR.error : status === 'warning' ? COLOR.warning : undefined;
    moduleRects.push(`<rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" rx="8" fill="${COLOR.moduleFill}" stroke="${stroke}" stroke-width="${diffColor ? 2 : 1.5}"${!diffColor || node.data.diffStatus === 'removed' ? ' stroke-dasharray="5 4"' : ''}/>`);
    moduleRects.push(`<rect x="${box.x}" y="${box.y}" width="${box.w}" height="${header.height}" rx="8" fill="${diffColor || COLOR.moduleHeader}"${diffColor ? ' fill-opacity="0.12"' : ''}/>`);
    moduleRects.push(glyph(node, box.x + 6, box.y + 5));
    if (statusDot) moduleRects.push(`<circle cx="${box.x + box.w - 14}" cy="${box.y + 17}" r="4" fill="${statusDot}"/>`);
    moduleRects.push(identityText(box, header, COLOR));
    moduleRects.push(sourceEvidence(node, box, COLOR));
    const moduleRows: Row[] = [];
    if (node.data.review && node.data.review.status !== 'unchanged') {
      moduleRects.push(text(box.x + 12, box.y + header.height + 16, graphReviewLabel(node.data.review), reviewInk(node.data.diffStatus, diffColor || COLOR.muted), 9.5));
    } else if (node.data.debugEvidence === 'previous' || node.data.debugChanges?.length) {
      moduleRows.push({ left: node.data.debugEvidence === 'previous' ? 'Previous run'
        : `Changed · ${node.data.debugChanges![0]}`, tone: node.data.debugEvidence === 'previous' ? 'muted' : 'warning' });
    } else if (count) {
      moduleRows.push({ left: `${count} layers · ${collapsed ? 'collapsed' : 'expanded'}`, tone: 'muted' });
    }
    if (collapsed) {
      moduleBoundaryRows(node.data).forEach(row => moduleRows.push({ left: row.label, right: row.value, tone: 'shape' }));
      const evidence = moduleEvidenceLabel(node.data);
      if (evidence) moduleRows.push({ left: evidence, tone: 'muted' });
      const contents = moduleContents(node.data);
      if (!count && contents.label) moduleRows.push({ left: contents.label, tone: 'muted' });
      if (node.data.capturePassThrough) moduleRows.push({ left: 'Input unchanged', tone: 'muted' });
      moduleRects.push(renderCardRows(box, moduleRows, COLOR, header.height + (node.data.review && node.data.review.status !== 'unchanged' ? 20 : 0)));
    } else if (moduleRows.length) {
      // Expanded frames reserve their body for children; structural context is a quiet caption.
      moduleRects.push(text(box.x + 12, box.y + header.height + 16, moduleRows[0].left, COLOR.muted, 9.5));
    }
    if (collapsed && Array.isArray(node.data?.modulePreviewNodes)) {
      const preview = node.data.modulePreviewNodes.slice(0, 24);
      const top = header.height + moduleRows.length * ROW_PITCH + 20;
      const span = { w: box.w - 40, h: Math.max(0, box.h - top - 20) };
      const xs = preview.map((entry) => entry.x), ys = preview.map((entry) => entry.y);
      const minX = Math.min(...xs, 0), minY = Math.min(...ys, 0);
      const scaleX = xs.length ? span.w / Math.max(1, Math.max(...xs) - minX || 1) : 1;
      const scaleY = ys.length ? span.h / Math.max(1, Math.max(...ys) - minY || 1) : 1;
      if (span.h > 0) preview.forEach((entry) => moduleRects.push(`<rect x="${box.x + 20 + (entry.x - minX) * scaleX}" y="${box.y + top + (entry.y - minY) * scaleY}" width="24" height="10" rx="2" fill="${COLOR.moduleHeader}" stroke="${COLOR.cardStroke}"/>`));
    }
  }
  for (const node of modules) moduleRects.push(renderPorts(node, boxes.get(node.id)!, router.sides.get(node.id)!, COLOR));
  parts.push(`<g>${moduleRects.join('')}</g>`);

  const cards: string[] = [];
  for (const node of visible) {
    if (node.type === 'moduleNode') continue;
    const box = boxes.get(node.id)!;
    const data = operationGroupCard(node.data);
    const isIo = node.type === 'inputNode' || node.type === 'outputNode';
    const header = isIo ? cardHeaderLayout(node.type === 'inputNode' ? 'Input' : 'Output', data.captureLabel, box.w)
      : layerCardHeader(data, box.w);
    const stroke = data.error ? COLOR.error
      : data.diffStatus === 'added' ? COLOR.added
        : data.diffStatus === 'removed' ? COLOR.removed
          : data.diffStatus === 'modified' ? COLOR.modified : data.diffStatus === 'rewired' ? COLOR.rewired : COLOR.cardStroke;
    cards.push(`<rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" rx="6" fill="${COLOR.cardFill}" stroke="${stroke}" stroke-width="1.5"/>`);
    cards.push(glyph(node, box.x + 6, box.y + 7));
    cards.push(identityText(box, header, COLOR));
    cards.push(sourceEvidence(node, box, COLOR));
    const reviewLabel = data.review ? graphReviewLabel(data.review) : data.diffStatus;
    if (reviewLabel) cards.push(text(box.x + 12, box.y + header.height + 16, reviewLabel, reviewInk(data.diffStatus, data.diffStatus ? COLOR[data.diffStatus] : COLOR.muted), 9.5, 600));
    const rows = !isIo && (data.review || data.sourceAssignment) ? cardRows(node) : compactRows(node);
    cards.push(renderCardRows(box, rows, COLOR, header.height + (reviewLabel ? 20 : 0)));
    cards.push(renderPorts(node, box, router.sides.get(node.id)!, COLOR));
  }
  parts.push(`<g>${cards.join('')}</g>`);

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${viewBox}">${parts.join('')}</svg>`;
}

export interface SvgNodeBounds { id: string; x: number; y: number; w: number; h: number }

/** Bounds for hit-testing a click on the rendered image back to a node. */
export function graphSvgBounds(nodes: ModelNode[]): SvgNodeBounds[] {
  const positions = absolutePositions(nodes);
  return nodes.filter((node) => !node.hidden && positions.has(node.id))
    .map((node) => {
      const box = boxOf(node, positions.get(node.id)!);
      return { id: node.id, x: box.x, y: box.y, w: box.w, h: box.h };
    });
}
