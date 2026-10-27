/**
 * Paint one measured number onto the graph instead of printing it.
 *
 * Exact introspected counts and captured output tensor footprints share the
 * cards' evidence rules. Legacy rounded counts and shape-only snapshots do
 * not supply exact measurements.
 *
 * Two rules keep this honest:
 *   - output bytes require complete MEASURED tensor evidence. An inferred or
 *     unknown shape yields no value, so a node is left uncoloured rather than
 *     tinted from a guess (a colour carries no provenance badge to warn with).
 *   - the scale comes from the values actually present, so the legend always
 *     names numbers that exist in this graph.
 */
import type { ModelNode } from '../types';
import { measuredOutputTensor, measuredParameterCount } from './graphSemantics';

export type OverlayMetric = 'none' | 'params' | 'activation';

export interface OverlayValue {
  id: string;
  value: number;
  /** 0..1 within this graph's range, for the colour ramp. */
  heat: number;
}

export interface OverlayScale {
  metric: OverlayMetric;
  values: Map<string, OverlayValue>;
  min: number;
  max: number;
  unit: string;
  /** Nodes skipped because the metric could not be derived honestly. */
  unmeasured: number;
}

export const EMPTY_OVERLAY: OverlayScale = {
  metric: 'none', values: new Map(), min: 0, max: 0, unit: '', unmeasured: 0,
};

export function overlayScale(nodes: readonly ModelNode[], metric: OverlayMetric): OverlayScale {
  if (metric === 'none') return EMPTY_OVERLAY;
  const read = (node: ModelNode) => {
    if (metric === 'params') return measuredParameterCount(node.data);
    return measuredOutputTensor(node.data)?.logicalBytes;
  };
  const unit = metric === 'params' ? 'params' : 'bytes';
  const raw: Array<{ id: string; value: number }> = [];
  let unmeasured = 0;

  for (const node of nodes) {
    if (node.hidden) continue;
    const value = read(node);
    if (value === undefined) { unmeasured += 1; continue; }
    raw.push({ id: node.id, value });
  }
  if (raw.length === 0) {
    return { ...EMPTY_OVERLAY, metric, unit, unmeasured };
  }

  const min = Math.min(...raw.map(entry => entry.value));
  const max = Math.max(...raw.map(entry => entry.value));
  const span = max - min;
  const values = new Map<string, OverlayValue>(
    // A flat graph (every node equal) is all-cold, not all-hot: there is no
    // outlier to point at, and painting every card red would invent one.
    raw.map(entry => [entry.id, { id: entry.id, value: entry.value, heat: span > 0 ? (entry.value - min) / span : 0 }]),
  );
  return { metric, values, min, max, unit, unmeasured };
}

/** Short, honest units. 32,768 bytes is "32 KB", not "0.03 MB". */
export function formatOverlayValue(value: number, unit: string): string {
  if (unit === 'bytes') {
    if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
    if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${Math.round(value)} B`;
  }
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return `${Math.round(value)}`;
}

/**
 * Cold-to-hot ramp. Deliberately not a hue rotation through green: green and
 * amber already mean "measured" and "inferred" on this canvas, and a metric
 * that borrowed them would read as a provenance claim.
 */
const RAMP = ['#1e3a5f', '#3b6fa8', '#6f9fd6', '#d6a13a', '#e5687a'];

export function heatColor(heat: number): string {
  const clamped = Math.max(0, Math.min(1, heat));
  const scaled = clamped * (RAMP.length - 1);
  const index = Math.min(RAMP.length - 2, Math.floor(scaled));
  const t = scaled - index;
  const from = RAMP[index];
  const to = RAMP[index + 1];
  const mix = (offset: number) => {
    const a = parseInt(from.slice(offset, offset + 2), 16);
    const b = parseInt(to.slice(offset, offset + 2), 16);
    return Math.round(a + (b - a) * t).toString(16).padStart(2, '0');
  };
  return `#${mix(1)}${mix(3)}${mix(5)}`;
}
