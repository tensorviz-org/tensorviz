import { wrapCardText } from './cardIdentity';

/** The exact same line layout drives card rendering and its graph footprint. */
export function boundaryTextLayout(label: string, value: string, width: number, compact = false, root = false) {
  const available = Math.max(48, width - 42);
  const labelLines = compact ? [label] : wrapCardText(label, available, root ? 9 : 11, root);
  const valueLines = wrapCardText(value, compact ? Math.max(48, available - 42) : available, 12, true);
  const height = compact ? Math.max(22, valueLines.length * 18 + 4)
    : Math.max(36, labelLines.length * 14 + valueLines.length * 18 + 4);
  return { labelLines, valueLines, height, compact };
}

export function sourceLinkLayout(location: { file: string; line: number }, width: number) {
  const value = `${location.file.split(/[\\/]/).pop()}:${location.line}`;
  const lines = wrapCardText(value, Math.max(48, width - 42), 10.5, true);
  return { lines, height: Math.max(18, lines.length * 15 + 3) };
}

export function definitionLinkLayout(file: string, width: number) {
  // Body padding and the source icon/gap share the source-location text column.
  const lines = wrapCardText(`defined in ${file}`, Math.max(48, width - 42), 10, true);
  return { lines, height: lines.length * 15 };
}

export function moduleRouteLayout(entry: string, exit: string, width: number) {
  const label = `${entry.split('__').pop()} → ${exit.split('__').pop()}`;
  const lines = wrapCardText(label, Math.max(48, width - 42), 10, true);
  return { label, lines, height: lines.length * 15 };
}

export function ioValueLayout(value: string, width: number) {
  const lines = wrapCardText(value, Math.max(48, width - 98), 11, true);
  return { lines, height: Math.max(16.5, lines.length * 16.5) };
}

/** Shared by captured operations, legacy tensor summaries and SVG paint. */
export function tensorRowLayout(label: string, value: string, width: number, lineHeight = 21) {
  const labelWidth = Math.min(65, Math.max(24, label.length * 7));
  const labelLines = wrapCardText(label, labelWidth, 10);
  const valueLines = wrapCardText(value, Math.max(48, width - 50 - labelWidth), 12, true);
  return { labelLines, valueLines, labelWidth, lineHeight,
    height: Math.max(21, labelLines.length * 14, valueLines.length * lineHeight) };
}
