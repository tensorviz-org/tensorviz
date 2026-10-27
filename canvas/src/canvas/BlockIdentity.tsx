import React from 'react';
import { BlockClassification, BlockKind, isBlockFamily } from './blockFamily';
import { BLOCK_GLYPHS } from './blockGlyphs';

/** The family colour, or the neutral glyph colour for structural and unclassified blocks. */
export const blockColor = (kind: BlockKind): string => isBlockFamily(kind) ? `var(--tv-family-${kind})`
  : kind === 'input' || kind === 'output' ? 'var(--tv-family-io)' : 'var(--tv-family-neutral)';

export const blockTitle = (block: BlockClassification): string => block.modifier ? `${block.label} · ${block.modifier}` : block.label;

/** One line glyph per kind, drawn in its family colour so hue and shape name the block together. */
export function BlockGlyph({ kind, size = 14 }: { kind: BlockKind; size?: number }) {
  // currentColor rather than a var() attribute: presentation attributes do not take
  // custom properties, and forced-colors mode can still override `color`.
  return <svg className="block-glyph" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" data-block={kind} style={{ color: blockColor(kind) }}>
    {BLOCK_GLYPHS[kind].map(([tag, attributes], index) => React.createElement(tag, { key: index, ...attributes }))}
  </svg>;
}

/** The card's glyph tile, named for assistive technology and on hover. Nothing else on
 *  the card carries the family colour. A family also tints its tile (theme
 *  `--tv-glyph-tint`) so the hue reads at a glance; neutral blocks keep a neutral tile. */
export function BlockIdentity({ block, className = 'ui2-node-icon', size = 14 }: { block: BlockClassification; className?: string; size?: number }) {
  const title = blockTitle(block);
  return <span className={className} role="img" aria-label={title} title={title} data-block={block.kind}
    style={isBlockFamily(block.kind) ? { ['--block-tint' as string]: blockColor(block.kind) } as React.CSSProperties : undefined}>
    <BlockGlyph kind={block.kind} size={size} />
  </span>;
}
