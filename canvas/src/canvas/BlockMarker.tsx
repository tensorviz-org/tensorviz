import React from 'react';
import { Activity, ArrowLeft, ArrowRight, Box, Boxes, CircleDashed, Focus, Frame, Grid3x3, Hash, LucideProps, Merge, Network,
  Package, Repeat, Ruler, Scale, Scaling, Shrink, Shuffle, Target } from 'lucide-react';
import { BlockClassification, BlockKind, isBlockFamily } from './blockFamily';

type Glyph = React.ComponentType<LucideProps>;

/** One line glyph per family, so hue is never the only cue. */
const GLYPHS: Record<BlockKind, Glyph> = {
  convolution: Grid3x3, linear: Network, normalization: Scale, activation: Activity, pooling: Shrink,
  dropout: CircleDashed, recurrent: Repeat, attention: Focus, embedding: Hash, padding: Frame, resampling: Scaling,
  shape: Shuffle, distance: Ruler, loss: Target, arithmetic: Merge,
  input: ArrowRight, output: ArrowLeft, container: Boxes, composite: Package, unknown: Box,
};

export const blockColor = (kind: BlockKind): string => isBlockFamily(kind) ? `var(--tv-family-${kind})`
  : kind === 'input' || kind === 'output' ? 'var(--tv-family-io)' : 'var(--tv-family-neutral)';

export const blockTitle = (block: BlockClassification): string => block.modifier ? `${block.label} · ${block.modifier}` : block.label;

export function BlockGlyph({ kind, size = 14 }: { kind: BlockKind; size?: number }) {
  const Icon = GLYPHS[kind];
  return <Icon size={size} aria-hidden="true" />;
}

/** The family's only colored area on a card: a short bar beside the title. */
export function BlockMarker({ block }: { block: BlockClassification }) {
  return <span className="block-marker" data-block={block.kind} aria-hidden="true"
    style={{ ['--block-color' as string]: blockColor(block.kind) } as React.CSSProperties} />;
}

/** Glyph tile plus marker, named for assistive technology and on hover. */
export function BlockIdentity({ block, className = 'ui2-node-icon', size = 14 }: { block: BlockClassification; className?: string; size?: number }) {
  const title = blockTitle(block);
  return <>
    <span className={className} role="img" aria-label={title} title={title} data-block={block.kind}><BlockGlyph kind={block.kind} size={size} /></span>
    <BlockMarker block={block} />
  </>;
}
