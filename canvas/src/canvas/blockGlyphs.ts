/**
 * @license Icon geometry from Lucide (lucide-react 0.511.0), ISC License.
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of
 * Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 * https://lucide.dev/license
 */
import type { BlockKind } from './blockFamily';

type GlyphNode = readonly [tag: 'path' | 'rect' | 'circle' | 'line' | 'polyline', attributes: Readonly<Record<string, string>>];

/** One 24x24 line glyph per block kind, shared by the cards and the SVG export so
 *  both draw the same icon in the family colour. Copied from the Lucide icons the
 *  Canvas already ships; the per-icon modules are ESM-only and cannot be imported
 *  everywhere this table is needed. */
export const BLOCK_GLYPHS: Record<BlockKind, readonly GlyphNode[]> = {
  convolution: [['rect', {width: '18', height: '18', x: '3', y: '3', rx: '2'}], ['path', {d: 'M3 9h18'}], ['path', {d: 'M3 15h18'}], ['path', {d: 'M9 3v18'}], ['path', {d: 'M15 3v18'}]],
  linear: [['rect', {x: '16', y: '16', width: '6', height: '6', rx: '1'}], ['rect', {x: '2', y: '16', width: '6', height: '6', rx: '1'}], ['rect', {x: '9', y: '2', width: '6', height: '6', rx: '1'}], ['path', {d: 'M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3'}], ['path', {d: 'M12 12V8'}]],
  normalization: [['path', {d: 'm16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z'}], ['path', {d: 'm2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z'}], ['path', {d: 'M7 21h10'}], ['path', {d: 'M12 3v18'}], ['path', {d: 'M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2'}]],
  activation: [['path', {d: 'M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2'}]],
  pooling: [['path', {d: 'm15 15 6 6m-6-6v4.8m0-4.8h4.8'}], ['path', {d: 'M9 19.8V15m0 0H4.2M9 15l-6 6'}], ['path', {d: 'M15 4.2V9m0 0h4.8M15 9l6-6'}], ['path', {d: 'M9 4.2V9m0 0H4.2M9 9 3 3'}]],
  dropout: [['path', {d: 'M10.1 2.182a10 10 0 0 1 3.8 0'}], ['path', {d: 'M13.9 21.818a10 10 0 0 1-3.8 0'}], ['path', {d: 'M17.609 3.721a10 10 0 0 1 2.69 2.7'}], ['path', {d: 'M2.182 13.9a10 10 0 0 1 0-3.8'}], ['path', {d: 'M20.279 17.609a10 10 0 0 1-2.7 2.69'}], ['path', {d: 'M21.818 10.1a10 10 0 0 1 0 3.8'}], ['path', {d: 'M3.721 6.391a10 10 0 0 1 2.7-2.69'}], ['path', {d: 'M6.391 20.279a10 10 0 0 1-2.69-2.7'}]],
  recurrent: [['path', {d: 'm17 2 4 4-4 4'}], ['path', {d: 'M3 11v-1a4 4 0 0 1 4-4h14'}], ['path', {d: 'm7 22-4-4 4-4'}], ['path', {d: 'M21 13v1a4 4 0 0 1-4 4H3'}]],
  attention: [['circle', {cx: '12', cy: '12', r: '3'}], ['path', {d: 'M3 7V5a2 2 0 0 1 2-2h2'}], ['path', {d: 'M17 3h2a2 2 0 0 1 2 2v2'}], ['path', {d: 'M21 17v2a2 2 0 0 1-2 2h-2'}], ['path', {d: 'M7 21H5a2 2 0 0 1-2-2v-2'}]],
  embedding: [['line', {x1: '4', x2: '20', y1: '9', y2: '9'}], ['line', {x1: '4', x2: '20', y1: '15', y2: '15'}], ['line', {x1: '10', x2: '8', y1: '3', y2: '21'}], ['line', {x1: '16', x2: '14', y1: '3', y2: '21'}]],
  padding: [['line', {x1: '22', x2: '2', y1: '6', y2: '6'}], ['line', {x1: '22', x2: '2', y1: '18', y2: '18'}], ['line', {x1: '6', x2: '6', y1: '2', y2: '22'}], ['line', {x1: '18', x2: '18', y1: '2', y2: '22'}]],
  resampling: [['path', {d: 'M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7'}], ['path', {d: 'M14 15H9v-5'}], ['path', {d: 'M16 3h5v5'}], ['path', {d: 'M21 3 9 15'}]],
  shape: [['path', {d: 'm18 14 4 4-4 4'}], ['path', {d: 'm18 2 4 4-4 4'}], ['path', {d: 'M2 18h1.973a4 4 0 0 0 3.3-1.7l5.454-8.6a4 4 0 0 1 3.3-1.7H22'}], ['path', {d: 'M2 6h1.972a4 4 0 0 1 3.6 2.2'}], ['path', {d: 'M22 18h-6.041a4 4 0 0 1-3.3-1.8l-.359-.45'}]],
  distance: [['path', {d: 'M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z'}], ['path', {d: 'm14.5 12.5 2-2'}], ['path', {d: 'm11.5 9.5 2-2'}], ['path', {d: 'm8.5 6.5 2-2'}], ['path', {d: 'm17.5 15.5 2-2'}]],
  loss: [['circle', {cx: '12', cy: '12', r: '10'}], ['circle', {cx: '12', cy: '12', r: '6'}], ['circle', {cx: '12', cy: '12', r: '2'}]],
  arithmetic: [['path', {d: 'm8 6 4-4 4 4'}], ['path', {d: 'M12 2v10.3a4 4 0 0 1-1.172 2.872L4 22'}], ['path', {d: 'm20 22-5-5'}]],
  input: [['path', {d: 'M5 12h14'}], ['path', {d: 'm12 5 7 7-7 7'}]],
  output: [['path', {d: 'm12 19-7-7 7-7'}], ['path', {d: 'M19 12H5'}]],
  container: [['path', {d: 'M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z'}], ['path', {d: 'm7 16.5-4.74-2.85'}], ['path', {d: 'm7 16.5 5-3'}], ['path', {d: 'M7 16.5v5.17'}], ['path', {d: 'M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z'}], ['path', {d: 'm17 16.5-5-3'}], ['path', {d: 'm17 16.5 4.74-2.85'}], ['path', {d: 'M17 16.5v5.17'}], ['path', {d: 'M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z'}], ['path', {d: 'M12 8 7.26 5.15'}], ['path', {d: 'm12 8 4.74-2.85'}], ['path', {d: 'M12 13.5V8'}]],
  composite: [['path', {d: 'M11 21.73a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73z'}], ['path', {d: 'M12 22V12'}], ['polyline', {points: '3.29 7 12 12 20.71 7'}], ['path', {d: 'm7.5 4.27 9 5.15'}]],
  unknown: [['path', {d: 'M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z'}], ['path', {d: 'm3.3 7 8.7 5 8.7-5'}], ['path', {d: 'M12 22V12'}]],
};

const escape = (value: string) => value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

/** The glyph as standalone SVG markup for the overview and export. */
export function blockGlyphSvg(kind: BlockKind, x: number, y: number, size: number, color: string): string {
  const shapes = BLOCK_GLYPHS[kind].map(([tag, attributes]) =>
    `<${tag} ${Object.entries(attributes).map(([key, value]) => `${key}="${escape(value)}"`).join(' ')}/>`).join('');
  return `<g transform="translate(${x} ${y}) scale(${size / 24})" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${shapes}</g>`;
}
