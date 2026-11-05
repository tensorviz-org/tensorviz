import { graphToSvg, graphSvgBounds } from '../../services/graphSvg';
import { ModelEdge, ModelNode } from '../../types';

const moduleNode = (id: string, position = { x: 100, y: 100 }, extra: any = {}): ModelNode =>
  ({
    id, type: 'moduleNode', position, style: { width: 400, height: 300 },
    data: { layerType: { id: 'module', name: 'Module' }, parameters: { name: 'Block' },
      isGroup: true, expanded: true, moduleInstanceId: id, moduleLayerCount: 5, ...extra },
  } as unknown as ModelNode);

const layer = (id: string, parentNode: string | undefined, position: { x: number; y: number }): ModelNode =>
  ({
    id, type: 'layerNode', parentNode, position,
    data: { layerType: { id: 'conv2d', name: 'Conv2d' }, parameters: { name: id, in_channels: 3, out_channels: 8 } },
  } as unknown as ModelNode);

const edge = (id: string, source: string, target: string, extra: any = {}): ModelEdge =>
  ({ id, source, target, data: {}, ...extra } as unknown as ModelEdge);

const svgDocument = (svg: string) => new DOMParser().parseFromString(svg, 'image/svg+xml');
// Frames, headers and cards; shapes inside a family glyph are not boxes.
const rects = (svg: string) => Array.from(svgDocument(svg).querySelectorAll('rect')).filter(rect => !rect.closest('[data-block]')).length;
const glyphs = (svg: string) => Array.from(svgDocument(svg).querySelectorAll('g[data-block]'));
const stroke = (glyph: Element) => glyph.querySelector('g')!.getAttribute('stroke');
const tile = (glyph: Element) => glyph.querySelector('rect')!.getAttribute('fill');
const titleTexts = (svg: string) => Array.from(svgDocument(svg).querySelectorAll('[data-card-title]'), element => element.textContent);
const subtitleTexts = (svg: string) => Array.from(svgDocument(svg).querySelectorAll('[data-card-subtitle]'), element => element.textContent);

describe('graphToSvg', () => {
  it('draws labels as SVG text rather than foreignObject HTML', () => {
    const svg = graphToSvg([layer('conv', undefined, { x: 0, y: 0 })], []);
    expect(svg).toContain('<text');
    expect(svg).not.toContain('foreignObject');
  });
  it('switches the export palette without changing graph bounds or review meaning', () => {
    const node = layer('a', undefined, { x: 0, y: 0 });
    node.data = { ...node.data, diffStatus: 'rewired', review: { status: 'rewired', params: {}, inputChanges: { added: 1, removed: 0 } } };
    const ink = graphToSvg([node], [], { theme: 'dark' });
    const paper = graphToSvg([node], [], { theme: 'light' });
    const parse = (svg: string) => new DOMParser().parseFromString(svg, 'image/svg+xml');
    expect(parse(paper).documentElement.getAttribute('viewBox')).toBe(parse(ink).documentElement.getAttribute('viewBox'));
    expect(paper).toContain('fill="#f3f2ee"');
    expect(paper).toContain('stroke="#2563d9"');
    expect(paper).toContain('fill="#1f52b4"');
    expect(paper).toContain('>inputs changed</text>');
    expect(ink).toContain('fill="#171817"');
    expect(ink).toContain('stroke="#8bbcff"');
  });
  it('exports the same result operation title as the interactive card', () => {
    const node = layer('conditional', undefined, { x: 0, y: 0 });
    node.data.layerType = { ...node.data.layerType, id: 'python', name: 'Python', displayName: 'Python' };
    node.data.parameters = { _raw_call: 'torch.where(masks.unsqueeze(-1), token.to(x.dtype).unsqueeze(0), x)',
      _source_result_call: { name: 'where', target: 'torch.where' } };
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toContain('Select by condition');
    expect(subtitleTexts(svg)).toContain('torch.where');
    expect(svg).not.toContain('>Python<');
  });
  it('keeps exact source identity without exposing an internal Python node id', () => {
    const node = layer('python@Model#4', undefined, { x: 0, y: 0 });
    node.data = { ...node.data, layerType: { ...node.data.layerType, id: 'python', name: 'Python', displayName: 'Python' },
      captureLabel: 'python@Model#4', parameters: { _raw_call: 'self.compare(features, question)', _source_result_call: { name: 'compare' } } };
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toContain('compare');
    expect(subtitleTexts(svg)).not.toContain('self.compare');
    expect(svg).not.toContain('python@Model#4');
  });
  it('keeps source-only evidence in a separate quiet icon beside the complete callable', () => {
    const node = layer('source', undefined, { x: 0, y: 0 });
    const callable = 'self.candidate_scorer.compare_temporal_representations';
    node.data = { ...node.data, debugEvidence: 'source', shapeEvidence: 'unknown',
      layerType: { ...node.data.layerType, id: 'python', name: 'Python', displayName: 'Python' },
      parameters: { _raw_call: `${callable}(candidates, question)`, _source_result_call: { name: 'compare_temporal_representations' } } };
    const svg = graphToSvg([node], []);
    const icon = svgDocument(svg).querySelector('[data-source-evidence="source"]');
    expect(icon?.getAttribute('aria-label')).toBe('Read from code · tensor shapes not recorded');
    expect(icon?.querySelector('path')).not.toBeNull();
    expect(subtitleTexts(svg)).toEqual([callable]);
    expect(svg).not.toContain('…');
  });
  it('exports explanations with the same enlarged hit bounds while keeping technical identity', () => {
    const nodes = [layer('score', undefined, { x: 0, y: 0 }), layer('next', undefined, { x: 0, y: 100 })];
    const before = JSON.stringify(nodes);
    const options = { annotationsVisible: true, explanationStates: { score: { requestId: 'request', nodeId: 'score', revision: 'revision', status: 'ready' as const,
      explanation: { title: 'Compare candidate representations', purpose: 'Compare candidate features against the question.',
        inputs: ['Candidate features', 'Question features'], outputs: ['Raw candidate scores'], steps: [], uncertainties: [],
        citations: [{ id: 'source', file: 'model.py', sourceDigest: 'a'.repeat(64), startLine: 10, endLine: 20 }] } } } };
    const bounds = graphSvgBounds(nodes, options);
    const svg = graphToSvg(nodes, [edge('flow', 'score', 'next')], options);
    const document = svgDocument(svg);
    expect(titleTexts(svg)).toContain('Conv2d');
    expect(document.querySelector('[data-agent-interpretation]')!.textContent).toContain('Candidate features · Question features');
    expect(bounds[1].y).toBeGreaterThanOrEqual(bounds[0].y + bounds[0].h + 24);
    expect(bounds[0].h).toBeGreaterThan(graphSvgBounds(nodes)[0].h);
    expect(JSON.stringify(nodes)).toBe(before);
  });
  it.each(['layerNode', 'inputNode', 'outputNode', 'moduleNode'])('exports %s with the same connection-only label and color', type => {
    const node = type === 'moduleNode' ? moduleNode('a') : { ...layer('a', undefined, { x: 0, y: 0 }), type };
    node.data = { ...node.data, diffStatus: 'rewired', review: { status: 'rewired', params: {}, inputChanges: { added: 1, removed: 1 } } };
    const svg = graphToSvg([node], []);
    expect(svg).toContain('>inputs changed</text>');
    expect(svg).toContain('stroke="#8bbcff"');
    expect(svg).not.toContain('>modified<');
    expect(svg).not.toContain('>rewired<');
  });
  it('exports cards without their static-analysis notes, as the canvas draws them', () => {
    const node = layer('a', undefined, { x: 0, y: 0 });
    node.data = { ...node.data, resolutionSeverity: 'info', resolutionDetail: 'transformers is external to the workspace' };
    expect(graphToSvg([node], [])).not.toContain('transformers is external');
  });
  it('exports both reasons when source and connections changed', () => {
    const node = layer('a', undefined, { x: 0, y: 0 });
    node.data = { ...node.data, diffStatus: 'modified', review: { status: 'modified', params: {}, inputChanges: { added: 1, removed: 1 } } };
    expect(graphToSvg([node], [])).toContain('>~ modified + inputs</text>');
  });
  it('wraps the graph in one standalone svg with a viewBox', () => {
    const svg = graphToSvg([layer('a', undefined, { x: 0, y: 0 })], []);
    expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg"')).toBe(true);
    expect(svg).toContain('viewBox=');
    expect(svg).toContain('</svg>');
    // background + one card, plus the card's family glyph and no marker bar
    expect(rects(svg)).toBe(2);
    expect(glyphs(svg)).toHaveLength(1);
    expect(svg).not.toContain('width="3" height="20"');
  });

  it('resolves module-relative child positions to absolute page coordinates', () => {
    const svg = graphToSvg([moduleNode('m'), layer('m__c', 'm', { x: 20, y: 60 })], []);
    // module at 100 + child offset 20
    expect(svg).toContain('x="120"');
    // module frame + header + card + background, plus one glyph each for module and card
    expect(rects(svg)).toBe(4);
    expect(glyphs(svg).map(glyph => glyph.getAttribute('data-block'))).toEqual(['composite', 'convolution']);
  });

  it('draws a module title with its structural layer count', () => {
    const svg = graphToSvg([moduleNode('m')], []);
    expect(svg).toContain('Block');
    expect(svg).toContain('5 layers');
  });

  it.each([true, false])('highlights changed module frames in SVG with expanded=%s without moving them', expanded => {
    const plain = moduleNode('root', undefined, { expanded });
    const changed = moduleNode('root', undefined, { expanded, diffStatus: 'modified' });
    const document = new DOMParser().parseFromString(graphToSvg([changed], []), 'image/svg+xml');
    const [frame, header] = Array.from(document.querySelectorAll('g rect'));
    expect(frame.getAttribute('stroke')).toBe('#f5c778');
    expect(frame.hasAttribute('stroke-dasharray')).toBe(false);
    expect(header.getAttribute('fill')).toBe('#f5c778');
    expect(header.getAttribute('fill-opacity')).toBe('0.12');
    expect(graphSvgBounds([changed])).toEqual(graphSvgBounds([plain]));
    expect(graphToSvg([plain], [])).not.toContain('#f5c778');
  });

  it('keeps a failing module red even when it also has source changes', () => {
    const node = moduleNode('root', undefined, { diffStatus: 'modified', moduleDiagnosticStatus: 'error' });
    const document = new DOMParser().parseFromString(graphToSvg([node], []), 'image/svg+xml');
    expect(document.querySelector('g rect')?.getAttribute('stroke')).toBe('#ff9298');
  });

  it('keeps ordinary and residual wires neutral so color remains reserved for state', () => {
    const a = layer('a', undefined, { x: 0, y: 0 });
    const b = layer('b', undefined, { x: 300, y: 0 });
    const svg = graphToSvg([a, b], [edge('wire', 'a', 'b'), edge('skip', 'a', 'b', { data: { role: 'residual' } })]);
    const document = new DOMParser().parseFromString(svg, 'image/svg+xml');
    const wires = Array.from(document.querySelectorAll('g path[marker-end]'));
    expect(wires.map(wire => wire.getAttribute('stroke'))).toEqual(['#636b64', '#636b64']);
    expect(wires[1].getAttribute('stroke-dasharray')).toBe('1 6');
    expect(svg).toContain('fill="#171817"');
    expect(svg).toContain('fill="#2a2c2a"');
  });

  it('omits hidden children', () => {
    const child = { ...layer('m__c', 'm', { x: 20, y: 60 }), hidden: true } as ModelNode;
    const svg = graphToSvg([moduleNode('m'), child], []);
    expect(svg).not.toContain('>m__c<');
  });

  it('escapes node labels', () => {
    const node = layer('a', undefined, { x: 0, y: 0 });
    node.data.layerType = { ...node.data.layerType, displayName: 'A & B <x>' };
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toContain('A & B <x>');
    expect(svg).toContain('&amp;');
    expect(svg).not.toContain('A & B <x>');
  });

  it.each([['aten.mean.dim', 'Mean'], ['aten.view.default', 'Reshape'], ['aten.bmm.default', 'Batch matrix multiply']])
  ('uses the readable %s tile name in overview and exported images', (target, title) => {
    const node = layer('captured', undefined, { x: 0, y: 0 });
    node.data.captureRef = { kind: 'operation' } as ModelNode['data']['captureRef'];
    node.data.captureLabel = target;
    node.data.layerType = { id: target, name: target, displayName: target } as ModelNode['data']['layerType'];
    node.data.parameters = {};
    node.data.outputShape = [1, 1280];
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toContain(title);
    expect(subtitleTexts(svg)).toContain(target);
    expect(svg).toContain('(1, 1280)');
  });

  it('renames legacy operations in exported images without changing graph data', () => {
    const node = layer('fx_42', undefined, { x: 0, y: 0 });
    const target = 'aten._native_batch_norm_legit_no_training.default';
    node.data.layerType = { id: target, name: target, displayName: target } as ModelNode['data']['layerType'];
    const before = JSON.stringify(node);
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toContain('Batch norm');
    expect(subtitleTexts(svg)).toContain(target);
    expect(JSON.stringify(node)).toBe(before);
  });

  it('draws an edge as a path between visible endpoints only', () => {
    const nodes = [layer('a', undefined, { x: 0, y: 0 }), layer('b', undefined, { x: 400, y: 0 })];
    const shown = graphToSvg(nodes, [edge('a-b', 'a', 'b')]);
    expect(shown).toContain('marker-end="url(#tv-arrow)"');
    const hiddenEdge = graphToSvg(nodes, [{ ...edge('a-b', 'a', 'b'), hidden: true } as ModelEdge]);
    expect(hiddenEdge).not.toContain('marker-end="url(#tv-arrow)"');
  });

  it('keeps tensor evidence while leaving the constructor list in selection details', () => {
    const node = layer('linear1', undefined, { x: 0, y: 0 });
    node.data.parameters = { in_features: 4, out_features: 2 };
    node.data.inputShape = [3, 4];
    node.data.outputShape = [3, 2];
    const svg = graphToSvg([node], []);
    expect(svg).not.toContain('in_features');
    expect(svg).not.toContain('out_features');
    expect(svg).toContain('>IN<');
    expect(svg).toContain('(3, 4)');
    expect(svg).toContain('>OUT<');
  });

  it('keeps ordinary source expressions in selection details', () => {
    const node = layer('slice', undefined, { x: 0, y: 0 });
    node.data.layerType = { id: 'narrow', name: 'Narrow' } as any;
    node.data.parameters = { dim: 1, start: 0, length: 16 };
    expect(graphToSvg([node], [])).not.toContain('x[');
  });

  it('wraps complete callable and class names in the same bounds used for hit testing', () => {
    const title = 'WNCausalConvolutionWithAdaptiveTemporalPadding';
    const callable = `video.audio.${title}`;
    const node = layer('custom', undefined, { x: 30, y: 40 });
    node.data.layerType = { ...node.data.layerType, id: 'custom_code', name: title, displayName: title };
    node.data.parameters = { _class: title, _raw_call: `${callable}(x)` };
    const svg = graphToSvg([node], []);
    const document = svgDocument(svg);
    expect(titleTexts(svg)).toEqual([title]);
    expect(subtitleTexts(svg)).toEqual([callable]);
    expect(document.querySelectorAll('[data-card-title] tspan').length).toBeGreaterThan(1);
    expect(svg).not.toContain('…');
    const bounds = graphSvgBounds([node])[0];
    const lineYs = Array.from(document.querySelectorAll('[data-card-title] tspan, [data-card-subtitle] tspan'), span => Number(span.getAttribute('y')));
    expect(Math.max(...lineYs)).toBeLessThan(bounds.y + bounds.h);
  });

  it('does not grow ordinary exported cards when a callable takes hundreds of parameters', () => {
    const base = layer('custom', undefined, { x: 0, y: 0 });
    base.data.layerType = { ...base.data.layerType, id: 'custom_code', name: 'MyBlock', displayName: 'MyBlock' };
    base.data.parameters = { _class: 'MyBlock' };
    const busy = { ...base, data: { ...base.data, parameters: { ...base.data.parameters,
      ...Object.fromEntries(Array.from({ length: 200 }, (_, index) => [`option_${index}`, index])) } } };
    expect(graphSvgBounds([busy])).toEqual(graphSvgBounds([base]));
    expect(graphToSvg([busy], [])).not.toContain('option_');
  });

  it('also preserves a complete custom module class and captured path', () => {
    const title = 'AsymmetricVideoMemoryWithConditionalTemporalScoring';
    const path = 'encoder.temporal_blocks.14.candidate_scorer';
    const node = moduleNode('custom-module', undefined, { expanded: false, moduleTemplateId: title, captureLabel: path });
    node.style = { width: 250, height: 300 };
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toEqual([title]);
    expect(subtitleTexts(svg)).toEqual([path]);
    expect(svg).not.toContain('…');
  });

  it('exports a header-only output without an invented blank body', () => {
    const node = { ...layer('out', undefined, { x: 0, y: 0 }), type: 'outputNode', data: {
      layerType: { id: 'output', name: 'Output' }, parameters: {}, captureLabel: 'To Module',
    } } as unknown as ModelNode;
    const svg = graphToSvg([node], []);
    expect(titleTexts(svg)).toEqual(['Output']);
    expect(subtitleTexts(svg)).toEqual(['To Module']);
    expect(graphSvgBounds([node])[0].h).toBeLessThan(80);
    expect(svg).not.toContain('Not recorded');
  });

  it('draws a tick per declared port', () => {
    const node = layer('n', undefined, { x: 0, y: 0 });
    node.data.outputPorts = ['p1', 'p2'];
    // two port ticks; a convolution's family glyph draws no circles
    expect((graphToSvg([node], []).match(/<circle /g) ?? []).length).toBe(2);
  });

  it('colours each family glyph without letting status recolor it', () => {
    const conv = layer('conv', undefined, { x: 0, y: 0 });
    const failed = { ...layer('bn', undefined, { x: 300, y: 0 }), data: { layerType: { id: 'batchnorm2d', name: 'BatchNorm2d' }, parameters: {}, error: 'Shape mismatch' } } as unknown as ModelNode;
    const ink = glyphs(graphToSvg([conv, failed], [], { theme: 'dark' }));
    expect(ink.map(glyph => [glyph.getAttribute('data-block'), stroke(glyph)]))
      .toEqual([['convolution', '#a8b9c7'], ['normalization', '#aabdae']]);
    // Ink tints the tile with 16% of the family colour, as the card does.
    expect(tile(ink[0])).toBe('#434848');
    // The card's own lucide grid: one frame and four rules.
    expect(Array.from(ink[0].querySelector('g')!.children, child => child.tagName)).toEqual(['rect', 'path', 'path', 'path', 'path']);
    const paper = glyphs(graphToSvg([conv], [], { theme: 'light' }));
    expect([stroke(paper[0]), tile(paper[0])]).toEqual(['#6c91af', '#f6f6f4']);
    // The failure is the card's red outline, separate from its family.
    expect(graphToSvg([failed], [])).toContain('stroke="#ff9298"');
  });

  it('shows a module warning as its own status dot beside a neutral composite glyph', () => {
    const svg = graphToSvg([moduleNode('m', undefined, { moduleStatus: 'warning' })], []);
    expect([stroke(glyphs(svg)[0]), tile(glyphs(svg)[0])]).toEqual(['#848d85', '#303230']);
    expect(svg).toContain('r="4" fill="#f5c778"');
  });

  it('keeps the exported family palette identical to the Canvas tokens', () => {
    const css = require('fs').readFileSync(require('path').join(__dirname, '../../styles/canvas.tokens.css'), 'utf8') as string;
    const block = (selector: string) => css.slice(css.indexOf(selector), css.indexOf('}', css.indexOf(selector)));
    const tokens = (text: string) => Object.fromEntries(Array.from(text.matchAll(/--tv-family-([a-z]+): (#[0-9a-f]{6});/g), match => [match[1], match[2]]));
    const { svgFamilyPalettes } = require('../../services/graphSvg');
    expect(tokens(block(':root {'))).toEqual(svgFamilyPalettes.dark);
    expect(tokens(block(":root[data-tv-theme='light'] {"))).toEqual(svgFamilyPalettes.light);
  });

  it('reports absolute bounds for hit-testing and skips hidden nodes', () => {
    const child = layer('m__c', 'm', { x: 20, y: 60 });
    const bounds = graphSvgBounds([moduleNode('m'), child, { ...child, id: 'gone', hidden: true } as ModelNode]);
    const module = bounds.find((box) => box.id === 'm')!;
    const inner = bounds.find((box) => box.id === 'm__c')!;
    expect(module).toMatchObject({ x: 100, y: 100, w: 400, h: 300 });
    expect(inner.x).toBe(120);
    expect(inner.y).toBe(160);
    expect(bounds.some((box) => box.id === 'gone')).toBe(false);
  });
});
