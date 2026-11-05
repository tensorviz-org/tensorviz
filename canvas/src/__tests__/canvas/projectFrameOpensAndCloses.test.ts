/**
 * A model frame the indexer built must open, say what it is, and close again.
 *
 * Measured in the running extension, on `examples/tensorviz-test`: pressing "Show
 * contents" on the TinyTransformer card opened a frame titled **module**, reading
 * **module · 0 layers**, with a green **Healthy** pill and a Collapse button that
 * did nothing. The document behind it was entirely correct — `moduleNode`, six
 * children, `data.label` "TinyTransformer" — and every visible part of it was
 * wrong, because `moduleNode` is the per-class canvas's frame and reads three
 * fields of its own that the project document never filled:
 *
 *   data.moduleTemplateId  → the title            (fell back to 'module')
 *   data.moduleLayerCount  → the "N layers" line  (fell back to 0)
 *   data.moduleInstanceId  → gates Collapse       (fell back to '', so it returned)
 *
 * The backend half is `test_research_document.py`. This is the canvas half: that
 * the frame goes back the way it came, and that it stops claiming things the
 * project graph never computed.
 *
 * A source-contract test, in the technique this repo already uses for the canvas:
 * ModuleShapeUtil imports tldraw and cannot be mounted under the CRA jest
 * transform.
 */
import fs from 'fs';
import path from 'path';
import { nodePresentation } from '../../canvas/nodePresentation';
import type { ModelNodeData } from '../../types';

const MODULE_SHAPE = path.join(__dirname, '../..', 'canvas', 'ModuleShapeUtil.tsx');
const source = fs.readFileSync(MODULE_SHAPE, 'utf8');

describe('a project frame closes the way it opened', () => {
  it('recognises itself from the one channel the project document has', () => {
    // `_kind` rides inside `data.parameters` because that is the only per-node bag
    // the card model is handed — a key placed beside it reaches the canvas and
    // then vanishes. research/document.py::_card_data states the same rule.
    expect(source).toContain(
      "const projectKind = typeof data.parameters?._kind === 'string' ? data.parameters._kind : '';",
    );
  });

  it('keeps class and portable module expansion local despite shared kind metadata', () => {
    expect(source).toContain("const isProjectFrame = showingProjectGraph() && projectKind !== '';");
  });

  it('collapses through the project channel, not through a callback it does not have', () => {
    // `onToggleExpand` is supplied by the per-class canvas for a module the user
    // grouped. A frame the indexer built has no module instance, so the handler
    // returned on its first line and the frame could never be closed.
    const start = source.indexOf('const toggleExpand = canvasButtonHandlers(');
    expect(start).toBeGreaterThan(-1);
    const handler = source.slice(start, start + 420);
    expect(handler).toContain('if (isProjectFrame) {');
    expect(handler).toContain('reportProjectExpand(shapeIdToNodeId(shape.id), !isExpanded)');
    // …and the per-class path is still there for the canvas that does have one.
    expect(handler).toContain('data.onToggleExpand(moduleInstanceId)');
  });

  it('claims no health it never checked, and no layout it cannot reset', () => {
    // "Healthy" is a shape-inference verdict; the project graph runs no inference,
    // so the pill asserted a clean bill of health nothing had looked at. Reset
    // layout and Ungroup drive callbacks and graph edits a read-only project
    // frame does not have.
    expect(source).toMatch(/\{!isProjectFrame && \(\s*statusBadge\s*\)\}/);
    expect(source).toContain('const canUngroup = !localBrowserMode() && !isProjectFrame && !graphIsReadOnly(editor);');
    const guardedButtons = (text: string) => text.match(/\{!isProjectFrame && contents\.canExpand && \(\s*(?:<CanvasTooltip\b[^>]*>\s*)?<button\s*className="expand-button"/g) || [];
    expect(guardedButtons(source)).toHaveLength(2);
    expect(guardedButtons(source.replaceAll('!isProjectFrame &&', 'true &&'))).toHaveLength(0);
  });
});


describe('the outline a reader sees is the card’s own', () => {
  const canvas = fs.readFileSync(path.join(__dirname, '../..', 'canvas', 'TldrawCanvas.tsx'), 'utf8');
  const layer = fs.readFileSync(path.join(__dirname, '../..', 'canvas', 'LayerShapeUtil.tsx'), 'utf8');
  const tokens = fs.readFileSync(path.join(__dirname, '../..', 'styles', 'ui2.tokens.css'), 'utf8');

  it('hides tldraw’s hover outline, and each card opts out of the selection box', () => {
    // Both are drawn from the shape's STORED box, which a card's rendered height
    // only approximates. Three commits nulled component keys this tldraw does not
    // have; the real levers are the indicator element and the per-shape opt-outs
    // the selection overlay util honours.
    const theme = fs.readFileSync(path.join(__dirname, '../..', 'canvas', 'tldraw-theme.css'), 'utf8');
    expect(theme).toContain('.tldraw-canvas-wrapper .tl-shape-indicator { display: none; }');
    for (const util of [layer, source]) {
      expect(util).toContain('override hideSelectionBoundsFg() {');
      expect(util).toContain('override hideResizeHandles() {');
    }
    expect(canvas).not.toMatch(/ShapeIndicators?: null/);
  });

  it('brightens the card’s own border on hover instead', () => {
    expect(layer).toContain("const isHovered = editor.getHoveredShapeId() === shape.id;");
    expect(layer).toContain("${isHovered ? ' hovered' : ''}");
    expect(source).toContain("${isHovered ? ' hovered' : ''}");
    expect(tokens).toContain('.ui2-reactflow-skin .ui2-node-card.hovered {');
  });

  it('keeps the Expand row exactly one row tall, so the box and the card agree', () => {
    // The first version added padding, a margin and a border; the card grew ~10px
    // and the shape's stored height did not.
    const start = tokens.indexOf('.ui2-reactflow-skin .ui2-node-expand-row {');
    const rule = tokens.slice(start, tokens.indexOf('}', start));
    expect(rule).toContain('height: 16.5px;');
    expect(rule).toContain('margin: 0;');
    expect(rule).toContain('padding: 0 6px;');
    expect(rule).not.toMatch(/\bborder:/);
  });

  it('titles a child from the document when the catalogue has no name for its op', () => {
    // `ModuleList` has no catalogue display name, and that card came out blank.
    const data = { layerType: { id: 'modulelist', name: 'ModuleList', displayName: '' }, parameters: {}, label: 'Encoder stages' } as unknown as ModelNodeData;
    expect(nodePresentation(data, 'child').title).toBe('Encoder stages');
    expect(nodePresentation({ ...data, layerType: { ...data.layerType, displayName: 'ModuleList' } }, 'child').title).toBe('ModuleList');
  });

  it('never titles a step "Custom Code" when the document named it', () => {
    // A function frame's step is op custom_code with no `_class`, so the custom
    // branch answers its placeholder. The step's name is the whole reason to draw
    // it: `F.cross_entropy → loss`.
    const data = { layerType: { id: 'custom_code', name: 'Custom Code', displayName: 'Custom Code' }, parameters: {}, label: 'F.cross_entropy → loss' } as unknown as ModelNodeData;
    expect(nodePresentation(data, 'function.step').title).toBe('F.cross_entropy → loss');
    expect(nodePresentation({ ...data, parameters: { _class: 'Criterion' } }, 'function.step').title).toBe('Criterion');
  });
});
