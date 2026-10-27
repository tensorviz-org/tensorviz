/**
 * A shape on a research card must be MEASURED, never invented.
 *
 * Measured in the running extension: every card on the project canvas — including
 * `token_accuracy`, a metric function, and `save_checkpoint`, which writes a file —
 * displayed `In (1, 3, 224, 224)` and `Out (1, 3, 224, 224)`. That is the default
 * image shape, stamped by the per-class shape inference over a graph of classes,
 * functions and files. None of them has a forward pass.
 *
 * Refused where it is COMPUTED, not where it is drawn. The first version of this
 * fix suppressed In/Out on every research card, which also hid the shapes Validate
 * had just traced — throwing away the answer along with the guess. A card that has
 * been measured shows what was measured.
 */
import fs from 'fs';
import path from 'path';

import { layerCardHeight } from '../../canvas/cardMetrics';

const read = (...parts: string[]) => fs.readFileSync(path.join(__dirname, '../..', ...parts), 'utf8');

describe('shape inference does not run over a project graph', () => {
  it('asks the nodes, not only the global flag', () => {
    // `showingProjectGraph()` is global and set by whichever projection arrived
    // last: opening any `.py` publishes a class projection and flips it false while
    // a project is still drawn. So the guard also asks what is on the canvas.
    const builder = read('components', 'ModelBuilder.tsx');
    const start = builder.indexOf('const triggerShapeInference = useCallback(');
    const guard = builder.slice(start, start + 1400);
    expect(guard).toContain('showingProjectGraph()');
    expect(guard).toContain("node.data?.parameters?._kind !== undefined");
  });
});

describe('a measured shape is still drawn', () => {
  const shapes = { inputShape: [1, 10], outputShape: [1, 10, 16] };

  it('draws In/Out on a research card that has been traced', () => {
    // Validate runs the class and the numbers are real. This is the regression the
    // first version of the fix caused: pressing Validate produced shapes and the
    // card refused to show them.
    const source = read('canvas', 'LayerShapeUtil.tsx');
    expect(source).toContain('const measuredRows = Boolean(data.inputShape || data.outputShape);');
    expect(source).not.toContain('!custom?.kind && Boolean(data.inputShape');
  });

  it('reserves the height for them, or the box is shorter than its content', () => {
    const params = { _kind: 'model', _label: 'TinyLM', bases: 'nn.Module', call: 'subclasses nn.Module' };
    const measured = layerCardHeight({ layerType: { id: 'custom_code' }, parameters: params, ...shapes });
    const bare = layerCardHeight({ layerType: { id: 'custom_code' }, parameters: params });
    // A bare card sits on the 120px floor; two measured rows have to lift it off.
    expect(bare).toBe(120);
    expect(measured).toBeGreaterThan(120);
  });
});
