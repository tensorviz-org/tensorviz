/**
 * The parameter editor must show parameters, not internal render props.
 *
 * Selecting the Input node listed five editable rows: the two real ones (name,
 * shape) plus _accent (a CSS variable, "var(--cat-embedding)"), _subtitle ("Input")
 * and _shape ("[1, 16]"). _shape is the worst of the three — it looks like the field
 * you want, is editable, and reports success on Apply while changing nothing,
 * because the real field is `shape` and _shape is derived from inference.
 *
 * A behaviour test, not a source-text pin. This used to read BuilderPageV2.tsx as
 * text — a 1200-character window off `const selectedNodeIdForDraft`, and a
 * 900-character BACKWARD window off the first literal "Reset" in the file — because
 * BuilderPageV2 pulls in tldraw and cannot be mounted under the CRA jest transform.
 * Both windows asserted that a particular sentence was written, not that the panel
 * answers correctly, and either one broke on a reflow or on the word "Reset"
 * appearing anywhere earlier in the file. The rule now lives in
 * src/utils/inspectorDraft.ts, which both call sites use, and this asserts what it
 * ANSWERS.
 *
 * Measured in a browser afterwards: the Input node's panel went from
 *   [_accent, _shape, _subtitle, name, shape] to [name, shape],
 * with every other card's parameters unchanged.
 */
import fs from 'fs';
import path from 'path';
import { inspectorDraftParameters } from '../../utils/inspectorDraft';

describe('the parameter editor hides internal render props', () => {
  it('drops every underscore-prefixed key', () => {
    const draft = inspectorDraftParameters({
      name: 'input',
      shape: [1, 16],
      _accent: 'var(--cat-embedding)',
      _subtitle: 'Input',
      _shape: '[1, 16]',
    });
    expect(Object.keys(draft).sort()).toEqual(['name', 'shape']);
  });

  it('still shows real parameters, so the filter is on the prefix alone', () => {
    // A blanket allow-list would drop layer parameters the frontend schema does
    // not know. Nothing here is in any catalogue, and all of it must survive.
    const draft = inspectorDraftParameters({
      out_features: 64,
      bias: false,
      elementwise_affine: true,
      some_op_specific_knob: 'auto',
    });
    expect(Object.keys(draft).sort()).toEqual([
      'bias', 'elementwise_affine', 'out_features', 'some_op_specific_knob',
    ]);
  });

  it('formats what it keeps the way the editor reads it back', () => {
    // The draft is the text in the input box, so it goes through formatParamValue:
    // an Input shape in its canonical comma form, booleans in the Python spelling.
    expect(inspectorDraftParameters({ shape: [1, 3, 224, 224] }).shape).toBe('1,3,224,224');
    expect(inspectorDraftParameters({ bias: false }).bias).toBe('False');
  });

  it('answers with an empty draft for a node with no parameters', () => {
    // The panel seeds from the selected node's parameters, which are absent
    // between selections; an exception there blanks the whole inspector.
    expect(inspectorDraftParameters(undefined)).toEqual({});
    expect(inspectorDraftParameters(null)).toEqual({});
    expect(inspectorDraftParameters({})).toEqual({});
  });

  it('is the only copy of the rule, so Reset cannot re-seed the rows the fields hid', () => {
    // Seeding and the Reset chip were two hand-written copies of one expression,
    // and the old pin existed because a filter added to one could miss the other.
    // Kept as a source check because the editor mounts inside tldraw and cannot be
    // rendered under the CRA jest transform — but a whole-file regex, not a
    // character-offset window: it survives any reflow and fails only if a
    // hand-rolled copy comes back.
    //
    // The file moved when the Inspector was removed: the fields are now a popover
    // anchored to the node they edit, which is where both call sites live.
    const editor = fs.readFileSync(
      path.join(__dirname, '../..', 'canvas', 'NodeParametersPopover.tsx'),
      'utf8',
    );
    expect(editor).not.toMatch(/startsWith\(\s*'_'\s*\)/);
    expect(editor.match(/inspectorDraftParameters\(/g) || []).toHaveLength(2);
  });
});
