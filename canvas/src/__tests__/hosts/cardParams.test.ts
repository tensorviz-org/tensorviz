import { cardParams, inputProvenance, inputProvenanceNote, shapeProvenance } from '../../ide/cardParams';
import { headerTitle, UNTITLED } from '../../ide/header';

/**
 * task.md P0's half-day hygiene, as tests: the three things the owner sees every hour.
 *
 * Each case here is a defect carried in from Phase 1 (§3.8). They are small, and that is
 * exactly why they were never fixed — but they are on screen on every card of every graph,
 * which makes them the most-seen bugs in the product.
 */
describe('cardParams', () => {
  it('hides the provenance keys the projection attaches', () => {
    // These were drawn as if the user had passed them to nn.Linear. They are how the graph
    // knows what it knows, not something anybody wrote.
    const { entries } = cardParams({
      in_features: 784,
      out_features: 128,
      _shape_provenance: 'observed',
      _out_shape: [2, 128],
    });

    expect(entries).toEqual([
      ['in_features', 784],
      ['out_features', 128],
    ]);
  });

  it('turns _repeat_count into a badge instead of a parameter', () => {
    // `_repeat_count: 2` in the parameter list reads as a constructor argument nobody
    // wrote. What it actually means is that this card stands for two identical blocks.
    const { entries, repeat } = cardParams({ hidden: 8, _repeat_count: 2 });

    expect(entries).toEqual([['hidden', 8]]);
    expect(repeat).toBe('×2');
  });

  it('says nothing when a card stands for exactly one block', () => {
    // A badge on every card would say nothing while costing the space that makes the badge
    // legible where it does mean something.
    expect(cardParams({ hidden: 8, _repeat_count: 1 }).repeat).toBeUndefined();
    expect(cardParams({ hidden: 8 }).repeat).toBeUndefined();
  });

  it('reads a repeat count that arrived as a string', () => {
    expect(cardParams({ _repeat_count: '3' }).repeat).toBe('×3');
  });

  it('keeps parameter order, because it is the constructor’s order', () => {
    // The user reads a card against their own `nn.Linear(784, 128)` call. Sorting the keys
    // would make that comparison work rather than be obvious.
    const { entries } = cardParams({ c: 3, a: 1, b: 2 });
    expect(entries.map(([key]) => key)).toEqual(['c', 'a', 'b']);
  });

  it('survives a node with no parameters at all', () => {
    expect(cardParams(undefined)).toEqual({ entries: [] });
    expect(cardParams({})).toEqual({ entries: [] });
  });
});

describe('shapeProvenance', () => {
  it('reads the four values the worker reports and nothing else', () => {
    expect(shapeProvenance({ _shape_provenance: 'observed' })).toBe('observed');
    expect(shapeProvenance({ _shape_provenance: 'unreached' })).toBe('unreached');
    expect(shapeProvenance({ _shape_provenance: 'guessed' })).toBeUndefined();
    expect(shapeProvenance({})).toBeUndefined();
  });
});

describe('inputProvenance', () => {
  it('reads the five values the ingest tier reports and nothing else', () => {
    expect(inputProvenance({ _input_provenance: 'declared' })).toBe('declared');
    expect(inputProvenance({ _input_provenance: 'explicit' })).toBe('explicit');
    expect(inputProvenance({ _input_provenance: 'seed-from-first-layer' })).toBe(
      'seed-from-first-layer',
    );
    expect(inputProvenance({ _input_provenance: 'search-candidate' })).toBe('search-candidate');
    expect(inputProvenance({ _input_provenance: 'assumed-default' })).toBe('assumed-default');
    expect(inputProvenance({ _input_provenance: 'measured' })).toBeUndefined();
    expect(inputProvenance({})).toBeUndefined();
  });

  it('keeps the key off the drawn parameter list like every other provenance key', () => {
    // The card draws constructor arguments. `_input_provenance: seed-from-first-layer`
    // rendered as a row reads as something the user passed to the Input node.
    const { entries } = cardParams({ shape: '2,8', _input_provenance: 'declared' });
    expect(entries).toEqual([['shape', '2,8']]);
  });
});

describe('inputProvenanceNote', () => {
  it('a failed candidate cannot be described as surviving a forward', () => {
    const note = inputProvenanceNote({ _input_provenance: 'failed-candidate' });
    expect(note?.label).toBe('failed guess');
    expect(note?.detail).toContain('failed during the forward pass');
    expect(note?.detail).not.toContain('survived');
  });
  it('says nothing when a person chose the input', () => {
    // The defect this whole fact exists for is a guess that reads like a statement.
    // The statement itself is the normal case and must stay unremarkable.
    expect(inputProvenanceNote({ _input_provenance: 'declared' })).toBeUndefined();
    expect(inputProvenanceNote({ _input_provenance: 'explicit' })).toBeUndefined();
    // And a graph the ingest tier never reported on — every one read from the source
    // alone — has no input to have a provenance, so it gets no note either.
    expect(inputProvenanceNote({ shape: '1,3,224,224' })).toBeUndefined();
    expect(inputProvenanceNote(undefined)).toBeUndefined();
  });

  it('marks every kind of guess, and names which kind in the long form', () => {
    const seeded = inputProvenanceNote({ _input_provenance: 'seed-from-first-layer' });
    const searched = inputProvenanceNote({ _input_provenance: 'search-candidate' });
    const assumed = inputProvenanceNote({ _input_provenance: 'assumed-default' });

    expect(seeded?.label).toBe('guessed');
    expect(searched?.label).toBe('guessed');
    expect(assumed?.label).toBe('default');
    // The card has room for a word; the Validate verdict has room for the reason,
    // and the three reasons are genuinely different things to have happened.
    expect(seeded?.detail).toContain("first layer's width");
    expect(searched?.detail).toContain('survived a forward pass');
    expect(assumed?.detail).toContain("architecture's built-in default");
    expect(new Set([seeded?.detail, searched?.detail, assumed?.detail]).size).toBe(3);
    // Every one of them names the way out.
    for (const note of [seeded, searched, assumed]) {
      expect(note?.detail).toContain('# tensorviz: input=');
    }
  });
});

describe('headerTitle', () => {
  it('names the class in IDE mode, never "Untitled Model"', () => {
    // The bug: the canvas reported a graph as untitled while the editor two panes away was
    // showing the file and the class it came from.
    const { title, subtitle } = headerTitle({
      cls: 'Net',
      file: 'models/resid.py',
      graphName: UNTITLED,
      ide: true,
    });

    expect(title).toBe('Net');
    expect(subtitle).toBe('models/resid.py');
  });

  it('prefers a real stored name over the file, but never over the class', () => {
    expect(headerTitle({ cls: 'Net', graphName: 'My encoder', ide: true }).title).toBe('Net');
    expect(headerTitle({ file: 'models/resid.py', graphName: 'My encoder', ide: true }).title).toBe(
      'My encoder',
    );
  });

  it('falls back to the file, then to a generic word, rather than to the placeholder', () => {
    expect(headerTitle({ file: 'models/resid.py', graphName: UNTITLED, ide: true }).title).toBe(
      'models/resid.py',
    );
    expect(headerTitle({ graphName: UNTITLED, ide: true }).title).toBe('Graph');
  });

  it('leaves the web product alone', () => {
    // Outside the IDE a graph really is an untitled document until somebody names it, and
    // renaming that placeholder is not this milestone's business.
    expect(headerTitle({ graphName: UNTITLED, ide: false }).title).toBe(UNTITLED);
    expect(headerTitle({ cls: 'Net', graphName: UNTITLED, ide: false }).title).toBe(UNTITLED);
  });

  it('does not repeat the file as its own subtitle', () => {
    const { title, subtitle } = headerTitle({ file: 'models/resid.py', ide: true });
    expect(title).toBe('models/resid.py');
    expect(subtitle).toBeUndefined();
  });
});
