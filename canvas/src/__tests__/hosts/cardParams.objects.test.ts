import { cardParams } from '../../ide/cardParams';
import { IO_CARD_HEIGHT, ioCardHeight, layerCardHeight } from '../../canvas/cardMetrics';

/**
 * The first thing a user saw when opening a model with a `nn.ModuleList` was a card
 * reading `modules  [object Object],[object Object],[object Object],[object Object]`.
 *
 * The parser reads the constructor call and records what it found — for
 * `nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])` that is
 * `_modules: [{op: "Linear", params: {...}}, ×4]` — and the card drew it with
 * `String(value)`. It is a real and useful thing for the projection to carry; it is not a
 * row on a card.
 *
 * These pin both halves: structured values never reach the parameter list, and the one
 * that says something a user wants ("how many of what") comes back as a summary.
 *
 * The key carries the `_` prefix for a second reason the card cannot see. The same
 * parameter bag is what the backend hands to a torch constructor, minus `name` and every
 * `_`-prefixed key; while the list was stored under `modules` — `nn.ModuleList`'s own
 * first argument — it was the one key that survived, and opening a perfectly good model
 * on the canvas drew `ModuleList — build error: dict is not a Module subclass`.
 */
describe('a card draws parameters, not data structures', () => {
  it('keeps tensor shape and permutation tuples, plus unevaluated source options', () => {
    expect(cardParams({ dims: [0, 3, 1, 2], _source_arguments: { shape: '(1, M, M, dim)', '**kwargs': 'kwargs' } }).entries)
      .toEqual([['dims', [0, 3, 1, 2]], ['shape', '(1, M, M, dim)'], ['**kwargs', 'kwargs']]);
    expect(cardParams({ nested: [[1, 2]], modules: [{ op: 'Linear' }] }).entries).toEqual([]);
  });
  it('drops a list-valued parameter from the rows', () => {
    // A bare (unprefixed) key, so this measures `isDrawableValue` and not the `_` filter.
    const { entries } = cardParams({
      modules: [{ op: 'Linear' }, { op: 'Linear' }],
      _lineno: 18,
    });
    expect(entries.map(([key]) => key)).toEqual([]);
  });

  it('drops an object-valued parameter from the rows', () => {
    const { entries } = cardParams({ cfg: { hidden: 8 }, in_features: 8 });
    expect(entries).toEqual([['in_features', 8]]);
  });

  it('keeps every scalar a constructor can take', () => {
    const { entries } = cardParams({
      in_features: 8,
      bias: false,
      name: 'block',
      padding_mode: 'zeros',
      dilation: null,
    });
    expect(entries).toEqual([
      ['in_features', 8],
      ['bias', false],
      ['name', 'block'],
      ['padding_mode', 'zeros'],
      ['dilation', null],
    ]);
  });

  it('summarises a container whose members are all the same class', () => {
    const { contents, entries } = cardParams({
      _modules: [
        { op: 'Linear', params: { in_features: 16, out_features: 16 } },
        { op: 'Linear', params: { in_features: 16, out_features: 16 } },
        { op: 'Linear', params: { in_features: 16, out_features: 16 } },
        { op: 'Linear', params: { in_features: 16, out_features: 16 } },
      ],
      _lineno: 18,
    });
    expect(contents).toBe('4 × Linear');
    expect(entries).toEqual([]);
  });

  it('counts a mixed container without claiming a class', () => {
    const { contents } = cardParams({
      _modules: [{ op: 'Conv2d' }, { op: 'BatchNorm2d' }, { op: 'ReLU' }],
    });
    expect(contents).toBe('3 layers');
  });

  it('unqualifies a dotted class name', () => {
    const { contents } = cardParams({
      _modules: [{ op: 'torch.nn.Linear' }, { op: 'torch.nn.Linear' }],
    });
    expect(contents).toBe('2 × Linear');
  });

  it('says nothing about a container it cannot read', () => {
    expect(cardParams({ _modules: [] }).contents).toBeUndefined();
    expect(cardParams({ _modules: [{}, {}] }).contents).toBe('2 layers');
    expect(cardParams({ in_features: 8 }).contents).toBeUndefined();
  });

  it('reads the member list off the key no torch constructor is handed', () => {
    // `modules` is `nn.ModuleList`'s own first argument, so a member list under that
    // name is forwarded to the constructor by every backend consumer that filters on
    // the `_` prefix — which is how a correct model came back as a build error. The
    // card reads the prefixed key and nothing else, so the two spellings cannot drift
    // back together unnoticed.
    expect(cardParams({ modules: [{ op: 'Linear' }, { op: 'Linear' }] }).contents)
      .toBeUndefined();
    expect(cardParams({ _modules: [{ op: 'Linear' }, { op: 'Linear' }] }).contents)
      .toBe('2 × Linear');
  });

  it('does not badge a container as a repeat', () => {
    // `×4` on a card whose header says `ModuleList` reads as four ModuleLists. The badge
    // means "this card stands for N identical blocks", which is true of the folded member
    // card the trace produces, not of the container.
    const { repeat } = cardParams({ _modules: [{ op: 'Linear' }, { op: 'Linear' }] });
    expect(repeat).toBeUndefined();
  });

  it('still badges a folded repeat from the trace', () => {
    const { repeat, entries } = cardParams({
      in_features: 8,
      out_features: 8,
      _repeat_count: 4,
      _class: 'Linear',
    });
    expect(repeat).toBe('×4');
    expect(entries.map(([key]) => key)).toEqual(['in_features', 'out_features']);
  });

  it('keeps constructor values and member metadata out of compact card geometry', () => {
    const shapes = { inputShape: [1, 16], outputShape: [1, 16] };
    const plain = layerCardHeight({ parameters: { in_features: 16, out_features: 16 }, ...shapes });
    const container = layerCardHeight({
      parameters: {
        in_features: 16,
        out_features: 16,
        _modules: [{ op: 'Linear' }, { op: 'Linear' }],
      },
      ...shapes,
    });
    expect(container).toBe(plain);
    const opaque = layerCardHeight({
      parameters: { in_features: 16, out_features: 16, cfg: { hidden: 8 } },
      ...shapes,
    });
    expect(opaque).toBe(plain);
  });

  it('reserves only the actual input/output rows, including input provenance', () => {
    expect(ioCardHeight({ parameters: { shape: '2,8', _input_provenance: 'declared' } }))
      .toBe(IO_CARD_HEIGHT);
    expect(ioCardHeight({ parameters: { name: 'Output' } }, 'output')).toBe(IO_CARD_HEIGHT);
    expect(ioCardHeight(undefined, 'output')).toBe(53);
    expect(ioCardHeight(undefined)).toBe(IO_CARD_HEIGHT);
    expect(ioCardHeight({ parameters: { shape: '1,8', _input_provenance: 'seed-from-first-layer' } }))
      .toBe(106);
  });
});
