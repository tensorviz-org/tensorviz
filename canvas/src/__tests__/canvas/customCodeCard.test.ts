/**
 * The custom-code card (task.md M6).
 *
 * `custom_code` and `python` are the two ops the registry deliberately does not
 * model, so their cards are drawn from provenance keys rather than an OpSpec.
 * Two things are worth pinning, and they are the two that break silently:
 *
 * 1. What the card SAYS. A class node has to name its class and where it is
 *    defined, because "defined in models/blocks.py:5" is the whole reason a graph
 *    of somebody's own code is better than a graph of nn.* — and a python node's
 *    identity is the statement itself, which nothing else on the canvas carries.
 * 2. That cardMetrics counts every row the component draws. tldraw's selection
 *    box, resize handles and hit area come from this number, so a row rendered
 *    and not counted is a card whose lower half cannot be clicked, and a card
 *    that overlaps the next one in the auto-layout.
 */
import { customCardModel } from '../../canvas/customCard';
import { layerCardHeight } from '../../canvas/cardMetrics';

const BLOCK = {
  _class: 'MyBlock',
  _import_path: 'models.blocks',
  _source_file: 'models/blocks.py',
  _source_line: 5,
  _raw_call: 'h = self.blk(h)',
  _opaque: true,
};

describe('what a custom-code card says', () => {
  it('names the class and where it is defined', () => {
    const card = customCardModel('custom_code', BLOCK);
    expect(card).not.toBeNull();
    expect(card!.title).toBe('MyBlock');
    expect(card!.snippet).toBe('h = self.blk(h)');
    expect(card!.location).toBe('models/blocks.py:5');
  });

  it('names what a python node does, with the statement underneath', () => {
    // Changed deliberately: the title used to quote the statement. On real code
    // that reads badly — a column of `q.view(B, T, ...`, `k.view(B, T, ...` is
    // three truncated quotes that are hard to tell apart. The operation is the
    // useful name; the statement is the detail, so it is no longer a repeat.
    const card = customCardModel('python', {
      _raw_call: 'x = x.transpose(1, 2)',
      _source_result_call: { name: 'transpose' },
      _decl_file: 'models/resid.py',
      _lineno: 24,
    });
    expect(card!.title).toBe('transpose');
    expect(card!.snippet).toBe('x = x.transpose(1, 2)');
    expect(card!.location).toBe('models/resid.py:24');
  });

  it('never offers the retired retrace expansion control', () => {
    expect(customCardModel('custom_code', BLOCK)).not.toHaveProperty('canExpand');
  });

  it('is null for a modelled layer, so every other card is untouched', () => {
    expect(customCardModel('linear', { in_features: 8 })).toBeNull();
    expect(customCardModel(undefined, undefined)).toBeNull();
  });

  it('keeps a long statement to one line of a ~200px card', () => {
    const card = customCardModel('python', {
      _raw_call: 'x = torch.cat([left_branch(x), right_branch(x), residual_tail(x)], dim=-1)',
      _source_result_call: { name: 'cat', target: 'torch.cat' },
    });
    // The title is now the operation, which is short by construction; the
    // statement moved to the snippet and is what still needs bounding.
    //
    // The branches prepare arguments; the result combines them.
    expect(card!.title).toBe('Concatenate');
    expect(card!.snippet.length).toBeLessThanOrEqual(60);
  });

  it('a statement that quotes itself never says the same thing twice', () => {
    // The guarantee the old title/snippet split existed to give, kept: whichever
    // way round they land, a card must not spend a row repeating its own heading.
    const named = customCardModel('python', { _raw_call: 'x = x.softmax(-1)' })!;
    expect(named.title).not.toBe(named.snippet);

    const unnamed = customCardModel('python', { _raw_call: 'x = x + 1' })!;
    expect(unnamed.snippet).toBe('');
  });
});

describe('cardMetrics keeps ordinary custom cards compact', () => {
  const custom = { layerType: { id: 'custom_code' }, parameters: BLOCK };

  it('keeps internal metadata and constructor arguments out of the card footprint', () => {
    const height = (parameters: Record<string, unknown>) => layerCardHeight({
      layerType: { id: 'linear', name: 'Linear' }, parameters,
      inputShape: [2, 8], outputShape: [2, 8],
    });
    const baseline = { ...BLOCK, in_features: 8, out_features: 8 };
    expect(height({ ...baseline, _params_M: 12, _source_digest: 'revision' })).toBe(height(baseline));
    expect(height({ ...baseline, uncommon_option: 128 })).toBe(height(baseline));
  });

  it('keeps source snippets and locations in inspection details without adding card rows', () => {
    const shaped = (parameters: Record<string, unknown>) => layerCardHeight({
      layerType: { id: 'custom_code' }, parameters,
      inputShape: [2, 8], outputShape: [2, 8],
    });
    const named = { _class: 'MyBlock', _raw_call: 'h = self.blk(h)' };
    const located = { ...named, _source_file: 'models/blocks.py', _source_line: 5 };
    expect(shaped(located)).toBe(shaped(named));
    expect(shaped({ ...located, _raw_call: 'h = self.blk(h, context=question)' })).toBe(shaped(located));

    const asWeb = shaped(BLOCK);
    (window as { __TENSORVIZ_IDE__?: boolean }).__TENSORVIZ_IDE__ = true;
    try { expect(shaped(BLOCK)).toBe(asWeb); }
    finally { delete (window as { __TENSORVIZ_IDE__?: boolean }).__TENSORVIZ_IDE__; }
  });

  it('does not grow a custom card when its constructor has hundreds of arguments', () => {
    const parameters = { ...BLOCK, ...Object.fromEntries(Array.from({ length: 200 }, (_, index) => [`argument_${index}`, index])) };
    const original = JSON.stringify(parameters);
    expect(layerCardHeight({ ...custom, parameters })).toBe(layerCardHeight(custom));
    expect(JSON.stringify(parameters)).toBe(original);
  });

  it('uses a header-only footprint when no summary facts are available', () => {
    expect(layerCardHeight({})).toBe(53);
    const ordinary = { layerType: { id: 'linear', name: 'Linear' } };
    expect(layerCardHeight(ordinary)).toBe(53);
    // A known feature pair is a compact fact, rather than an arbitrary argument list.
    expect(layerCardHeight({ ...ordinary, parameters: { in_features: 8, out_features: 4 } })).toBeGreaterThan(layerCardHeight(ordinary));
  });
});

describe('an opaque python card names the operation, not the source', () => {
  it('explains the outer selection instead of listing argument preparation', () => {
    const card = customCardModel('python', {
      _raw_call: 'torch.where(masks.unsqueeze(-1), token.to(x.dtype).unsqueeze(0), x)',
      _source_calls: ['unsqueeze', 'to', 'unsqueeze', 'where'],
      _source_result_call: { name: 'where', target: 'torch.where' },
    })!;
    expect(card.title).toBe('Select by condition');
    expect(card.callable).toBe('torch.where');
    expect(card.snippet).toContain('torch.where(');
  });
  it('does not assign Torch meaning to arbitrary methods with the same name', () => {
    expect(customCardModel('python', { _raw_call: 'query.where(x.clone())',
      _source_calls: ['clone', 'where'], _source_result_call: { name: 'where' } })!.title).toBe('where');
  });
  it('summarises a chained tensor call', () => {
    // nanoGPT line 58. Quoting this truncates to `q.view(B, T, ...`, which is
    // indistinguishable from the k and v lines above and below it.
    const card = customCardModel('python', {
      _raw_call: 'q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)',
      _source_calls: ['view', 'transpose'],
      _source_result_call: { name: 'transpose' },
    })!;
    expect(card.title).toBe('transpose');
    expect(card.snippet).toContain('q.view(B, T');
  });

  it('names a single operation', () => {
    const card = customCardModel('python', {
      _raw_call: "att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))",
      _source_result_call: { name: 'masked_fill' },
    })!;
    expect(card.title).toBe('masked_fill');
  });

  it('keeps the result operation visible even with many preparatory calls', () => {
    const card = customCardModel('python', {
      _raw_call: 'y = a.view(1).transpose(0, 1).contiguous().reshape(2).sum()',
      _source_calls: ['view', 'transpose', 'contiguous', 'reshape', 'sum'],
      _source_result_call: { name: 'sum' },
    })!;
    expect(card.title).toBe('sum');
  });

  it('falls back to the source when nothing is called', () => {
    const card = customCardModel('python', { _raw_call: 'x = x + 1' })!;
    expect(card.title).toBe('x = x + 1');
    expect(card.snippet).toBe('');
  });

  it('uses the proven result call and falls back to source for old snapshots', () => {
    const params = { _raw_call: 'F.normalize(student_output.float(), dim=-1)' };
    expect(customCardModel('python', { ...params, _source_calls: ['float', 'normalize'],
      _source_result_call: { name: 'normalize', target: 'torch.nn.functional.normalize' } })!.title).toBe('Normalize');
    expect(customCardModel('python', params)!.title).toMatch(/^F\.normalize\(student_output\.float\(\)/);
  });

  it('ignores a bare name that merely looks like a call', () => {
    const card = customCardModel('python', { _raw_call: 'view = 3' })!;
    expect(card.title).toBe('view = 3');
  });
});

describe('a project-graph card explains itself', () => {
  const DATASET = {
    _kind: 'dataset',
    _label: 'CharCorpus',
    _reason: 'subclasses Dataset',
    _provenance: 'static',
    _source_file: 'data/char_dataset.py',
    _lineno: 26,
  };

  it('is named for the thing, not for "Custom Code"', () => {
    // Every research node rendered as an unlabelled box before this: the card
    // model is handed `data.parameters` and nothing else, and the keys were
    // placed beside it.
    const card = customCardModel('custom_code', DATASET)!;
    expect(card.title).toBe('CharCorpus');
  });

  it('says which rule classified it, so a wrong guess can be spotted', () => {
    const card = customCardModel('custom_code', DATASET)!;
    expect(card.kind).toBe('dataset');
    expect(card.reason).toBe('subclasses Dataset');
    expect(card.snippet).toBe('subclasses Dataset');
  });

  it('carries provenance so a guess draws differently from a fact', () => {
    expect(customCardModel('custom_code', DATASET)!.provenance).toBe('static');
    expect(
      customCardModel('custom_code', { ...DATASET, _provenance: 'inferred' })!.provenance,
    ).toBe('inferred');
    // Absent means read from the code, not "unknown": the project tier always
    // states it, and the per-class tier has no research provenance to state.
    expect(customCardModel('custom_code', { _kind: 'model', _label: 'Net' })!.provenance).toBe(
      'static',
    );
  });

  it('points at the line that declares it', () => {
    expect(customCardModel('custom_code', DATASET)!.location).toBe('data/char_dataset.py:26');
  });

  it('falls back to the kind when a node somehow has no label', () => {
    expect(customCardModel('custom_code', { _kind: 'trainer' })!.title).toBe('trainer');
  });

  it('never offers expand-in-place — a research node has no traced inside', () => {
    expect(customCardModel('custom_code', DATASET)).not.toHaveProperty('canExpand');
  });

  it('leaves the per-class custom-code card exactly as it was', () => {
    const card = customCardModel('custom_code', {
      _class: 'Block',
      _raw_call: 'h = self.blk(h)',
      _decl_file: 'models/blocks.py',
      _lineno: 5,
    })!;
    expect(card.title).toBe('Block');
    expect(card.kind).toBeUndefined();
    expect(card).not.toHaveProperty('canExpand');
  });
});

/**
 * The Validate row (step 5.9).
 *
 * The card has to answer "can this be checked?" before the button is pressed,
 * because a refusal that arrives afterwards has already wasted the press. And the
 * row has to be counted by cardMetrics for the reason at the top of this file.
 */
const RESEARCH = {
  _kind: 'model',
  _label: 'GPT',
  _reason: 'subclasses nn.Module',
  _provenance: 'static',
  _source_file: 'model.py',
  _source_line: 118,
};

describe('the Validate row', () => {
  it('carries the index verdict onto the card', () => {
    const card = customCardModel('custom_code', {
      ...RESEARCH,
      _validatable: {
        state: 'needs',
        detail: 'GPT needs `config`, and this project declares one: GPTConfig.',
        remedy: 'tensorviz.init("GPT", config="GPTConfig")',
      },
    });
    expect(card!.validatable).toEqual({
      state: 'needs',
      detail: 'GPT needs `config`, and this project declares one: GPTConfig.',
      remedy: 'tensorviz.init("GPT", config="GPTConfig")',
    });
  });

  it('ignores a malformed verdict rather than blanking the card', () => {
    // The card is drawn from whatever the projection happened to send. A verdict
    // with a state nobody defined must cost the reader nothing.
    const card = customCardModel('custom_code', { ...RESEARCH, _validatable: { state: 'maybe' } });
    expect(card!.validatable).toBeUndefined();
    expect(card!.title).toBe('GPT');
  });

  it('an unsupported node draws no row, so the height counts none', () => {
    // A dataset has no forward pass. A row saying so on every such card is noise
    // that teaches the reader to stop reading the row.
    const unsupported = {
      ...RESEARCH,
      _kind: 'dataset',
      _validatable: { state: 'unsupported', detail: 'A dataset has no forward pass to check.', remedy: '' },
    };
    const withoutVerdict = { ...RESEARCH, _kind: 'dataset' };
    expect(
      layerCardHeight({ layerType: { id: 'custom_code' }, parameters: unsupported } as never),
    ).toBe(
      layerCardHeight({ layerType: { id: 'custom_code' }, parameters: withoutVerdict } as never),
    );
  });

  it('keeps validation availability in inspection without growing an ordinary card', () => {
    const ready = {
      ...RESEARCH,
      _validatable: { state: 'ready', detail: '', remedy: '' },
    };
    const taller = layerCardHeight({
      layerType: { id: 'custom_code' },
      parameters: ready,
    } as never);
    const plain = layerCardHeight({
      layerType: { id: 'custom_code' },
      parameters: RESEARCH,
    } as never);
    expect(taller).toBe(plain);
  });

  it('keeps validation refusal details out of the ordinary card footprint', () => {
    const needs = {
      ...RESEARCH,
      _validatable: { state: 'needs', detail: 'GPT needs `config`.', remedy: 'x' },
    };
    const ready = { ...RESEARCH, _validatable: { state: 'ready', detail: '', remedy: '' } };
    expect(
      layerCardHeight({ layerType: { id: 'custom_code' }, parameters: needs } as never),
    ).toBe(
      layerCardHeight({ layerType: { id: 'custom_code' }, parameters: ready } as never),
    );
  });
});

describe('opening a project frame', () => {
  it('offers Expand for a node that has an inside', () => {
    const card = customCardModel('custom_code', { ...RESEARCH, _expandable: true });
    expect(card!.projectExpandable).toBe(true);
  });

  it('is not the same affordance as tracing a level deeper', () => {
    const card = customCardModel('custom_code', { ...RESEARCH, _expandable: true });
    expect(card).not.toHaveProperty('canExpand');
  });

  it('a node with nothing inside offers nothing, and measures nothing', () => {
    const card = customCardModel('custom_code', RESEARCH);
    expect(card!.projectExpandable).toBe(false);
    expect(
      layerCardHeight({
        layerType: { id: 'custom_code' },
        parameters: { ...RESEARCH, _expandable: true },
      } as never),
    ).toBeGreaterThan(
      layerCardHeight({ layerType: { id: 'custom_code' }, parameters: RESEARCH } as never),
    );
  });
});

describe('a per-class card names its class, not the word "class"', () => {
  // Reported on a real model: F5AECStftXAttn drew three identical cards reading
  // "class" — nn.ModuleList([CrossAttnBlock(...)]), RotaryEmbedding(...) and
  // nn.ModuleList([DiTBlock(...)]). Every per-class custom node carries
  // `_kind: 'class'` and no `_label`, so the generic word beat `_class`, which held
  // the real name all along.
  it('falls back to the class name before the generic kind', () => {
    const card = customCardModel('custom_code', { _kind: 'class', _class: 'RotaryEmbedding' });
    expect(card!.title).toBe('RotaryEmbedding');
  });

  it('names what a uniform container actually holds', () => {
    const card = customCardModel('custom_code', {
      _kind: 'class',
      _class: 'ModuleList',
      _modules: [{ op: 'CrossAttnBlock', params: {} }, { op: 'CrossAttnBlock', params: {} }],
    });
    expect(card!.title).toBe('CrossAttnBlock ×2');
  });

  it('names a single-entry container without a count', () => {
    const card = customCardModel('custom_code', {
      _kind: 'class', _class: 'Sequential', _modules: [{ op: 'Block', params: {} }],
    });
    expect(card!.title).toBe('Block');
  });

  it('does not name one arbitrary member of a mixed container', () => {
    const card = customCardModel('custom_code', {
      _kind: 'class',
      _class: 'Sequential',
      _modules: [{ op: 'Conv1d', params: {} }, { op: 'Mish', params: {} }],
    });
    expect(card!.title).toBe('Sequential');
  });

  it('still titles a research card by its role', () => {
    expect(customCardModel('custom_code', { _kind: 'trainer' })!.title).toBe('trainer');
    expect(customCardModel('custom_code', { _kind: 'model', _label: 'Net' })!.title).toBe('Net');
  });
});
