import * as assert from 'node:assert/strict';

import { autoOpenClass, declaresModule, moduleClasses, shouldStartBackend } from '../../activation';

/** P0 deleted the Models view's "Start tensorViz" row on the grounds that opening a folder with
 *  a model in it is the instruction. Nothing then started the backend, the class list comes
 *  from the backend, and a fresh window sat on "Indexing…" forever. This is the rule that
 *  decides a folder is worth starting one for, so it is the rule that has to be right. */
suite('activation.declaresModule', () => {
  test('the shapes a torch module is actually written in', () => {
    assert.ok(declaresModule('class Net(nn.Module):\n    pass\n'), 'nn.Module');
    assert.ok(declaresModule('class Net(Module):\n    pass\n'), 'a bare Module import');
    assert.ok(declaresModule('class Net(torch.nn.Module):\n    pass\n'), 'fully qualified');
    assert.ok(declaresModule('class Net(Base, nn.Module):\n    pass\n'), 'several bases');
    assert.ok(declaresModule('class Net( nn.Module ):\n    pass\n'), 'spaces inside the parens');
    assert.ok(
      declaresModule('if TYPE_CHECKING:\n    class Net(nn.Module):\n        pass\n'),
      'an indented class is ordinary and still counts',
    );
  });

  test('a file with no direct torch base does not use the auto-open shortcut', () => {
    assert.equal(declaresModule(''), false);
    assert.equal(declaresModule('x = 1\n'), false);
    assert.equal(declaresModule('class Config:\n    pass\n'), false);
    assert.equal(declaresModule('class Config(BaseModel):\n    pass\n'), false, 'pydantic is not torch');
    assert.equal(declaresModule('class Thing(ModuleFinder):\n    pass\n'), false, 'a longer name that merely starts with Module');
  });

  test('a commented-out class does not start a backend', () => {
    // The cheapest false positive to remove, and the one most likely to exist: people comment
    // a class out rather than delete it.
    assert.equal(declaresModule('# class Old(nn.Module):\n#     pass\n'), false);
    assert.equal(declaresModule('    # class Old(nn.Module):\n'), false);
  });

  test('prose that merely mentions the phrase does not start a backend', () => {
    assert.equal(declaresModule('"""Subclass nn.Module to define a model."""\n'), false);
  });
});

suite('activation.shouldStartBackend', () => {
  const base = {
    languageId: 'python',
    scheme: 'file',
    text: 'class Net(nn.Module):\n    pass\n',
    backendRunning: false,
    trusted: true,
    startFailed: false,
  };

  test('a python file with a module in it, in a trusted window, starts it', () => {
    assert.equal(shouldStartBackend(base), true);
  });

  test('plain JAX functions start static discovery with the same trust guards', () => {
    const text = 'import jax.numpy as jnp\ndef net(x): return jnp.sin(x)\n';
    assert.equal(shouldStartBackend({ ...base, text }), true);
    assert.equal(shouldStartBackend({ ...base, text, trusted: false }), false);
    assert.equal(shouldStartBackend({ ...base, text, scheme: 'git' }), false);
    assert.equal(shouldStartBackend({ ...base, text: 'def helper(x): return x\n' }), false);
  });

  test('cold discovery starts for imported and indirect bases without a literal nn.Module', () => {
    const sources = [
      'from transformers import PreTrainedModel\nclass NeoAudioQwenModel(SmartTurnMixin, PreTrainedModel):\n    pass\n',
      'from diffusers import ModelMixin as Base\nclass Model(Base):\n    pass\n',
      'from torch import nn as layers\nclass Model(layers.Module):\n    pass\n',
      'from .base import Parent\nclass Child(Parent):\n    pass\n',
      'class Model(\n    Mixin,\n    # an imported alias\n    Base,\n):\n    pass\n',
    ];
    for (const text of sources) {
      assert.deepEqual(moduleClasses(text), [], 'the direct-base shortcut cannot discover these models');
      assert.equal(shouldStartBackend({ ...base, text }), true, text);
      assert.equal(shouldStartBackend({ ...base, text }), true, 'a fresh host after reload must scan again');
      for (const guard of [
        { backendRunning: true }, { startFailed: true }, { trusted: false },
        { languageId: 'markdown' }, { scheme: 'git' },
      ]) {
        assert.equal(shouldStartBackend({ ...base, text, ...guard }), false);
      }
    }
  });

  test('the startup hint permits a scan without classifying an unknown base as a model', () => {
    const text = 'class Settings(BaseModel):\n    pass\n';
    assert.equal(shouldStartBackend({ ...base, text }), true);
    assert.deepEqual(moduleClasses(text), [], 'only source discovery may classify an imported base');
  });

  test('plain files, empty bases and commented-out subclasses do not start discovery', () => {
    for (const text of [
      '', 'x = 1\n', 'class Config:\n    pass\n', 'class Config( ):\n    pass\n',
      '# class Old(PreTrainedModel):\n', '    # class Old(Base):\n',
      'class Empty(\n    # Base\n):\n    pass\n',
      '"""Subclass PreTrainedModel to define a model."""\n',
    ]) {
      assert.equal(shouldStartBackend({ ...base, text }), false, text);
    }
  });

  test('never twice, and never after a failure', () => {
    // One failed start per window is a diagnostic; one per save is a loop, and saves are
    // exactly what this fires on.
    assert.equal(shouldStartBackend({ ...base, backendRunning: true }), false);
    assert.equal(shouldStartBackend({ ...base, startFailed: true }), false);
  });

  test('an untrusted window never spawns python', () => {
    // D20 adopted VS Code's own Workspace Trust instead of asking again in our words. Starting
    // a backend in a restricted window would make that adoption a lie.
    assert.equal(shouldStartBackend({ ...base, trusted: false }), false);
  });

  test('only real python files on disk', () => {
    assert.equal(shouldStartBackend({ ...base, languageId: 'markdown' }), false);
    assert.equal(shouldStartBackend({ ...base, scheme: 'git' }), false, 'a diff view is not an open folder');
  });
});

suite('activation.autoOpenClass', () => {
  const base = { isFork: true, graphShowing: false, enabled: true, classes: ['Net', 'Block'] };

  test('the fork draws the first class in the file', () => {
    assert.equal(autoOpenClass(base), 'Net');
  });

  test('never in someone else\'s editor', () => {
    // The graph beside the code is this product's proposition. Doing it uninvited inside stock
    // VS Code or Cursor is taking over a window somebody opened for something else.
    // Another editor does not open a canvas unasked...
    assert.equal(autoOpenClass({ ...base, isFork: false }), undefined);
    // ...but it does when the user has written the setting themselves. The fork
    // gate was a veto with no way past it, so the first thing anyone does in a
    // normal editor — open a model file — did nothing, with no hint that a
    // setting governed it.
    assert.equal(
      autoOpenClass({ ...base, isFork: false, enabledExplicitly: true }),
      'Net',
    );
    // Default `true` is nobody's decision, and must not read as consent.
    assert.equal(
      autoOpenClass({ ...base, isFork: false, enabled: true, enabledExplicitly: false }),
      undefined,
    );
    // An explicit `false` still wins, in the fork as anywhere else.
    assert.equal(
      autoOpenClass({ ...base, enabled: false, enabledExplicitly: true }),
      undefined,
    );
  });

  test('never a second canvas, and never when the setting is off', () => {
    assert.equal(autoOpenClass({ ...base, graphShowing: true }), undefined);
    // The setting, not a memory of the last close: the canvas is the product, and closing it
    // once must not silently switch the product off for this folder.
    assert.equal(autoOpenClass({ ...base, enabled: false }), undefined);
  });

  test('closing the canvas is not a refusal — the next model file draws again', () => {
    // graphShowing false again after a close, setting untouched: it opens.
    assert.equal(autoOpenClass({ ...base, graphShowing: false }), 'Net');
  });

  test('a file with no classes draws nothing', () => {
    assert.equal(autoOpenClass({ ...base, classes: [] }), undefined);
  });
});

suite('activation.moduleClasses', () => {
  test('names every torch module in the file, in source order', () => {
    // Why: auto-open draws the FIRST of these without asking the backend, so the order and
    // the base rule have to be exactly declaresModule's — one regex, two callers.
    const text = [
      'import torch.nn as nn',
      'class Stem(nn.Module):',
      '    pass',
      'class Helper:',
      '    pass',
      'class Net(Base, torch.nn.Module):',
      '    pass',
      'class Plain(Module):',
      '    pass',
    ].join('\n');
    assert.deepEqual(moduleClasses(text), ['Stem', 'Net', 'Plain']);
  });

  test('a commented-out class and a class with no torch base are not modules', () => {
    assert.deepEqual(moduleClasses('# class Old(nn.Module):\nclass Cfg(dict):\n    pass\n'), []);
    assert.equal(declaresModule('# class Old(nn.Module):\n'), false);
  });

  test('declaresModule is moduleClasses being non-empty', () => {
    const text = 'class FiveLayerCNN(nn.Module):\n    pass\n';
    assert.equal(declaresModule(text), true);
    assert.deepEqual(moduleClasses(text), ['FiveLayerCNN']);
  });
});

suite('autoOpenClass: a restored panel is not a drawn graph', () => {
  const base = { isFork: true, enabled: true, classes: ['Net', 'Other'] };

  test('a canvas that has been sent a projection suppresses auto-open', () => {
    // The case the flag was written for: the user is already looking at a graph,
    // and opening a second file must not redraw the canvas under them.
    assert.equal(autoOpenClass({ ...base, graphShowing: true }), undefined);
  });

  test('a restored but empty canvas still opens the file the user is looking at', () => {
    // A window reload hands the panel back through deserializeWebviewPanel before
    // any file is opened. Reading "a graph is showing" from the panel's existence
    // meant auto-open could never fire again for the rest of the session: the
    // canvas stayed empty, switching Python files did nothing, and the only way
    // out was to close it. The caller now asks whether a projection was ever sent.
    assert.equal(autoOpenClass({ ...base, graphShowing: false }), 'Net');
  });
});

suite('autoOpenClass: a graph of another file is not a reason to stay', () => {
  const base = { isFork: true, enabled: true, classes: ['Net'] };

  test('the canvas already showing THIS file is the no-op', () => {
    // The caller passes graphShowing only when the projection is of the file being
    // opened. Re-drawing it would move cards the user had arranged for no reason.
    assert.equal(autoOpenClass({ ...base, graphShowing: true }), undefined);
  });

  test('a graph of a DIFFERENT file must be replaced, not kept', () => {
    // The bug this pins: the canvas drew simple.py, the user opened blocks.py, and the
    // canvas went on showing simple.py — badged "Traced — shapes measured", naming no
    // file — for the rest of the session. The graph beside the code has to be the code
    // that is actually open.
    assert.equal(autoOpenClass({ ...base, graphShowing: false }), 'Net');
  });
});

suite('which class the canvas opens', () => {
  const base = { isFork: true, graphShowing: false, enabled: true };

  test('a generated model opens on the model, not on the helper declared above it', () => {
    // code_generation emits helper modules FIRST and the model LAST, so "the first class"
    // opened a helper for every modular model the product itself wrote. Worse, the helper
    // takes required constructor arguments, so Validate answered "Failed at build".
    const source = [
      'class EncoderStack(nn.Module):',
      '    def __init__(self, module_type, num_layers):',
      '        super().__init__()',
      '',
      'class BertBase(nn.Module):',
      '    def __init__(self):',
      '        super().__init__()',
      "        self.encoder_stack = EncoderStack(module_type='encoder_stack', num_layers=12)",
    ].join('\n');
    assert.equal(
      autoOpenClass({ ...base, classes: ['EncoderStack', 'BertBase'], source }),
      'BertBase',
    );
  });

  test('the __main__ example block does not make the model look like a component', () => {
    // Every generated file ends with `model = BertBase()`. Counting that instantiation
    // excluded the model along with the helper, so nothing was left and the rule fell back
    // to the first class — the exact bug it was written to fix.
    const source = [
      'class EncoderStack(nn.Module):',
      '    def __init__(self, module_type, num_layers):',
      '        super().__init__()',
      '',
      'class BertBase(nn.Module):',
      '    def __init__(self):',
      '        super().__init__()',
      "        self.encoder_stack = EncoderStack(module_type='encoder_stack', num_layers=12)",
      '',
      'if __name__ == "__main__":',
      '    model = BertBase()',
      '    x = torch.randint(0, 30522, (1, 512))',
    ].join('\n');
    assert.equal(
      autoOpenClass({ ...base, classes: ['EncoderStack', 'BertBase'], source }),
      'BertBase',
    );
  });

  test('a single-class file is unchanged', () => {
    const source = 'class Net(nn.Module):\n    def __init__(self):\n        super().__init__()';
    assert.equal(autoOpenClass({ ...base, classes: ['Net'], source }), 'Net');
  });

  test('a hand-written file whose model comes first still opens on it', () => {
    const source = [
      'class Net(nn.Module):',
      '    def __init__(self):',
      '        super().__init__()',
      '        self.block = Block()',
      '',
      'class Block(nn.Module):',
      '    pass',
    ].join('\n');
    assert.equal(autoOpenClass({ ...base, classes: ['Net', 'Block'], source }), 'Net');
  });

  test('several unreferenced classes keep the old, predictable answer', () => {
    const source = 'class A(nn.Module):\n    pass\n\nclass B(nn.Module):\n    pass';
    assert.equal(autoOpenClass({ ...base, classes: ['A', 'B'], source }), 'A');
  });

  test('without the source it degrades to what it always did', () => {
    assert.equal(autoOpenClass({ ...base, classes: ['EncoderStack', 'BertBase'] }), 'EncoderStack');
  });
});
