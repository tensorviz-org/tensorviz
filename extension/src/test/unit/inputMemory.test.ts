/**
 * The per-class input memory: what a trace ran on, remembered so nobody has to type
 * `# tensorviz: input=` into their own source file.
 *
 * Every test here is one of the three ways a naive implementation of that gets it
 * wrong, and each one was measured against the real worker before it was written:
 *
 *  1. `diagnostics.example_input` is populated on a FAILED trace too. A two-argument
 *     `Linear(8, 8)` model exits 4 with "mat1 and mat2 shapes cannot be multiplied
 *     (672x224 and 8x8)" and still reports `[[1, 3, 224, 224], [1]]` — the candidate
 *     that did NOT work. Remembering it pins the model to an image shape for ever, in
 *     a file that is recommended for commit.
 *  2. `--input` also accepts a Python expression, which is the only way to hand an
 *     `Embedding` integers. Regenerating the remembered spec from measured dimensions
 *     would replace that expression with floats.
 *  3. The `.py` is the truth. `--input` OUTRANKS `# tensorviz: input=` in the worker
 *     (measured: `--input 5,8` against a file declaring `2,8` reports
 *     `input_provenance: "explicit"`), so a memory passed blindly beats the line the
 *     user just typed.
 */

import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import {
  declaresInput,
  inputMemoryFor,
  measuredInputShape,
  readInputMemory,
  rememberMeasuredShape,
  writeInputMemory,
} from '../../inputMemory';
import { WorkerEnvelope } from '../../worker';

/** An envelope the way the worker really prints one. `example_input` is a list of
 *  shapes, one per forward argument. */
function envelope(
  exampleInput: unknown,
  error: WorkerEnvelope['error'] = null,
): WorkerEnvelope {
  return {
    protocol_version: 1,
    result: { diagnostics: { example_input: exampleInput } },
    spans: {},
    error,
    worker: { python: '3.12.0' },
  };
}

const FAILED = {
  type: 'RuntimeError',
  message: 'mat1 and mat2 shapes cannot be multiplied (672x224 and 8x8)',
  frames: [],
};

let scratch: string;
let sidecar: string;

suite('inputMemory', () => {
  setup(() => {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-input-memory-'));
    sidecar = path.join(scratch, '.tensorviz', 'views', 'models', 'net.py#Net.view.json');
  });

  teardown(() => {
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  // ── trap 1: a failed trace still reports an example input ──────────────────

  test('the shapes a FAILED trace tried are not what the model is fed', () => {
    // The exact envelope the two-argument fixture produced: exit 4, 40 candidates,
    // and an example_input naming the image shape that raised. Believing it would
    // teach tensorViz that a Linear(8, 8) model takes pictures.
    const measured = measuredInputShape(envelope([[1, 3, 224, 224], [1]], FAILED));
    assert.equal(measured, undefined);
  });

  test('a failed trace does not overwrite what a successful one already measured', () => {
    writeInputMemory(sidecar, { input_shape: [1, 8] });
    const wrote = rememberMeasuredShape(sidecar, envelope([[1, 3, 224, 224], [1]], FAILED));

    assert.equal(wrote, false);
    assert.deepEqual(readInputMemory(sidecar).input_shape, [1, 8]);
  });

  test('a successful trace is remembered, first argument first', () => {
    assert.equal(rememberMeasuredShape(sidecar, envelope([[1, 8]])), true);
    assert.deepEqual(readInputMemory(sidecar).input_shape, [1, 8]);
  });

  test('an envelope with no measured tensor is not a measurement', () => {
    // A tree-only result: nothing ran, so there is nothing to remember, and an
    // empty shape drawn as an Input node would be a card with no dimensions.
    assert.equal(measuredInputShape(envelope([])), undefined);
    assert.equal(measuredInputShape(envelope(undefined)), undefined);
    assert.equal(measuredInputShape(undefined), undefined);
    assert.equal(measuredInputShape(envelope([['x', 8]])), undefined);
    assert.equal(measuredInputShape(envelope([[0, 8]])), undefined);
  });

  // ── trap 2: the spec and the drawing hint are different things ─────────────

  test('a remembered expression survives every later measurement', () => {
    // `torch.randint` is how a user says "integers, not floats" to an Embedding.
    // Harvesting must never rewrite it as the dimensions the run happened to see:
    // "1,16" is the same SHAPE and a float tensor, which the Embedding refuses.
    const expression = 'torch.randint(0, 50257, (1, 16))';
    writeInputMemory(sidecar, { input: expression });

    rememberMeasuredShape(sidecar, envelope([[1, 16]]));

    const memory = readInputMemory(sidecar);
    assert.equal(memory.input, expression, 'the re-feedable spec is the user\'s, verbatim');
    assert.deepEqual(memory.input_shape, [1, 16], 'the drawing hint is still the measurement');
  });

  // ── trap 3: the file outranks the memory ──────────────────────────────────

  test('a file that declares its own input is not second-guessed', () => {
    writeInputMemory(sidecar, { input: '1,3,224,224', input_shape: [1, 3, 224, 224] });
    const declared = 'import torch.nn as nn\n# tensorviz: input=2,8\nclass Net(nn.Module):\n    pass\n';

    // The remembered `input` is dropped, because as `--input` it would beat the
    // declaration in the worker. The measured shape is kept: the backend ranks it below the CLASS's own
    // declaration, and in a file where only another class declares, it is the
    // only thing that draws this class with a shape it has been traced with.
    assert.deepEqual(inputMemoryFor(sidecar, declared), { input_shape: [1, 3, 224, 224] });
    // Without the declaration the same sidecar is used in full.
    assert.deepEqual(inputMemoryFor(sidecar, 'class Net(nn.Module):\n    pass\n'), {
      input: '1,3,224,224',
      input_shape: [1, 3, 224, 224],
    });
  });

  test('the declaration is recognised however it is spaced', () => {
    assert.equal(declaresInput('#tensorviz:input=2,8'), true);
    assert.equal(declaresInput('#  tensorviz :  input = 2,8'), false, 'not the grammar either Python copy accepts');
    assert.equal(declaresInput('# tensorviz: input = torch.randn(2, 8)'), true);
    assert.equal(declaresInput('# tensorviz: init=Net(width=8)'), false);
    assert.equal(declaresInput(''), false);
  });

  // ── the sidecar is shared with the layout, and is committed ────────────────

  test('the layout half of the sidecar is never touched', () => {
    // The same file carries the node positions the user arranged by hand. Losing
    // those to a trace-on-save would be a far worse bug than the one this fixes.
    fs.mkdirSync(path.dirname(sidecar), { recursive: true });
    fs.writeFileSync(
      sidecar,
      JSON.stringify({ graph_id: 'g1', node_positions: { l1: { x: 640, y: 220 } } }),
      'utf8',
    );

    rememberMeasuredShape(sidecar, envelope([[2, 8]]));

    const saved = JSON.parse(fs.readFileSync(sidecar, 'utf8'));
    assert.equal(saved.graph_id, 'g1');
    assert.deepEqual(saved.node_positions, { l1: { x: 640, y: 220 } });
    assert.deepEqual(saved.input_shape, [2, 8]);
  });

  test('remembering the same measurement twice does not rewrite the file', () => {
    // This file is recommended for commit. A trace on every save that rewrote it
    // every time would put a churning diff in front of the user for ever.
    assert.equal(rememberMeasuredShape(sidecar, envelope([[2, 8]])), true);
    const first = fs.statSync(sidecar).mtimeMs;
    assert.equal(rememberMeasuredShape(sidecar, envelope([[2, 8]])), false);
    assert.equal(fs.statSync(sidecar).mtimeMs, first);
  });

  test('an unreadable or absent sidecar is an empty memory, not a failure', () => {
    // The common case on the first open, and the one that must not throw inside a
    // save handler: a workspace with no sidecar behaves exactly as it did before.
    assert.deepEqual(readInputMemory(sidecar), {});
    fs.mkdirSync(path.dirname(sidecar), { recursive: true });
    fs.writeFileSync(sidecar, '{ not json', 'utf8');
    assert.deepEqual(readInputMemory(sidecar), {});
  });

  test('a sidecar written by a different build is read conservatively', () => {
    // Anything that is not a positive-integer shape or a non-empty string is not a
    // memory: it falls through to the tiers that guess rather than being drawn.
    fs.mkdirSync(path.dirname(sidecar), { recursive: true });
    fs.writeFileSync(
      sidecar,
      JSON.stringify({ input: '', input_shape: [2, 'eight'] }),
      'utf8',
    );
    assert.deepEqual(readInputMemory(sidecar), {});
  });
});
