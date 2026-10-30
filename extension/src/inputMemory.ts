/**
 * What this class is fed, remembered beside the layout it is drawn with.
 *
 * The problem: tensorViz has to decide what tensor to push through a model before it
 * can show anything, and for most real models nothing in the source says. The answer
 * used to be a line the user typed into their own `.py` — `# tensorviz: input=2,8` —
 * and typing it is the last piece of homework between opening a file and seeing
 * measured shapes. Builds before 1bfb0e51b took that homework on: the trace that
 * succeeded was asked what it ran on and the answer was written down next to the
 * layout. Builds since only read it back, for the drawing (see below).
 *
 * It lives in `.tensorviz/views/<rel>#<Class>.view.json` — the same per-(file, class)
 * sidecar the saved node positions already use, which is the **un-ignored** half of
 * `.tensorviz/`: the repository's own `.gitignore` excludes that directory's contents
 * and then re-includes `views/`, so a layout — and now an input — travels with the
 * project while `runs/` and `cache/` never do. That is
 * deliberate. "What is this model fed" is a property of the model, not of the machine
 * that traced it, and it is exactly what the committed `# tensorviz: input=` line used
 * to say — so it travels with the repository for the same reason the declaration did,
 * and a teammate opening the class gets shapes on the first open instead of a canvas of
 * red. Nothing here records a path, a port, an interpreter or a token, which is what
 * the ignored half of `.tensorviz/` exists for.
 *
 * Nothing writes this memory any more. The harvesting went with the retired editor
 * canvas controller, and the Canvas never updates input-memory sidecars: opening and
 * refresh only read them. A sidecar therefore holds what an earlier build wrote, and
 * its `input_shape` can be older than the class's last trace.
 *
 * TWO fields, never one, and this is the trap that makes them two:
 *
 *  * `input` is a **re-feedable spec** — the worker's `--input` grammar, which accepts
 *    shapes (`"2,8"`, `"2,8;2"`) *or a Python expression* (`torch.randint(0, 50257,
 *    (1, 16))`). An expression is the only way to say "integers, not floats" to an
 *    `Embedding`, so it must be stored verbatim and never regenerated from measured
 *    dimensions.
 *  * `input_shape` is a **drawing hint** — the integer dimensions the run measured,
 *    which is all the torch-free AST tier can use, and which stays correct even when
 *    `input` is an expression that tier could never evaluate.
 *
 * Collapsing them would feed a float tensor to an `Embedding` on the next trace.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { WorkerEnvelope } from './worker';

/** The two keys this module owns in the view sidecar. Everything else in that file —
 *  `graph_id`, `node_positions` — belongs to the layout and is preserved untouched. */
export interface InputMemory {
  /**
   * `--input`, verbatim, exactly as a person typed it.
   *
   * Written only by an earlier build's example-input prompt, never composed from
   * measured dimensions. Nothing writes it now, and no trace is fed it. See
   * `rememberMeasuredShape` for why the shape search's own answer was deliberately NOT
   * recorded here.
   */
  input?: string;
  /** The dimensions of the FIRST argument of a successful trace, as an earlier build
   *  saved them. Nothing updates them now, so they can be older than the class's last
   *  trace. The drawing tier draws one Input node, so it can carry one shape. */
  input_shape?: number[];
}

/**
 * Does this buffer declare an input anywhere?
 *
 * A file-wide PRESENCE test, not a second parser: it reads whether the line exists and
 * never what it says or which class it belongs to. Both are the backend's business
 * (`backend/services/ingest/input_declarations.py` gives each class only its own
 * declaration), and this side deliberately does not have an opinion about them — that
 * is why the backend takes `measured_input_shape` as a separate field, ranked below the
 * class's own declaration, instead of letting the host pre-rank the two.
 *
 * The question is only about the remembered `input`, which would become the worker's
 * `--input`, and on that side `--input` OUTRANKS the file
 * (`services/ingest/service.py::ingest_source` — measured: `--input 5,8` against a file
 * declaring `2,8` reports `example_input=[[5, 8]]`, `input_provenance="explicit"`). So a
 * remembered value passed blindly would make this cache beat the line the user just
 * typed, on the one tier that actually runs the model. The test is conservative for
 * that one field: any declaration text, even another class's or one inside a string
 * the backend ignores, makes the remembered `input` stand aside. No trace request
 * reads the remembered `input` today (the Canvas passes `input: undefined`), so the
 * answer has no effect until one does.
 */
export function declaresInput(source: string): boolean {
  return /#\s*tensorviz:\s*input\s*=/.test(source);
}

/**
 * The shapes this envelope may be believed about, or `undefined`.
 *
 * **The gate is `error === null`, and it is the whole function.** `diagnostics.example_input`
 * is populated on a FAILED trace too: a two-argument `Linear(8, 8)` model measured here
 * exits 4 with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (672x224 and 8x8)`
 * after 40 candidates and still reports `example_input = [[1, 3, 224, 224], [1]]` — the
 * last thing the search TRIED, which is precisely the shape that did not work. Writing
 * that down would pin the model to an image shape for ever, in a committed file.
 */
export function measuredInputShape(envelope: WorkerEnvelope | undefined): number[] | undefined {
  if (!envelope || envelope.error) {
    return undefined;
  }
  const diagnostics = (envelope.result as { diagnostics?: { example_input?: unknown } } | null)
    ?.diagnostics;
  const first = (diagnostics?.example_input as unknown[] | undefined)?.[0];
  if (!Array.isArray(first) || first.length === 0) {
    return undefined;
  }
  // A shape is whole positive numbers. Anything else is not one, and a half-read
  // envelope must fall through to the tiers that guess rather than be drawn.
  const dims = first.map((dim) => (typeof dim === 'number' ? dim : Number.NaN));
  return dims.every((dim) => Number.isInteger(dim) && dim > 0) ? dims : undefined;
}

/**
 * The memory this buffer may be traced and drawn with: the sidecar's, without the
 * remembered `input` when the file declares one.
 *
 * `input_shape` is always returned. The backend draws with it only a class that has no
 * declaration of its own, so it never beats one, even an expression the drawing cannot
 * read, and in a file where only another class declares it draws this class with the
 * shape an earlier build stored for it (see the header).
 *
 * The one call site of the whole module on the read side, so that "the `.py` wins" is a
 * single decision made once rather than a rule each surface re-derives.
 */
export function inputMemoryFor(sidecarFile: string, source: string): InputMemory {
  const memory = readInputMemory(sidecarFile);
  if (declaresInput(source)) {
    delete memory.input;
  }
  return memory;
}

/** The sidecar's input memory, or an empty one. A missing or unreadable file is not an
 *  error: the memory is an optimisation, and a workspace without one must behave exactly
 *  as it did before this existed. */
export function readInputMemory(file: string): InputMemory {
  try {
    const saved = JSON.parse(fs.readFileSync(file, 'utf8')) as InputMemory;
    const input = typeof saved.input === 'string' && saved.input ? saved.input : undefined;
    const shape =
      Array.isArray(saved.input_shape) &&
      saved.input_shape.length > 0 &&
      saved.input_shape.every((dim) => Number.isInteger(dim) && dim > 0)
        ? saved.input_shape
        : undefined;
    return { ...(input ? { input } : {}), ...(shape ? { input_shape: shape } : {}) };
  } catch {
    return {};
  }
}

/**
 * Merge `memory` into the sidecar, keeping the layout half byte for byte.
 *
 * No production code has called this since 1bfb0e51b, directly or through
 * `rememberMeasuredShape`; only unit tests do.
 *
 * Returns whether it wrote. A no-op write is refused on purpose: this file is
 * recommended for commit, and a trace-on-save that rewrote it every time would put a
 * churning diff in front of the user after every keystroke-and-save.
 */
export function writeInputMemory(file: string, memory: InputMemory): boolean {
  let saved: Record<string, unknown> = {};
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8')) as unknown;
    if (parsed && typeof parsed === 'object') {
      saved = parsed as Record<string, unknown>;
    }
  } catch {
    /* no sidecar yet, or one this build cannot read: the layout half has nothing to lose */
  }
  const next = { ...saved, ...memory };
  if (JSON.stringify(next) === JSON.stringify(saved)) {
    return false;
  }
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
  return true;
}

/**
 * Harvest: write down what a successful trace ran on.
 *
 * Builds before 1bfb0e51b called this after every successful trace. No production code
 * calls it now; only its unit tests do.
 *
 * Only `input_shape` — the drawing hint — is harvested from the trace itself, and the
 * omission is deliberate rather than an oversight. The re-feedable `input` was written
 * ONLY by the prompt, because feeding a value back arrives at the backend as `--input`
 * and is reported as `input_provenance: "explicit"`, whose documented meaning is "a
 * person chose this" (`services/ingest/spec.py`). Replaying the shape search's own
 * answer through it would launder a guess into a choice on the card the user reads —
 * every subsequent open of a model nobody ever described would claim somebody had
 * described it. So the search keeps re-running and keeps reporting
 * `search-candidate` / `seed-from-first-layer` / `assumed-default` truthfully, while the
 * shape it found still reaches the drawing tier through `measured_input_shape`, which is
 * a hint and is ranked as one.
 *
 * (The honest way to replay a searched value is a `remembered` rung of `InputProvenance`
 * alongside the existing five. That is a `services/ingest/spec.py` +
 * `services/ingest/service.py` change and is left for whoever owns those.)
 */
export function rememberMeasuredShape(file: string, envelope: WorkerEnvelope | undefined): boolean {
  const shape = measuredInputShape(envelope);
  return shape ? writeInputMemory(file, { input_shape: shape }) : false;
}
