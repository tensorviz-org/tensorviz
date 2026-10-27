import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  INIT_COMMENT,
  configCall,
  declaredInit,
  initSkeleton,
  missingArguments,
  overridesArgument,
} from '../../initComment';

/**
 * task.md P0's acceptance: `# tensorviz: init=Net(hidden=8)` becomes
 * `--overrides '{"hidden": 8}'`, and positional arguments are refused.
 *
 * The cases here mirror `backend/tests/graph/test_ide_init_comment.py::DRIFT_CASES` on
 * purpose. There are three readers of this line — the parser that draws the graph, the
 * ingest tier that builds the object, and this host, which assembles the worker's argv —
 * and if they disagree about what counts as a declaration, the graph that is drawn, the
 * model that is built and the command that is run all disagree with each other.
 */
suite('initComment.declaredInit', () => {
  test('a declaration becomes the --overrides JSON the worker takes', () => {
    const found = declaredInit('# tensorviz: init=Net(hidden=8)', 'Net');
    assert.deepEqual(found?.overrides, { hidden: 8 });
    assert.equal(found?.error, undefined);
    assert.equal(overridesArgument(found), '{"hidden":8}');
  });

  test('several keyword arguments survive, in the order written', () => {
    const found = declaredInit('# tensorviz: init=Net(hidden=8, depth=2)', 'Net');
    assert.deepEqual(found?.overrides, { hidden: 8, depth: 2 });
    assert.equal(overridesArgument(found), '{"hidden":8,"depth":2}');
  });

  test('positional arguments are refused, and say what to write instead', () => {
    // The refusal is the feature. Accepting `Net(8, 2)` would mean guessing a parameter
    // order the host cannot see; ignoring it would mean the user edits a line that does
    // nothing at all.
    const found = declaredInit('# tensorviz: init=Net(8, 2)', 'Net');
    assert.ok(found);
    assert.deepEqual(found.overrides, {});
    assert.match(found.error ?? '', /Name the arguments/);
    assert.equal(overridesArgument(found), undefined, 'a refused line passes no overrides');
  });

  test('a call that builds an argument is accepted, and sends no overrides', () => {
    // The host cannot build this: the name lives in the user's module, which only the
    // worker imports. So the declaration is accepted and `--overrides` carries nothing,
    // which is what makes the worker read the line itself.
    const found = declaredInit('# tensorviz: init=Net(cfg=load_config())', 'Net');
    assert.equal(found?.error, undefined);
    assert.deepEqual(found?.overrides, { cfg: configCall('load_config', {}) });
    assert.equal(overridesArgument(found), undefined);
  });

  test('a config call suppresses the other arguments too, rather than sending half', () => {
    // `ingest_source` takes `dict(ref.overrides) or _init_from_comment(...)`, so a partial
    // dict WINS and the config is never built. Sending half is worse than sending none.
    const found = declaredInit('# tensorviz: init=Net(cfg=ModelConfig(width=8), depth=2)', 'Net');
    assert.equal(found?.error, undefined);
    assert.equal(overridesArgument(found), undefined);
  });

  test('a value that is neither a literal nor a call is still refused', () => {
    const bare = declaredInit('# tensorviz: init=Net(cfg=DEFAULT_CONFIG)', 'Net');
    assert.match(bare?.error ?? '', /literal/);
    assert.equal(overridesArgument(bare), undefined);

    // A dotted path is a different mistake and says so: the worker resolves one name
    // against the module it imported, and will not walk an attribute chain.
    const dotted = declaredInit('# tensorviz: init=Net(cfg=config.ModelConfig())', 'Net');
    assert.match(dotted?.error ?? '', /dotted path/);
    assert.equal(overridesArgument(dotted), undefined);
  });

  test('a line that is not a call is refused even though it names a class', () => {
    const found = declaredInit('# tensorviz: init=Net', 'Net');
    assert.ok(found, 'a malformed line must reach the user, not be filtered away');
    assert.ok(found.error);
  });

  test('literal containers, strings and bools survive the round trip', () => {
    // Python spellings, because that is what people write in a Python file.
    const found = declaredInit(
      "# tensorviz: init=Net(dims=[4, 8], size=(2, 3), name='a', flag=True, none=None)",
      'Net',
    );
    assert.deepEqual(found?.overrides, {
      dims: [4, 8],
      size: [2, 3],
      name: 'a',
      flag: true,
      none: null,
    });
    assert.equal(found?.error, undefined);
  });

  test('a comma inside a container does not split the argument', () => {
    const found = declaredInit('# tensorviz: init=Net(cfg={"a": 1, "b": 2}, n=3)', 'Net');
    assert.deepEqual(found?.overrides, { cfg: { a: 1, b: 2 }, n: 3 });
  });

  test('the declaration for another class is not this class’s declaration', () => {
    const two = '# tensorviz: init=Encoder(width=4)\n# tensorviz: init=Decoder(width=16)\n';
    assert.deepEqual(declaredInit(two, 'Encoder')?.overrides, { width: 4 });
    assert.deepEqual(declaredInit(two, 'Decoder')?.overrides, { width: 16 });
    assert.equal(declaredInit(two, 'Net'), undefined);
  });

  test('a file that declares nothing declares nothing', () => {
    // undefined, not an empty object: "the user said nothing" and "the user said build it
    // with no arguments" are different, and only the first falls through to the ordinary
    // no-argument construction.
    assert.equal(declaredInit('class Net(nn.Module):\n    pass\n', 'Net'), undefined);
    assert.equal(declaredInit(''), undefined);
  });

  test('the line number points at the comment, so a diagnostic can land on it', () => {
    const source = 'import torch\n\n# tensorviz: init=Net(hidden=8)\nclass Net:\n';
    assert.equal(declaredInit(source, 'Net')?.line, 2);
  });

  test('whitespace around the marker is optional, exactly as in both Python copies', () => {
    assert.deepEqual(declaredInit('#tensorviz:init=Net(hidden=8)', 'Net')?.overrides, {
      hidden: 8,
    });
  });
});

suite('initComment.missingArguments', () => {
  test('keeps structured binding names when the worker no longer repeats Python exception text', () => {
    assert.deepEqual(missingArguments('ExecutionRequirement: binding: missing_argument', ['cfg']), ['cfg']);
    assert.deepEqual(missingArguments('Use `# tensorviz: init=Net(width=8)`', ['width']), ['width=8']);
    assert.deepEqual(missingArguments('binding failed', ['cfg', 'cfg', 'not a name']), ['cfg']);
  });

  test('names the arguments Python said were missing', () => {
    // The value of the diagnostic is the NAME. "tensorViz could not construct Transformer"
    // and nothing else sends the user to read a traceback, which is the work the IDE exists
    // to remove.
    assert.deepEqual(
      missingArguments("__init__() missing 1 required positional argument: 'cfg'"),
      ['cfg'],
    );
    assert.deepEqual(
      missingArguments("__init__() missing 2 required positional arguments: 'cfg' and 'depth'"),
      ['cfg', 'depth'],
    );
  });

  test('invents nothing when the message names no argument', () => {
    assert.deepEqual(missingArguments('__init__() takes 2 positional arguments but 1 was given'), []);
    assert.deepEqual(missingArguments('TypeError: something else entirely'), []);
  });
});

suite('initComment.initSkeleton', () => {
  test('the Quick Fix inserts the argument names, left for the user to fill in', () => {
    assert.equal(
      initSkeleton('Transformer', ['cfg']),
      '# tensorviz: init=Transformer(cfg=)',
    );
    assert.equal(initSkeleton('Net', ['hidden', 'depth']), '# tensorviz: init=Net(hidden=, depth=)');
  });

  test('with no names known it still gives a line to edit rather than nothing', () => {
    assert.equal(initSkeleton('Net', []), '# tensorviz: init=Net(arg=)');
  });
});

/** The parity seam, and the reason the suite above is not enough on its own.
 *
 *  The cases above were written to mirror `backend/tests/graph/test_ide_init_comment.py`, but a
 *  mirror nobody looks in is just a second copy: that file's DRIFT_CASES could gain a case, or
 *  the pattern in `utils/code_parser.py` could change, and nothing in TypeScript would go red.
 *  The Python drift guard holds its two readers together the same way; this holds the third.
 *
 *  Same technique as probes.parity.test.ts: the rule is lifted out of the other language's
 *  SOURCE TEXT at runtime, so restructuring it fails loudly here instead of quietly checking
 *  nothing. What is liftable across the language boundary is the pattern (a regex is a regex)
 *  and the list of cases. What is NOT liftable is Python's `ast.literal_eval`, so the answers
 *  themselves are recorded below — verified once against both Python readers — and the
 *  coverage check is what keeps that record from going stale: neither list can grow, shrink or
 *  be reworded without this test naming the case that moved. */

const BACKEND = path.resolve(__dirname, '..', '..', '..', '..', 'backend');
const CODE_PARSER = path.join(BACKEND, 'utils', 'code_parser.py');
const INGEST = path.join(BACKEND, 'services', 'ingest', 'service.py');
const DRIFT_GUARD = path.join(BACKEND, 'tests', 'graph', 'test_ide_init_comment.py');

/** The `# tensorviz: init=` pattern as `file` spells it, in JavaScript's spelling. The only
 *  difference between the two languages here is Python's named group, which this reader does
 *  not name — so a change to anything else is a real divergence and must be reported. */
function liftPattern(file: string): string {
  const match = /^_?INIT_COMMENT = re\.compile\(r"(.+)"\)$/m.exec(fs.readFileSync(file, 'utf8'));
  if (!match) {
    throw new Error(
      `Could not find the init-comment pattern in ${file}. It was restructured; update this ` +
        'parity test (and check initComment.ts still agrees with it) rather than deleting the check.',
    );
  }
  return match[1].replace('(?P<value>', '(');
}

/** The declarations the Python drift guard runs both of its readers over. */
function liftDriftCases(): { source: string; cls: string }[] {
  const block = /^DRIFT_CASES = \[$([\s\S]*?)^\]$/m.exec(fs.readFileSync(DRIFT_GUARD, 'utf8'));
  if (!block) {
    throw new Error(
      `Could not find DRIFT_CASES in ${DRIFT_GUARD}. It was restructured or renamed; update ` +
        'this parity test rather than deleting the check.',
    );
  }
  const entries = [...block[1].matchAll(/\(\s*("(?:[^"\\]|\\.)*")\s*,\s*("(?:[^"\\]|\\.)*")\s*\)/g)];
  if (entries.length === 0) {
    throw new Error(`DRIFT_CASES in ${DRIFT_GUARD} parsed as empty — the cases are no longer ` +
      'plain double-quoted pairs, so this test was reading nothing. Update it.');
  }
  return entries.map((entry) => ({
    source: JSON.parse(entry[1]) as string,
    cls: JSON.parse(entry[2]) as string,
  }));
}

/** What this reader must answer for each shared case, keyed by the declaration text.
 *
 *  `undefined` means "no declaration at all", which is a different answer from a refused one.
 *  Every row here was checked against `declared_init` in both `utils/code_parser.py` and
 *  `services/ingest/service.py` when it was written; the coverage test below is what stops the
 *  set of rows and the set of Python cases from parting company afterwards. */
const ANSWERS = new Map<string, { overrides: Record<string, unknown>; refused: boolean } | undefined>([
  ['# tensorviz: init=Net(hidden=8)', { overrides: { hidden: 8 }, refused: false }],
  ['# tensorviz: init=Net(hidden=8, depth=2)', { overrides: { hidden: 8, depth: 2 }, refused: false }],
  ['# tensorviz: init=Net(8)', { overrides: {}, refused: true }],
  // A call that builds the argument. `load()` was refused until the config-object form
  // landed; the rule is "a bare name, called with literal keyword arguments", and it does
  // not ask whether the name is a class — so this one is now accepted in all three readers.
  ['# tensorviz: init=Net(cfg=load())', { overrides: { cfg: configCall('load', {}) }, refused: false }],
  [
    '# tensorviz: init=Net(cfg=ModelConfig())',
    { overrides: { cfg: configCall('ModelConfig', {}) }, refused: false },
  ],
  [
    '# tensorviz: init=Net(cfg=ModelConfig(width=8), depth=2)',
    { overrides: { cfg: configCall('ModelConfig', { width: 8 }), depth: 2 }, refused: false },
  ],
  // A dotted path is not resolved, a call inside the call is not built, and a bare name is
  // not evaluated. Three different mistakes, each refused on its own terms.
  ['# tensorviz: init=Net(cfg=config.ModelConfig())', { overrides: {}, refused: true }],
  ['# tensorviz: init=Net(cfg=ModelConfig(inner=Other()))', { overrides: {}, refused: true }],
  ['# tensorviz: init=Net(cfg=DEFAULT_CONFIG)', { overrides: {}, refused: true }],
  ['# tensorviz: init=Net', { overrides: {}, refused: true }],
  // Nothing after the `=`, so there is no declaration to read: the pattern needs one character.
  ['# tensorviz: init=', undefined],
  ['#tensorviz:init=Net(hidden=8)', { overrides: { hidden: 8 }, refused: false }],
  // Another class's line is not this class's, in either language.
  ['# tensorviz: init=Other(hidden=8)', undefined],
  ['', undefined],
  // Python's `False` is JSON's `false` by the time the worker receives it.
  ['# tensorviz: init=Net(dims=[1, 2], flag=False)', { overrides: { dims: [1, 2], flag: false }, refused: false }],
]);

suite('initComment parity: initComment.ts vs backend', () => {
  test('all three readers spell the declaration the same way', () => {
    // The pattern is duplicated three times for a reason the Python copies explain: the worker
    // runs with the USER's repository first on sys.path, so the ingest tier cannot import the
    // parser's copy, and the extension cannot import either. Duplicated, not free to differ.
    assert.equal(
      liftPattern(CODE_PARSER),
      INIT_COMMENT.source,
      'utils/code_parser.py and initComment.ts no longer match a declaration the same way',
    );
    assert.equal(
      liftPattern(INGEST),
      INIT_COMMENT.source,
      'services/ingest/service.py and initComment.ts no longer match a declaration the same way',
    );
  });

  test('every case the Python drift guard runs is a case this reader answers', () => {
    const cases = liftDriftCases();
    for (const { source } of cases) {
      assert.ok(
        ANSWERS.has(source),
        `DRIFT_CASES gained ${JSON.stringify(source)}. Add it to ANSWERS with what this reader ` +
          'answers for it — a case Python guards and TypeScript does not is the drift itself.',
      );
    }
    for (const source of ANSWERS.keys()) {
      assert.ok(
        cases.some((one) => one.source === source),
        `${JSON.stringify(source)} is no longer in DRIFT_CASES. Python stopped guarding it, or ` +
          'reworded it; follow the change here rather than leaving a row that guards nothing.',
      );
    }
  });

  test('this reader answers every shared case the way the Python readers do', () => {
    for (const { source, cls } of liftDriftCases()) {
      const named = JSON.stringify(source);
      assert.ok(ANSWERS.has(source), `no recorded answer for ${named} — see the coverage test`);
      const expected = ANSWERS.get(source);
      const found = declaredInit(source, cls);
      if (expected === undefined) {
        assert.equal(found, undefined, `${named} must declare nothing, as it does in Python`);
        continue;
      }
      assert.ok(found, `${named} declares something in Python and nothing here`);
      assert.deepEqual(found.overrides, expected.overrides, `${named} builds different arguments`);
      // WHETHER the line was refused is the shared contract; the wording of the refusal is
      // written once in each language, exactly as the Python guard treats it.
      assert.equal(found.error !== undefined, expected.refused, `${named} is refused on one side only`);
    }
  });

  test('a bare class name is the one line the three readers still read differently', () => {
    // Python answers cls="" for `# tensorviz: init=Net`: the line named no class it will act
    // on, so `declared_init`'s filter cannot swallow it and the user is told about it whichever
    // class was asked for. This reader answers cls="Net" instead — readCall's bare-identifier
    // branch — so asking for a different class drops the line silently, which is exactly what
    // initComment.ts's own doc comment promises will never happen.
    //
    // Pinned here rather than left to be discovered: the fix belongs in initComment.ts, and
    // when it lands this test goes red and should be deleted along with the divergence.
    assert.equal(declaredInit('# tensorviz: init=Net', 'Encoder'), undefined);
    // Both languages do agree whenever the name happens to be the class being asked about.
    assert.ok(declaredInit('# tensorviz: init=Net', 'Net')?.error);
  });
});
