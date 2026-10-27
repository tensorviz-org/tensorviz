import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { countLayerTypes } from '../../stack';

/** The readiness probe exists twice on purpose (task.md M0): the extension cannot import from
 *  canvas/, and run-e2e.mjs cannot import from extension/. Two copies of a rule that must
 *  agree is a bug waiting to happen — a backend that reads "serving" to `make e2e` and "not
 *  serving" to the extension, or the reverse, is a bad afternoon. This test is the seam that
 *  keeps them honest until M2's sync-protocol.mjs makes one of them a generated copy.
 *
 *  The .mjs is a top-level script that parses argv and calls process.exit, so it cannot be
 *  imported. Its counting rule is lifted out of the source text instead — which also means this
 *  test fails loudly if that function is restructured, rather than silently checking nothing. */

const RUN_E2E = path.resolve(
  __dirname,
  '..',
  '..',
  '..',
  '..',
  'canvas',
  'tests',
  'e2e',
  'run-e2e.mjs',
);

const COUNTING_RULE =
  /try \{\s*(const payload = JSON\.parse\(result\.body\);[\s\S]*?)\n\s*if \(count === 0\)/;

function loadE2eCounter(): (body: string) => number {
  const source = fs.readFileSync(RUN_E2E, 'utf8');
  const match = COUNTING_RULE.exec(source);
  if (!match) {
    throw new Error(
      `Could not find the layer-type counting rule in ${RUN_E2E}. ` +
        'It was restructured; update this parity test (and check stack.ts still agrees with it) ' +
        'rather than deleting the check.',
    );
  }
  const body = `const result = { body };\ntry {\n${match[1]}\nreturn count;\n} catch { return 0; }`;
  return new Function('body', body) as (body: string) => number;
}

suite('readiness probe parity: stack.ts vs run-e2e.mjs', () => {
  const fixtures: { name: string; body: string; expected: number }[] = [
    // What server_v2.py:980-983 actually answers.
    { name: 'a populated layer list', body: JSON.stringify({ layers: [{ id: 'conv2d' }, { id: 'linear' }] }), expected: 2 },
    // A backend that is up but not serving the model plane — the case both probes exist for.
    { name: 'an empty layer list', body: JSON.stringify({ layers: [] }), expected: 0 },
    // A proxy, a login page, or an HTML error — must count as zero, never throw.
    { name: 'a non-JSON body', body: '<!DOCTYPE html><title>Sign in</title>', expected: 0 },
    // Both copies deliberately tolerate a bare array, so both must keep tolerating it: a
    // shape only one of them accepts is exactly the divergence this test is here to catch.
    { name: 'a bare array', body: JSON.stringify([{ id: 'conv2d' }, { id: 'linear' }, { id: 'relu' }]), expected: 3 },
  ];

  test('both implementations agree on every fixture', () => {
    const fromE2e = loadE2eCounter();
    for (const fixture of fixtures) {
      const mine = countLayerTypes(fixture.body);
      const theirs = fromE2e(fixture.body);
      assert.equal(mine, fixture.expected, `stack.ts miscounted ${fixture.name}`);
      assert.equal(theirs, fixture.expected, `run-e2e.mjs miscounted ${fixture.name}`);
      assert.equal(mine, theirs, `probes disagree on ${fixture.name}`);
    }
  });
});
