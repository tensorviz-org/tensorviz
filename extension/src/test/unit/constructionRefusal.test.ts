/** What the page says when a model will not build.
 *
 * Exit 3 from the worker means "construction raised", which is far wider than "the
 * constructor needs arguments". Reported on a real model: F5AECStftXAttn, whose every
 * constructor argument already has a default, failed with
 * ModuleNotFoundError: No module named 'x_transformers' — and the page answered
 * "Supply its constructor arguments with # tensorviz: init= and retry", sending the user
 * to edit a file that was never wrong.
 */
import * as assert from 'node:assert/strict';
import { constructionRefusal } from '../../localBrowserSync';
import type { WorkerEnvelope } from '../../worker';

const envelope = (error: Record<string, unknown> | null): WorkerEnvelope =>
  ({ protocol_version: 1, result: null, spans: {}, error, worker: {} } as unknown as WorkerEnvelope);

suite('construction refusal', () => {
  test('an installed-library import failure never asks to reinstall a missing dependency', () => {
    const detail = constructionRefusal(envelope({ diagnostic: {
      phase: 'import', code: 'execution_failed', names: ['ValueError'],
    } }), 'VQModel');
    assert.match(detail, /package initialization code \(ValueError\).*library-version compatibility/);
    assert.doesNotMatch(detail, /missing dependency|Install/);
    const unresolved = constructionRefusal(envelope({ diagnostic: { phase: 'import', code: 'unresolved_import' } }), 'VQModel');
    assert.match(unresolved, /required library versions and import paths/);
    assert.doesNotMatch(unresolved, /Install the missing/);
  });
  test('unknown forward arity requests a real recipe instead of constructor edits', () => {
    const detail = constructionRefusal(envelope({ type: 'ExecutionRequirement', message: 'private', frames: [],
      diagnostic: { phase: 'binding', code: 'unknown_signature', field: ['inputs'] } }), 'DinoVisionTransformer');
    assert.match(detail, /needs explicit inputs/);
    assert.match(detail, /args and kwargs.*tensorviz.toml/);
    assert.doesNotMatch(detail, /init=|private|could not be constructed/);
  });
  test('a missing import is named, with what to do about it', () => {
    const detail = constructionRefusal(envelope({ type: 'ModuleNotFoundError',
      message: "No module named 'x_transformers'", frames: [], missing_module: 'x_transformers' }), 'F5AECStftXAttn');
    assert.match(detail, /imports "x_transformers"/);
    assert.match(detail, /Install.*selected Python environment/);
    assert.doesNotMatch(detail, /init=|Select Python Environment/, 'missing dependencies should be repaired in the capture environment');
    const diagnosed = constructionRefusal(envelope({ diagnostic: {
      phase: 'construction', code: 'missing_dependency', names: ['x_transformers'],
    } }), 'F5AECStftXAttn');
    assert.match(diagnosed, /x_transformers.*Install.*selected Python environment/);
    assert.doesNotMatch(diagnosed, /Select Python Environment/);
  });

  test('another exception names its type rather than guessing', () => {
    const detail = constructionRefusal(envelope({ type: 'TypeError',
      message: "__init__() missing 1 required positional argument: 'cfg'", frames: [] }), 'Net');
    assert.match(detail, /TypeError was raised/);
    assert.match(detail, /tensorviz.toml/, 'the profile owns explicit setup');
    assert.doesNotMatch(detail, /declare.*constructor|init=/, 'TypeError alone cannot identify the missing setup');
    assert.doesNotMatch(detail, /missing 1 required/, 'exception text stays in the Output channel');
  });

  test('no envelope at all still says something actionable', () => {
    const detail = constructionRefusal(undefined, 'Net');
    assert.match(detail, /Net could not be constructed/);
    assert.match(detail, /tensorViz Output/);
  });

  test('named bindings distinguish construction and forward inputs', () => {
    const detail = constructionRefusal(envelope({ diagnostic: {
      phase: 'binding', code: 'missing_argument', field: ['init'], names: ['dims'], remedy: 'edit_profile',
    } }), 'Whisper');
    assert.match(detail, /constructor arguments.*Missing arguments \(dims\)/);
    assert.doesNotMatch(detail, /needs explicit inputs/);
  });

  test('legacy diagnostic identifiers cannot inject arbitrary exception text', () => {
    const detail = constructionRefusal(envelope({ missing_module: 'secret\nPASSWORD=123', type: 'Bad\nsecret' }), 'Net');
    assert.doesNotMatch(detail, /secret|PASSWORD|123/);
  });

  test('factory and device failures direct users to the relevant setup', () => {
    assert.match(constructionRefusal(envelope({ diagnostic: { code: 'invalid_factory' } }), 'Net'), /exact selected model class/);
    assert.match(constructionRefusal(envelope({ diagnostic: { code: 'unavailable_device' } }), 'Net'), /configured device/);
  });

  test('the class is always named, so a multi-class file is unambiguous', () => {
    for (const cls of ['Net', 'F5AECStftXAttn']) {
      assert.match(constructionRefusal(envelope(null), cls), new RegExp(`^${cls} `));
    }
  });
});
