import * as assert from 'node:assert/strict';
import { locateConfigField, CONFIG_STARTER, modelConfigStarter } from '../../validationConfig';

suite('tensorviz.toml field navigation', () => {
  function token(source: string, field: (string | number)[], expected: string, exact = true) {
    const location = locateConfigField(source, field);
    assert.equal(source.slice(location.start, location.end), expected);
    assert.equal(location.exact, exact);
    return location;
  }
  test('distinguishes profiles, quoted and dotted keys', () => {
    const source = '[profiles.first]\ndevice = "cpu"\n[profiles."second"]\ninputs.kwargs.x = 1\ndevice = "bad"';
    const found = token(source, ['profiles', 'second', 'device'], 'device');
    assert.equal(found.start, source.lastIndexOf('device'));
    token(source, ['profiles', 'second', 'inputs', 'kwargs', 'x'], 'inputs');
  });
  test('locates nested generators and elements in inline tables and arrays', () => {
    const source = `[profiles.run]
inputs = { args = [{kind = "tensor", shape = [2, 3], generator = { kind = "full", value = 0 }}], kwargs = {} }
`;
    token(source, ['profiles', 'run', 'inputs', 'args', 0, 'generator', 'value'], 'value');
    token(source, ['profiles', 'run', 'inputs', 'args', 0, 'shape', 1], '3');
    token(source, ['profiles', 'run', 'inputs', 'args', 0, 'dtype'], '{', false);
  });
  test('missing fields anchor to the nearest table and retain the precise path', () => {
    token('[profiles.run]\nkind = "model"', ['profiles', 'run', 'device'], '[', false);
    assert.deepEqual(locateConfigField('', ['profiles']), { start: 0, end: 0, exact: false });
  });
  test('comments and multiline strings cannot impersonate another field', () => {
    const source = `# device = "fake"
[profiles.run]
script = """other
[profiles.fake]
device = "cpu"
"""
args = ["a,b", "device = 'fake'", 'x#y']
# inputs = {}
capture = { model = "__main__:Model", call = 1 }
`;
    token(source, ['profiles', 'run', 'capture', 'call'], 'call');
    token(source, ['profiles', 'run', 'args', 2], "'x#y'");
    assert.equal(locateConfigField(source, ['profiles', 'fake', 'device']).exact, false);
  });
  test('array-of-table fields point to the selected invocation argument', () => {
    const source = '[profiles.run]\n[[profiles.run.inputs.args]]\nkind = "tensor"\n[profiles.run.inputs.args.generator]\nkind = "full"\nvalue = 1\n[[profiles.run.inputs.args]]\nkind = "none"';
    const first = token(source, ['profiles', 'run', 'inputs', 'args', 0, 'generator', 'value'], 'value');
    const second = token(source, ['profiles', 'run', 'inputs', 'args', 1, 'kind'], 'kind');
    assert.ok(first.start < second.start);
    assert.equal(second.start, source.lastIndexOf('kind'));
  });
  test('oversize invalid config has bounded navigation work', () => {
    assert.deepEqual(locateConfigField('x'.repeat(262145), ['profiles']), { start: 0, end: 0, exact: false });
  });
  test('starter does not supply guessed execution values', () => {
    for (const line of CONFIG_STARTER.split('\n').filter(line => line && !line.startsWith('#'))) {
      assert.ok(['version = 1', '[profiles.run]'].includes(line));
    }
  });
  test('model starter identifies the target but leaves constructor and tensors unresolved', () => {
    const starter = modelConfigStarter('whisper.model:Whisper');
    assert.match(starter, /model = "whisper.model:Whisper"/);
    assert.match(starter, /Saving a complete profile during graph setup continues/);
    assert.match(starter, /Explicit example execution settings/);
    const settings = starter.split('\n').filter(line => !line.startsWith('#')).join('\n');
    assert.doesNotMatch(settings, /^(init|inputs|factory)\s*=/m);
  });
});
