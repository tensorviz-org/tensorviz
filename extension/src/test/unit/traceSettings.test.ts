import * as assert from 'node:assert/strict';
import { browserTraceSettings } from '../../traceSettings';

suite('saved model validation', () => {
  test('validation defaults on and respects an explicit opt-out', () => {
    assert.deepEqual(browserTraceSettings(() => undefined), { onSave: true, tier: 'real' });
    assert.equal(browserTraceSettings(section => section === 'traceOnSave' ? false : undefined).onSave, false);
  });

  test('the trace tier remains an explicit choice', () => {
    assert.equal(browserTraceSettings(section => section === 'traceTier' ? 'fake' : true).tier, 'fake');
    assert.equal(browserTraceSettings(() => 'auto').tier, 'auto');
    assert.equal(browserTraceSettings(() => 'unknown').tier, 'real');
  });
});
