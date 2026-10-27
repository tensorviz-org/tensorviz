import * as assert from 'node:assert/strict';
import { browserTraceSettings } from '../../traceSettings';

suite('deprecated capture on save', () => {
  test('traceOnSave true does not schedule a capture', () => {
    const read = (section: string) => section === 'traceOnSave' ? true : section === 'traceTier' ? 'real' : undefined;
    assert.deepEqual(browserTraceSettings(read), { onSave: false, tier: 'real' });
  });

  test('the trace tier is still read', () => {
    assert.equal(browserTraceSettings(section => section === 'traceTier' ? 'fake' : true).tier, 'fake');
    assert.equal(browserTraceSettings(() => 'not-a-tier').tier, 'real');
  });
});
