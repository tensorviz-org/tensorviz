import * as assert from 'node:assert/strict';
import { graphToRestore, retiredCanvasSerializer, storedOpenGraph } from '../../openGraphState';

suite('open graph restore', () => {
  const record = { workspaceFolder: '/workspace/demo', file: 'models/net.py', cls: 'Net' };

  test('keeps a class target and drops traversal or empty records', () => {
    assert.deepEqual(storedOpenGraph(record), record);
    assert.equal(storedOpenGraph({ ...record, file: '../net.py' }), undefined);
    assert.equal(storedOpenGraph({ ...record, cls: '' }), undefined);
    assert.equal(storedOpenGraph(null), undefined);
  });

  test('a stored class fills an empty canvas and does not replace one that is open', () => {
    assert.deepEqual(graphToRestore(record, false), record);
    assert.equal(graphToRestore(record, true), undefined);
    assert.equal(graphToRestore(undefined, false), undefined);
  });

  test('a restored editor panel is closed and the stored graph is requested again', async () => {
    let restored = 0;
    const panel = { disposed: false, dispose() { this.disposed = true; } };
    await retiredCanvasSerializer(() => { restored += 1; }).deserializeWebviewPanel(panel);
    assert.equal(panel.disposed, true);
    assert.equal(restored, 1);
  });
});
