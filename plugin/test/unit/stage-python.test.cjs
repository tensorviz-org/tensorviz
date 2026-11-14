const {test} = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
test('compiler Python staging requires the exact bounded archive hash', async () => {
  const {verifiedPythonArchive} = await import('../../scripts/stage-windows-arm-python.mjs');
  const bytes = Buffer.from('test-only fixture archive');
  const hash = createHash('sha256').update(bytes).digest('hex');
  await assert.rejects(verifiedPythonArchive(new Response(bytes), hash, undefined), /size limit/);
  assert.deepEqual(await verifiedPythonArchive(new Response(bytes), hash, bytes.length), bytes);
  await assert.rejects(verifiedPythonArchive(new Response(bytes), '0'.repeat(64), bytes.length), /checksum/);
  await assert.rejects(verifiedPythonArchive(new Response(bytes), hash, bytes.length-1), /size limit/);
  await assert.rejects(verifiedPythonArchive(new Response('', {status:404}), hash, bytes.length), /download failed/);
});
