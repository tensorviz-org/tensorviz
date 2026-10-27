const { test } = require('node:test');
const assert = require('node:assert/strict');
const { webcrypto } = require('node:crypto');

test('release licensing rejects expired, tampered and unsuitable keys without exposing them', async () => {
  const { verifyRendererLicense } = await import('../../scripts/renderer-license.mjs');
  const pair = await webcrypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify']);
  const verificationKey = Buffer.from(await webcrypto.subtle.exportKey('spki', pair.publicKey)).toString('base64');
  const sign = async (hosts = ['*'], flags = 16) => {
    const payload = Buffer.from(JSON.stringify(['test-only', hosts, flags, '2030-01-01']));
    const signature = await webcrypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, pair.privateKey, payload);
    return `tldraw-test/${payload.toString('base64')}.${Buffer.from(signature).toString('base64')}`;
  };
  const options = { verificationKey, now: Date.parse('2029-12-31') };
  const key = await sign();
  assert.deepEqual(await verifyRendererLicense(key, options), { type: 'trial', expires: '2030-01-01', hosts: ['*'] });
  await assert.rejects(verifyRendererLicense(key, { ...options, now: Date.parse('2030-01-01') }), /expired/);
  await assert.rejects(verifyRendererLicense(await sign(['tensorviz.dev']), options), /all-hosts or native/);
  await assert.rejects(verifyRendererLicense(await sign(['*'], 20), options), /internal-only/);
  const tampered = key.slice(0, key.indexOf('.') + 1) + 'AAAA';
  for (const invalid of [undefined, '', 'private-invalid-value', tampered]) {
    await assert.rejects(verifyRendererLicense(invalid, options), error =>
      error.message === 'A valid signed TLDRAW_LICENSE_KEY is required for publication.');
  }
  await assert.rejects(verifyRendererLicense(key), /valid signed/, 'A test signer cannot authorize a production release');
});
