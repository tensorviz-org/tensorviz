import { webcrypto } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Public verification key and payload format from @tldraw/editor 5.0.1.
const publicKey = 'MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEHJh0uUfxHtCGyerXmmatE368Hd9rI6LH9oPDQihnaCryRFWEVeOvf9U/SPbyxX74LFyJs5tYeAHq5Nc0Ax25LQ';

export async function verifyRendererLicense(key, { now = Date.now(), verificationKey = publicKey } = {}) {
  let payload;
  try {
    const [data, signature, extra] = key.replace(/[\u200B-\u200D\uFEFF\r\n]/g, '').split('.');
    const [prefix, encoded] = data.split('/');
    if (!prefix.startsWith('tldraw-') || extra !== undefined) throw new Error();
    const bytes = Buffer.from(encoded, 'base64');
    const cryptoKey = await webcrypto.subtle.importKey('spki', Buffer.from(verificationKey, 'base64'),
      { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
    if (!await webcrypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, cryptoKey,
      Buffer.from(signature, 'base64'), bytes)) throw new Error();
    payload = JSON.parse(bytes.toString('utf8'));
  } catch {
    // Never include the credential or decoder input in release logs.
    throw new Error('A valid signed TLDRAW_LICENSE_KEY is required for publication.');
  }
  const [, hosts, flags, expires] = payload;
  if (!Array.isArray(hosts) || !Number.isInteger(flags) || !Number.isFinite(Date.parse(expires))) {
    throw new Error('The renderer license has unsupported metadata.');
  }
  if (flags & 4) throw new Error('An internal-only renderer license cannot be published.');
  if (!hosts.includes('*') && !(flags & 32)) {
    throw new Error('The local renderer requires an all-hosts or native-app license.');
  }
  // New releases require a current key, even where the SDK offers a grace period
  // for existing installations. A trial has no grace period.
  if (now >= Date.parse(expires)) throw new Error('The renderer license has expired. Renew it before publishing.');
  return { type: flags & 16 ? 'trial' : 'licensed', expires, hosts };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await verifyRendererLicense(process.env.TLDRAW_LICENSE_KEY)));
}
