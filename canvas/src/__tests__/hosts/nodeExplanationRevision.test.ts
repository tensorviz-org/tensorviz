import { createHash, webcrypto } from 'crypto';
import { TextEncoder as NodeTextEncoder } from 'util';
import { nodeExplanationRevision } from '../../ide/nodeExplanationRevision';

it('matches the host graph revision and changes with capture or source identity', async () => {
  const originalCrypto = Object.getOwnPropertyDescriptor(global, 'crypto');
  const originalEncoder = global.TextEncoder;
  Object.defineProperty(global, 'crypto', { configurable: true, value: webcrypto });
  global.TextEncoder = NodeTextEncoder as typeof TextEncoder;
  const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Model', documentVersion: 3, sourceDigest: 'source', backendRevision: 7 };
  try {
    const revision = await nodeExplanationRevision(target, 'capture');
    expect(revision).toBe(createHash('sha256').update(JSON.stringify([3, 'source', 7, 'capture'])).digest('hex'));
    expect(await nodeExplanationRevision(target, 'other-capture')).not.toBe(revision);
    expect(await nodeExplanationRevision({ ...target, sourceDigest: 'edited' }, 'capture')).not.toBe(revision);
    expect(await nodeExplanationRevision(target)).toBe(createHash('sha256').update(JSON.stringify([3, 'source', 7, undefined])).digest('hex'));
  } finally {
    if (originalCrypto) Object.defineProperty(global, 'crypto', originalCrypto);
    global.TextEncoder = originalEncoder;
  }
});
