import corpus from '../../../../tests/fixtures/contracts/local-browser-protocol.json';
import {
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_ROLES,
  LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES,
  isLocalBrowserMessage,
} from '../../ide/protocol.generated';

describe('local browser protocol', () => {
  it('exposes the frozen version and roles', () => {
    expect(LOCAL_BROWSER_PROTOCOL_VERSION).toBe(1);
    expect(LOCAL_BROWSER_ROLES).toEqual(['host', 'view']);
  });

  it('accepts every canonical valid message', () => {
    expect(corpus.valid).not.toHaveLength(0);
    for (const message of corpus.valid) {
      expect(isLocalBrowserMessage(message, corpus.expectedWorkspaceSessionId)).toBe(true);
    }
  });

  it('counts UTF-8 bytes for portable transfer instead of allowing oversized Unicode', () => {
    const message = corpus.valid.find(item => item.type === 'captureExportResult')!;
    const oversized = { ...message, capture: { version: 1, sourceNames: 'anonymized', capture: { label: '界'.repeat(Math.floor(LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES / 3) + 1) } } };
    expect(isLocalBrowserMessage(oversized, corpus.expectedWorkspaceSessionId)).toBe(false);
  });

  it('refuses every canonical invalid message', () => {
    expect(corpus.invalid).not.toHaveLength(0);
    for (const message of corpus.invalid) {
      expect(isLocalBrowserMessage(message, corpus.expectedWorkspaceSessionId)).toBe(false);
    }
  });
});
