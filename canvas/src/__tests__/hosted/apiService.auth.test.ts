/**
 * Local Canvas requests carry the per-launch token and do not load Firebase.
 */
const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  jest.resetAllMocks();
  jest.resetModules();
});

function mockFetchOk() {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => ({}),
    text: async () => '{}',
  } as Response);
}

function lastFetchHeaders(): Record<string, string> {
  return (global.fetch as jest.Mock).mock.calls[0][1].headers as Record<string, string>;
}

describe('ApiService local launch authentication', () => {
  it('a local entry needs no cloud module and preserves its per-launch authentication', async () => {
    jest.resetModules();
    jest.doMock('../../lib/firebase', () => { throw new Error('Local Canvas must not load Firebase'); });
    const { ApiService } = require('../../services/apiService');
    const { setLocalApiToken } = require('../../config/apiBase');
    setLocalApiToken('test-local-launch');
    mockFetchOk();
    await ApiService.request('/api/health');
    expect(lastFetchHeaders()['X-tensorViz-Token']).toBe('test-local-launch');
    expect(lastFetchHeaders().Authorization).toBeUndefined();
  });
});
