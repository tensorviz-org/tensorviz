/** Shared API error parsing and the live palette template-inventory request.
 * All network calls are intercepted with a global.fetch mock. */

import { ApiService } from '../../services/apiService';

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  jest.resetAllMocks();
});

function mockFetch(status: number, body: unknown, statusText?: string) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: statusText ?? (status === 200 ? 'OK' : 'Error'),
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response);
}

function mockFetchNetworkError(message = 'Network failure') {
  global.fetch = jest.fn().mockRejectedValue(new Error(message));
}

describe('ApiService error format parsing', () => {
  it('unpacks a structured detail into step, message and hint', async () => {
    // A backend that answers {"detail": {...}} would otherwise surface to the user as
    // "[object Object]", which is the failure this formatting exists to prevent.
    mockFetch(500, {
      detail: { step: 'template-inventory', message: 'registry unavailable', hint: 'retry shortly' },
    });

    await expect(ApiService.getPaperTemplateInventory()).rejects.toThrow(
      'API Error (500) [template-inventory] registry unavailable Hint: retry shortly'
    );
  });

  it('includes duration_ms in the error message when present in detail', async () => {
    mockFetch(500, { detail: { step: 'template-inventory', message: 'boom', duration_ms: 1234 } });

    await expect(ApiService.getPaperTemplateInventory()).rejects.toThrow('Duration: 1234ms');
  });

  it('throws a plain string detail as the error message', async () => {
    mockFetch(404, { detail: 'inventory not found' });

    await expect(ApiService.getPaperTemplateInventory()).rejects.toThrow(
      'API Error (404): inventory not found'
    );
  });

  it('rethrows network errors instead of reporting an API status', async () => {
    mockFetchNetworkError('Network failure');

    await expect(ApiService.getPaperTemplateInventory()).rejects.toThrow('Network failure');
  });
});

describe('ApiService.getPaperTemplateInventory', () => {
  it('GETs /api/papers/template-inventory', async () => {
    // The layer palette loads its template blocks from this on mount, and its failure
    // path only sets a message — a broken URL here renders an empty palette silently.
    mockFetch(200, { summary: {}, templates: [], dynamic_template_ids: [], dynamic_mappings: {} });

    const result = await ApiService.getPaperTemplateInventory();

    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain('/api/papers/template-inventory');
    expect(init?.method ?? 'GET').toBe('GET');
    expect(result.templates).toEqual([]);
  });
});
