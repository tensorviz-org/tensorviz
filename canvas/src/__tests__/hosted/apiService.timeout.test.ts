import { ApiService } from '../../services/apiService';

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  jest.useRealTimers();
});

it('aborts a bounded request and reports an actionable timeout', async () => {
  jest.useFakeTimers();
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => {
    options?.signal?.addEventListener('abort', () => reject(new Error('aborted')));
  })) as jest.Mock;

  const pending = ApiService.request('/api/slow', { timeoutMs: 25 });
  await Promise.resolve();
  jest.advanceTimersByTime(25);

  await expect(pending).rejects.toThrow('API request timed out after 25ms');
  expect((global.fetch as jest.Mock).mock.calls[0][1]).not.toHaveProperty('timeoutMs');
});

it('preserves an explicit caller cancellation as cancellation, not timeout', async () => {
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => {
    options?.signal?.addEventListener('abort', () => reject(new Error('cancelled')));
  })) as jest.Mock;
  const controller = new AbortController();

  const pending = ApiService.request('/api/cancelled', {
    signal: controller.signal,
    timeoutMs: 5_000,
  });
  await Promise.resolve();
  controller.abort();

  await expect(pending).rejects.toThrow('cancelled');
});
