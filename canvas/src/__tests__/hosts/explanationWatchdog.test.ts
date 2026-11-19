import { ExplanationWatchdog } from '../../ide/explanationWatchdog';

describe('explanation reply watchdog', () => {
  let watchdog: ExplanationWatchdog;
  beforeEach(() => { jest.useFakeTimers(); watchdog = new ExplanationWatchdog(); });
  afterEach(() => { watchdog.clear(); jest.useRealTimers(); });

  it('retains every live request clock while passive selection receipts churn', () => {
    const a = jest.fn(), b = jest.fn();
    watchdog.observe('model:a', 'a', 'loading', { timeoutMs: 10, expire: a });
    watchdog.observe('model:b', 'b', 'loading', { timeoutMs: 20, expire: b });
    for (let i = 0; i < 300; i++) watchdog.observe(`node:${i}`, `selection-${i}`, 'unavailable');
    jest.advanceTimersByTime(10);
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).not.toHaveBeenCalled();
    watchdog.observe('model:b', 'b', 'ready');
    jest.advanceTimersByTime(20);
    expect(b).not.toHaveBeenCalled();
  });

  it('expires once and refuses late success or activity for the expired identity', () => {
    const expire = jest.fn();
    watchdog.observe('node:n', 'request', 'loading', { timeoutMs: 60, expire });
    jest.advanceTimersByTime(60);
    expect(expire).toHaveBeenCalledTimes(1);
    expect(expire).toHaveBeenCalledWith('timeout');
    expect(watchdog.observe('node:n', 'request', 'ready')).toBe(false);
    expect(watchdog.observe('node:n', 'request', 'loading', { timeoutMs: 60, expire })).toBe(false);
    jest.advanceTimersByTime(600);
    expect(expire).toHaveBeenCalledTimes(1);
  });

  it('resets node inactivity when source documentation arrives, and stops on terminal replies', () => {
    const expire = jest.fn(), pending = { timeoutMs: 60, expire };
    watchdog.observe('node:n', 'request', 'queued', pending);
    jest.advanceTimersByTime(50);
    watchdog.observe('node:n', 'request', 'queued', pending);
    jest.advanceTimersByTime(50);
    expect(expire).not.toHaveBeenCalled();
    watchdog.observe('node:n', 'request', 'ready');
    expect(watchdog.observe('node:n', 'request', 'ready')).toBe(true);
    expect(watchdog.observe('node:n', 'request', 'loading', pending)).toBe(false);
    jest.advanceTimersByTime(60);
    expect(expire).not.toHaveBeenCalled();
  });

  it('accepts a cached node result after unavailable without reopening generation', () => {
    const expire = jest.fn(), pending = { timeoutMs: 60, expire };
    watchdog.observe('node:n', 'selection', 'unavailable');
    expect(watchdog.observe('node:n', 'selection', 'queued', pending)).toBe(false);
    expect(watchdog.observe('node:n', 'selection', 'loading', pending)).toBe(false);
    expect(watchdog.observe('node:n', 'selection', 'ready')).toBe(true);
    expect(watchdog.observe('node:n', 'selection', 'unavailable')).toBe(false);
    jest.advanceTimersByTime(60);
    expect(expire).not.toHaveBeenCalled();
  });

  it('keeps cancelled, interrupted, superseded and unavailable model receipts terminal', () => {
    watchdog.observe('node:cancelled', 'selection', 'cancelled');
    expect(watchdog.observe('node:cancelled', 'selection', 'ready')).toBe(false);
    watchdog.observe('node:interrupted', 'selection', 'loading', { timeoutMs: 60, expire: jest.fn() });
    watchdog.interrupt();
    expect(watchdog.observe('node:interrupted', 'selection', 'ready')).toBe(false);
    watchdog.observe('node:n', 'old', 'unavailable');
    watchdog.observe('node:n', 'new', 'unavailable');
    expect(watchdog.observe('node:n', 'old', 'ready')).toBe(false);
    watchdog.observe('model', 'pass', 'unavailable');
    expect(watchdog.observe('model', 'pass', 'ready')).toBe(false);
  });

  it('requires advancing model progress to renew an acknowledged run', () => {
    const expire = jest.fn();
    watchdog.observe('model', 'request', 'loading', { timeoutMs: 10, expire });
    jest.advanceTimersByTime(5);
    watchdog.observe('model', 'request', 'loading', { timeoutMs: 75, expire }, 0);
    jest.advanceTimersByTime(60);
    watchdog.observe('model', 'request', 'loading', { timeoutMs: 75, expire }, 1);
    jest.advanceTimersByTime(60);
    expect(watchdog.observe('model', 'request', 'loading', { timeoutMs: 75, expire }, 0)).toBe(false);
    expect(watchdog.observe('model', 'request', 'loading', { timeoutMs: 75, expire }, 1)).toBe(true);
    jest.advanceTimersByTime(15);
    expect(expire).toHaveBeenCalledTimes(1);
  });

  it('allows each genuine cold-provider phase once without letting phase replay renew inactivity', () => {
    const expire = jest.fn(), pending = { timeoutMs: 75, expire };
    watchdog.observe('model', 'request', 'loading', pending, 0, 'Connecting to agent…');
    jest.advanceTimersByTime(45);
    watchdog.observe('model', 'request', 'loading', pending, 0, 'Reading model components…');
    jest.advanceTimersByTime(60);
    expect(expire).not.toHaveBeenCalled();
    watchdog.observe('model', 'request', 'loading', pending, 0, 'Connecting to agent…');
    watchdog.observe('model', 'request', 'loading', pending, 0, 'Reading model components…');
    jest.advanceTimersByTime(15);
    expect(expire).toHaveBeenCalledTimes(1);
  });

  it('retires superseded requests and permits a deliberate retry after expiry', () => {
    const old = jest.fn(), current = jest.fn();
    watchdog.observe('node:n', 'old', 'loading', { timeoutMs: 10, expire: old });
    watchdog.observe('node:n', 'new', 'queued', { timeoutMs: 20, expire: current });
    expect(watchdog.accepts('node:n', 'old', 'ready')).toBe(false);
    jest.advanceTimersByTime(20);
    expect(old).not.toHaveBeenCalled();
    expect(current).toHaveBeenCalledTimes(1);
    watchdog.retry('node:n', 'new', { timeoutMs: 10, expire: current });
    expect(watchdog.observe('node:n', 'new', 'loading', { timeoutMs: 60, expire: current })).toBe(true);
    watchdog.observe('node:n', 'new', 'ready');
    jest.advanceTimersByTime(60);
    expect(current).toHaveBeenCalledTimes(1);
  });

  it('clears timers on scope/disposal and interrupts pending work once when the provider changes', () => {
    const expire = jest.fn(), complete = jest.fn();
    watchdog.observe('model', 'request', 'loading', { timeoutMs: 75, expire });
    watchdog.observe('node:n', 'node', 'ready', { timeoutMs: 60, expire: complete });
    watchdog.interrupt(); watchdog.interrupt();
    expect(expire).toHaveBeenCalledTimes(1);
    expect(expire).toHaveBeenCalledWith('provider');
    expect(complete).not.toHaveBeenCalled();
    watchdog.clear();
    watchdog.observe('node:n', 'new', 'loading', { timeoutMs: 60, expire });
    watchdog.clear();
    jest.advanceTimersByTime(100);
    expect(expire).toHaveBeenCalledTimes(1);
    expect(jest.getTimerCount()).toBe(0);
  });
});
