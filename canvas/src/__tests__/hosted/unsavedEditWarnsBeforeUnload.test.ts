/**
 * Closing the tab on an uncommitted edit used to destroy it silently.
 *
 * A parameter change reaches the server about 4s later (autosave debounce plus
 * round trip). Closing the tab inside that window lost the edit with no dialog,
 * no message, and nothing afterwards to say it had happened — while reload and
 * in-app navigation were both covered by the local draft queue. That asymmetry
 * is what made it dangerous: the app had taught users the "Saved" chip could be
 * trusted, so there was no reason to wait before closing.
 *
 * Two things can be wrong here and both are tested: the rule the handler
 * applies, and the queue predicate the rule reads.
 */
import { warnIfUnsaved } from '../../utils/unloadGuard';
import { RevisionedGraphSaveQueue } from '../../services/revisionedGraphSaveQueue';

function fakeEvent() {
  return { preventDefault: jest.fn(), returnValue: undefined as unknown as string };
}

const clean = { hasUncommittedWork: () => false, needsSave: () => false };
const inFlight = { hasUncommittedWork: () => true, needsSave: () => false };
const dirtyOnly = { hasUncommittedWork: () => false, needsSave: () => true };

describe('the unload rule', () => {
  it('prompts when the queue is still holding the user’s work', () => {
    const event = fakeEvent();

    const prompted = warnIfUnsaved(event, inFlight);

    expect(prompted).toBe(true);
    // Both are required: browsers honour preventDefault plus a string
    // returnValue, and setting only one of them shows no dialog at all.
    expect(event.preventDefault).toHaveBeenCalled();
    expect(event.returnValue).toBe('');
  });

  it('stays silent on a clean exit', () => {
    const event = fakeEvent();

    const prompted = warnIfUnsaved(event, clean, {});

    // A warning users learn to dismiss is a warning they will dismiss on the one
    // occasion it mattered, so this must not fire when nothing is pending.
    expect(prompted).toBe(false);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(event.returnValue).toBeUndefined();
  });

  it('prompts while the edit is still behind the autosave debounce', () => {
    // The half the first attempt at this fix missed entirely. The queue cannot
    // see an edit until the debounce hands it over, and measured on the live app
    // a layer insert left hasUncommittedWork() false for the first ~300ms —
    // precisely when closing the tab destroyed the edit. needsSave() is the same
    // dirty check the autosave effect uses, so the two cannot disagree.
    const event = fakeEvent();

    expect(warnIfUnsaved(event, dirtyOnly, { name: 'M' })).toBe(true);
    expect(event.preventDefault).toHaveBeenCalled();
  });

  it('does not prompt on a dirty canvas when no request is supplied', () => {
    // The caller skips building a request for an empty never-saved canvas, and
    // an absent request must not be read as "clean" by accident nor as "dirty".
    const event = fakeEvent();

    expect(warnIfUnsaved(event, dirtyOnly)).toBe(false);
  });

  it('stays silent when no editor is mounted', () => {
    const event = fakeEvent();

    expect(warnIfUnsaved(event, null)).toBe(false);
    expect(warnIfUnsaved(event, undefined)).toBe(false);
    expect(event.preventDefault).not.toHaveBeenCalled();
  });
});

describe('the queue predicate the rule reads', () => {
  it('is false on a fresh queue', () => {
    expect(new RevisionedGraphSaveQueue(jest.fn(), {}).hasUncommittedWork()).toBe(false);
  });

  it('is true from the moment an edit is enqueued until its commit resolves', async () => {
    let release: (value: unknown) => void = () => {};
    const persist = jest.fn(() => new Promise((resolve) => { release = resolve; }));
    const queue = new RevisionedGraphSaveQueue(persist as any, {});
    queue.configure('g1', 1, null);

    queue.enqueue({ name: 'M', graph: { nodes: [], edges: [] } } as any, {});

    // This is precisely the window in which the edit was being lost.
    expect(queue.hasUncommittedWork()).toBe(true);

    release({ revision: 2 });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(queue.hasUncommittedWork()).toBe(false);
  });
});
