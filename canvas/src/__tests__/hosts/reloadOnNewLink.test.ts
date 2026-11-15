/** A new view link opened in a tab that already shows the page changes only the
 *  fragment. The page must reload so its transport trades the new credential. */
import { reloadOnNewLink } from '../../ide/reloadOnNewLink';

function fakeTab(hash = '') {
  const events = new EventTarget();
  const location = { hash, reload: jest.fn() };
  const tab = {
    location,
    addEventListener: events.addEventListener.bind(events),
    removeEventListener: events.removeEventListener.bind(events),
  } as unknown as Parameters<typeof reloadOnNewLink>[0];
  const open = (next: string) => {
    location.hash = next;
    events.dispatchEvent(new Event('hashchange'));
  };
  return { tab, location, open };
}

describe('reloadOnNewLink', () => {
  it('reloads when a new view link arrives in the same tab', () => {
    const { tab, location, open } = fakeTab();
    reloadOnNewLink(tab);
    open('#session=ws-2&credential=fresh');
    expect(location.reload).toHaveBeenCalledTimes(1);
  });

  it.each(['', '#', '#/graph', '#session=ws-2', '#credential=fresh', '#section-2'])(
    'ignores the fragment %p', hash => {
      const { tab, location, open } = fakeTab('#session=ws-1&credential=used');
      reloadOnNewLink(tab);
      open(hash);
      expect(location.reload).not.toHaveBeenCalled();
    });

  it('stops listening when unsubscribed', () => {
    const { tab, location, open } = fakeTab();
    reloadOnNewLink(tab)();
    open('#session=ws-2&credential=fresh');
    expect(location.reload).not.toHaveBeenCalled();
  });
});
