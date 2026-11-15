import { decodeLocalBrowserFragment } from './protocol.generated';

type LinkWindow = Pick<Window, 'addEventListener' | 'removeEventListener'> & {
  location: Pick<Location, 'hash' | 'reload'>;
};

/** A view link opened in a tab that already shows this page differs from it only
 *  after `#`, so the browser keeps the page and fires `hashchange` instead of
 *  loading the link. The transport reads and erases the fragment once, at load,
 *  so reload to let it trade the new credential. Any other fragment is ignored.
 *  Returns the unsubscribe. */
export function reloadOnNewLink(target: LinkWindow = window): () => void {
  const onHashChange = () => {
    if (decodeLocalBrowserFragment(target.location.hash) !== null) target.location.reload();
  };
  target.addEventListener('hashchange', onHashChange);
  return () => target.removeEventListener('hashchange', onHashChange);
}
