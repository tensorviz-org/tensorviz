import React, { useSyncExternalStore } from 'react';
import { createPortal } from 'react-dom';
import { CloudUpload, Copy, Download, ExternalLink, LogIn, X } from 'lucide-react';
import { bridgeActive, subscribeBridge } from './bridge';
import { IDLE_PUBLISH, type LocalBrowserPublishClient, type PublishState } from './localBrowserPublishClient';

const noSubscription = () => () => {};
const idle = () => IDLE_PUBLISH;

/** In VS Code, Publish adds the graph to the member's tensorViz account, where they
 *  share it. A plugin view publishes through its host and shows the steps here. The
 *  extension's browser page has no host to sign in through, so it saves a file instead. */
export default function PublishGraphButton({ disabled, publishing, openLink = openInBrowser }: {
  disabled: boolean;
  /** The plugin host's Publish, which signs in through the website and reports back here. */
  publishing?: LocalBrowserPublishClient | null;
  openLink?: (url: string) => void;
}) {
  const bridge = useSyncExternalStore(subscribeBridge, bridgeActive);
  const state = useSyncExternalStore(publishing ? publishing.subscribe : noSubscription, publishing ? () => publishing.state : idle);
  const button = React.useRef<HTMLButtonElement>(null);
  const publish = bridge || !!publishing;
  const command = publish ? 'share-graph-file' : 'export-graph-file';
  const run = () => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command } }));
  const running = state.phase === 'preparing' || state.phase === 'signing_in' || state.phase === 'uploading';
  return <>
    <button ref={button} type="button" className="local-canvas-control" disabled={disabled || running}
      aria-label={publish ? 'Publish graph' : 'Export graph'}
      title={publish ? 'Add this graph to your tensorViz account, then choose who can see it' : 'Save a read-only snapshot with the current layout'}
      onClick={run}>
      {publish ? <CloudUpload size={15} aria-hidden="true" /> : <Download size={15} aria-hidden="true" />}
      <span className="local-canvas-control-label">{running ? 'Publishing…' : publish ? 'Publish' : 'Export graph'}</span>
    </button>
    {publishing && state.phase !== 'idle' && <PublishStatus anchor={button} state={state} client={publishing} openLink={openLink} retry={run} />}
  </>;
}

function openInBrowser(url: string): void {
  window.open(url, '_blank', 'noopener,noreferrer');
}

const HEADINGS: Record<Exclude<PublishState['phase'], 'idle'>, string> = {
  preparing: 'Publishing to tensorViz', signing_in: 'Sign in to publish', uploading: 'Publishing to tensorViz',
  published: 'Published to tensorViz', failed: 'Could not publish',
};

/** The steps of one publish, beside the button: nothing opens the website until the member clicks. */
function PublishStatus({ anchor, state, client, openLink, retry }: {
  anchor: React.RefObject<HTMLButtonElement | null>;
  state: Exclude<PublishState, { phase: 'idle' }>;
  client: LocalBrowserPublishClient;
  openLink: (url: string) => void;
  retry: () => void;
}) {
  const [position, setPosition] = React.useState<React.CSSProperties | null>(null);
  const [copied, setCopied] = React.useState<'copied' | 'manual' | null>(null);
  React.useEffect(() => setCopied(null), [state]);
  React.useLayoutEffect(() => {
    const place = () => {
      const rect = anchor.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(320, window.innerWidth - 16);
      setPosition({ width, top: rect.bottom + 8, left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)) });
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); };
  }, [anchor]);
  if (!position) return null;
  const finished = state.phase === 'published' || state.phase === 'failed';
  const cancellable = state.phase === 'preparing' || state.phase === 'signing_in';
  const copy = (url: string) => {
    void Promise.resolve().then(() => navigator.clipboard.writeText(url))
      .then(() => setCopied('copied'), () => setCopied('manual'));
  };
  return createPortal(<div role="dialog" aria-label="Publish to tensorViz" data-testid="publish-status"
    className="local-canvas-popover local-canvas-action-panel local-canvas-publish" style={position}>
    <div className="local-canvas-popover-heading"><strong>{HEADINGS[state.phase]}</strong>
      {finished && <button type="button" className="local-canvas-control" aria-label="Close" onClick={() => client.dismiss()}><X size={15} /></button>}
    </div>
    <div className="local-canvas-publish-body">
      <p role="status">{state.detail}</p>
      {state.phase === 'signing_in' && <p className="local-canvas-publish-note">Pick a Google account on tensorviz.dev. Publishing continues here when you return.</p>}
      {state.phase === 'published' && copied === 'manual' && <input className="local-canvas-publish-link" readOnly aria-label="Graph link"
        value={state.graph.url} onFocus={event => event.currentTarget.select()} />}
      <div className="local-canvas-publish-actions">
        {state.phase === 'signing_in' && <button type="button" className="local-canvas-control local-canvas-publish-primary"
          onClick={() => openLink(state.signInUrl)}><LogIn size={15} aria-hidden="true" /><span>Continue in browser</span></button>}
        {state.phase === 'published' && <>
          <button type="button" className="local-canvas-control local-canvas-publish-primary" onClick={() => openLink(state.graph.shareUrl)}>
            <ExternalLink size={15} aria-hidden="true" /><span>Choose who can see it</span></button>
          <button type="button" className="local-canvas-control" onClick={() => copy(state.graph.url)}>
            <Copy size={15} aria-hidden="true" /><span>{copied === 'copied' ? 'Link copied' : 'Copy link'}</span></button>
        </>}
        {state.phase === 'failed' && <button type="button" className="local-canvas-control local-canvas-publish-primary"
          onClick={() => { client.dismiss(); retry(); }}>Try again</button>}
        {cancellable && <button type="button" className="local-canvas-control" onClick={() => client.cancel()}>Cancel</button>}
      </div>
    </div>
  </div>, document.fullscreenElement ?? document.body);
}
