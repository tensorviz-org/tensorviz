import React, { useSyncExternalStore } from 'react';
import { createPortal } from 'react-dom';
import { MessageSquare, X } from 'lucide-react';
import { LOCAL_BROWSER_FEEDBACK_MAX_LENGTH, type LocalBrowserFeedbackKind } from './protocol.generated';
import type { FeedbackDraft, LocalBrowserFeedbackClient } from './localBrowserFeedbackClient';

const KINDS: Array<{ value: LocalBrowserFeedbackKind; label: string }> = [
  { value: 'bug', label: 'Something is wrong' }, { value: 'idea', label: 'Idea' }, { value: 'other', label: 'Other' },
];

/** Feedback to the tensorViz team: an icon in the Canvas header and a short form beside it.
 *  Nothing leaves until the member presses Send; the graph details go only when ticked. */
export default function FeedbackButton({ client, graph }: {
  client: LocalBrowserFeedbackClient;
  /** The open model's class and block count, offered to include. */
  graph: { model: string; nodes: number } | null;
}) {
  const [open, setOpen] = React.useState(false);
  const button = React.useRef<HTMLButtonElement>(null);
  // The compact header moves this button; a finished result never reopens in its new place.
  React.useEffect(() => () => client.reset(), [client]);
  return <>
    <button ref={button} type="button" className="local-canvas-control" aria-label="Send feedback" aria-expanded={open}
      title="Send feedback to the tensorViz team" data-testid="feedback-button" onClick={() => setOpen(value => !value)}>
      <MessageSquare size={15} aria-hidden="true" />
    </button>
    {open && <FeedbackForm anchor={button} client={client} graph={graph} close={() => { client.reset(); setOpen(false); }} />}
  </>;
}

function FeedbackForm({ anchor, client, graph, close }: {
  anchor: React.RefObject<HTMLButtonElement | null>;
  client: LocalBrowserFeedbackClient;
  graph: { model: string; nodes: number } | null;
  close: () => void;
}) {
  const state = useSyncExternalStore(client.subscribe, () => client.state);
  const [kind, setKind] = React.useState<LocalBrowserFeedbackKind>('bug');
  const [message, setMessage] = React.useState('');
  const [includeGraph, setIncludeGraph] = React.useState(true);
  const [position, setPosition] = React.useState<React.CSSProperties | null>(null);
  React.useLayoutEffect(() => {
    const place = () => {
      const rect = anchor.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(340, window.innerWidth - 16);
      setPosition({ width, top: rect.bottom + 8, left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)) });
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); };
  }, [anchor]);
  React.useEffect(() => {
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape' && state.phase !== 'sending') close(); };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [close, state.phase]);
  if (!position) return null;
  const sending = state.phase === 'sending';
  const send = (event: React.FormEvent) => {
    event.preventDefault();
    const draft: FeedbackDraft = { kind, message, graph: includeGraph ? graph : null };
    client.send(draft);
  };
  return createPortal(<div role="dialog" aria-label="Send feedback" data-testid="feedback-form"
    className="local-canvas-popover local-canvas-action-panel local-canvas-publish local-canvas-feedback" style={position}>
    <div className="local-canvas-popover-heading"><strong>{state.phase === 'sent' ? 'Feedback sent' : 'Send feedback'}</strong>
      <button type="button" className="local-canvas-control" aria-label="Close" disabled={sending} onClick={close}><X size={15} /></button>
    </div>
    {state.phase === 'sent'
      ? <div className="local-canvas-publish-body"><p role="status">{state.detail}</p>
        <div className="local-canvas-publish-actions"><button type="button" className="local-canvas-control" onClick={close}>Close</button></div></div>
      : <form className="local-canvas-publish-body" onSubmit={send}>
        <div className="local-canvas-feedback-kinds" role="radiogroup" aria-label="Kind of feedback">
          {KINDS.map(option => <button key={option.value} type="button" role="radio" aria-checked={kind === option.value}
            className={`local-canvas-control${kind === option.value ? ' is-active' : ''}`} disabled={sending}
            onClick={() => setKind(option.value)}>{option.label}</button>)}
        </div>
        <textarea className="local-canvas-feedback-text" aria-label="Your feedback" autoFocus rows={5} disabled={sending}
          maxLength={LOCAL_BROWSER_FEEDBACK_MAX_LENGTH} placeholder="What happened, or what would help?"
          value={message} onChange={event => setMessage(event.target.value)} />
        {graph && <label className="local-canvas-feedback-check">
          <input type="checkbox" checked={includeGraph} disabled={sending} onChange={event => setIncludeGraph(event.target.checked)} />
          <span>Include the model name ({graph.model}) and its block count</span>
        </label>}
        <p className="local-canvas-publish-note">Goes to the tensorViz team with your tensorViz version and operating system. No source code or file paths.</p>
        {state.phase === 'failed' && <p role="alert">{state.detail}</p>}
        <div className="local-canvas-publish-actions">
          <button type="submit" className="local-canvas-control local-canvas-publish-primary" disabled={sending || !message.trim()}>
            {sending ? 'Sending…' : 'Send'}</button>
        </div>
      </form>}
  </div>, document.fullscreenElement ?? document.body);
}
