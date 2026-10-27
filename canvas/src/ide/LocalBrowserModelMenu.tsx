import React from 'react';
import { ChevronDown, CloudUpload, Download, FolderOpen, Workflow } from 'lucide-react';
import { bridgeActive } from './bridge';
import type { LocalBrowserTargetIdentity } from './protocol.generated';

export default function LocalBrowserModelMenu({ target, exportDisabled, fileActionsAvailable = true }: {
  target: LocalBrowserTargetIdentity | null;
  exportDisabled: boolean;
  fileActionsAvailable?: boolean;
}) {
  const [open, setOpen] = React.useState(false);
  const root = React.useRef<HTMLDivElement>(null);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const id = React.useId();

  React.useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      trigger.current?.focus();
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [open]);

  const command = (name: string) => {
    window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: name } }));
    setOpen(false);
    trigger.current?.focus();
  };

  return <div ref={root} className="local-browser-model-menu">
    <button ref={trigger} type="button" className="local-browser-model-trigger"
      aria-label="Model and file actions" aria-expanded={open} aria-controls={id}
      title={target ? `${target.cls} · ${target.file}` : 'tensorViz'}
      onClick={() => setOpen(value => !value)}>
      <span className="local-browser-brand-mark"><Workflow size={16} aria-hidden="true" /></span>
      <span className="local-browser-model-name"><strong>{target && <span className="local-browser-product">tensorViz /</span>}{target?.cls ?? 'tensorViz'}</strong>
        {target && <small>{target.file}</small>}</span>
      <ChevronDown size={14} aria-hidden="true" />
    </button>
    {open && <div id={id} className="local-browser-model-actions" role="group" aria-label="Model actions">
      {target && <p>{target.file}</p>}
      {/* In VS Code a graph goes to the member's tensorViz account, where they share it;
          the browser page has no host to sign in through, so it saves a file instead. */}
      {fileActionsAvailable && <>{bridgeActive()
        ? <button type="button" disabled={exportDisabled} title="Add this graph to your tensorViz account, then choose who can see it"
          onClick={() => command('share-graph-file')}><CloudUpload size={15} aria-hidden="true" />Add to tensorViz</button>
        : <button type="button" disabled={exportDisabled} title="Save a read-only snapshot with the current layout"
          onClick={() => command('export-graph-file')}><Download size={15} aria-hidden="true" />Export graph</button>}
      <a href="./viewer/index.html" target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
        <FolderOpen size={15} aria-hidden="true" />Open graph file viewer</a></>}
    </div>}
  </div>;
}
