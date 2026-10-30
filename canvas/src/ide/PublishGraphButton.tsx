import React from 'react';
import { CloudUpload, Download } from 'lucide-react';
import { bridgeActive } from './bridge';

/** In VS Code, Publish adds the graph to the member's tensorViz account, where they
 *  share it; the browser page has no host to sign in through, so it saves a file instead. */
export default function PublishGraphButton({ disabled }: { disabled: boolean }) {
  const publish = bridgeActive();
  const command = publish ? 'share-graph-file' : 'export-graph-file';
  return <button type="button" className="local-canvas-control" disabled={disabled}
    aria-label={publish ? 'Publish graph' : 'Export graph'}
    title={publish ? 'Add this graph to your tensorViz account, then choose who can see it' : 'Save a read-only snapshot with the current layout'}
    onClick={() => window.dispatchEvent(new CustomEvent('builder-command', { detail: { command } }))}>
    {publish ? <CloudUpload size={15} aria-hidden="true" /> : <Download size={15} aria-hidden="true" />}
    <span className="local-canvas-control-label">{publish ? 'Publish' : 'Export graph'}</span>
  </button>;
}
