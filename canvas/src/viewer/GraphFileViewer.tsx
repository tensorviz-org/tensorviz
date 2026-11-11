import React, { useCallback, useEffect, useRef, useState } from 'react';
import { GraphFile, GRAPH_FILE_MAX_BYTES, parseGraphFile } from '../services/graphFile';
import { downloadGraphFile } from '../ide/graphFileExport';
import GraphFileCanvas, { GraphFileCanvasHandle } from './GraphFileCanvas';
import WalkthroughPanel from './WalkthroughPanel';
import SavedViewsPanel from './SavedViewsPanel';
import ComparisonPanel, { ComparisonSide } from './ComparisonPanel';
import { comparisonCounterpart } from '../services/capturedGraphComparison';
import DiscussionPanel, { DiscussionDraft } from './DiscussionPanel';
import { annotationId, captureAnchor, GraphTarget, SavedGraphView, updateArchitecture } from '../services/graphCollaboration';
import './graphFileViewer.css';

// Imported files are owned by this viewer; annotation edits replace the file.
// Keep this boundary local: callers of ComparisonPanel may hold mutable inputs.
const ViewerComparisonPanel = React.memo(ComparisonPanel);

type Panel = 'walkthrough' | 'discussions' | 'views' | 'compare' | null;
const statusLabels = {
  static: 'Static snapshot · model not executed',
  traced: 'Traced snapshot · recorded shapes, not revalidated here',
  stale: 'Stale snapshot · freshness unconfirmed at export',
};
export default function GraphFileViewer(): React.ReactElement {
  const [file, setFile] = useState<GraphFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [revision, setRevision] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [pendingFile, setPendingFile] = useState<GraphFile | null>(null);
  const [panel, setPanel] = useState<Panel>(null);
  const [comparison, setComparison] = useState<GraphFile | null>(null);
  const [comparisonSide, setComparisonSide] = useState<ComparisonSide>('after');
  const [initialTarget, setInitialTarget] = useState<GraphTarget | undefined>();
  const comparisonViews = useRef<Partial<Record<ComparisonSide, SavedGraphView>>>({});
  const [selectedTarget, setSelectedTarget] = useState<GraphTarget | null>(null);
  const [draft, setDraft] = useState<DiscussionDraft | null>(null);
  const [authorName, setAuthorName] = useState('');
  const unsaved = dirty || Boolean(draft?.body.trim());
  const fileRef = useRef(file); fileRef.current = file;
  const draftRef = useRef(draft); draftRef.current = draft;
  const canvasRef = useRef<GraphFileCanvasHandle | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const comparisonInputRef = useRef<HTMLInputElement | null>(null);
  const updateInputRef = useRef<HTMLInputElement | null>(null);
  const requestRef = useRef(0);
  const dirtyRef = useRef(unsaved); dirtyRef.current = unsaved;
  useEffect(() => () => { requestRef.current += 1; }, []);
  useEffect(() => {
    if (!unsaved) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [unsaved]);
  const showError = useCallback((error: unknown) => setError(error instanceof Error ? error.message : 'This action could not be completed.'), []);
  const clearComparison = () => { setComparison(null); setComparisonSide('after'); setInitialTarget(undefined); comparisonViews.current = {}; };
  const navigateComparison = useCallback((side: ComparisonSide, target?: GraphTarget, followSelection = true) => {
    try {
      if (side === comparisonSide) { if (target) canvasRef.current?.focusTarget(target); return true; }
      if (canvasRef.current) {
        const view = canvasRef.current.captureView(); comparisonViews.current[comparisonSide] = view;
        const selected = view.selectedEdge ?? (view.selectedNodeIds.length === 1 ? { kind: 'node' as const, nodeId: view.selectedNodeIds[0] } : undefined);
        if (!target && followSelection && selected && comparison && file) target = comparisonCounterpart(comparison, file, comparisonSide, selected);
      }
      setSelectedTarget(null); setInitialTarget(target); setComparisonSide(side); return true;
    } catch (error) { showError(error); return false; }
  }, [comparisonSide, comparison, file, showError]);
  const switchPanel = useCallback((next: Panel) => {
    if (comparisonSide === 'before' && !navigateComparison('after', undefined, false)) return;
    setPanel(next);
  }, [comparisonSide, navigateComparison]);
  const chooseComparison = useCallback(() => comparisonInputRef.current?.click(), []);
  const closeComparison = useCallback(() => switchPanel(null), [switchPanel]);
  const replace = (next: GraphFile) => {
    clearComparison();
    setFile(next); setDirty(false); setPendingFile(null); setPanel(next.collaboration.openingViewId ? 'views' : null); setDraft(null); setSelectedTarget(null); setError(null); setRevision(value => value + 1);
  };
  const change = (next: GraphFile) => { setFile(next); setDirty(true); setError(null); };
  const save = () => {
    if (!file) return;
    try {
      if (draft?.body.trim()) throw new Error('Add your draft message or discard it before downloading.');
      const valid = parseGraphFile(JSON.stringify(file));
      downloadGraphFile(valid); setFile(valid); setDirty(false); setError(null);
    } catch (error) { showError(error); }
  };
  const focus = (target: GraphTarget) => { try { canvasRef.current?.focusTarget(target); } catch (error) { showError(error); } };
  const startDiscussion = (target: GraphTarget, threadId?: string) => {
    if (!file) return;
    try {
      if (draft?.body.trim()) throw new Error('Add or discard your draft before starting another message.');
      const thread = file.collaboration.threads.find(thread => thread.id === threadId);
      if (!thread && file.collaboration.threads.length >= 200) throw new Error('A graph file can contain up to 200 discussions.');
      setDraft({ anchor: thread?.anchor ?? captureAnchor(file, target), threadId, author: authorName, body: '' });
      setPanel('discussions'); setError(null);
    } catch (error) { showError(error); }
  };
  const submitMessage = () => {
    if (!file || !draft?.body.trim()) return;
    try {
      const message = { id: annotationId(), author: draft.author.trim() || 'Anonymous', body: draft.body.trim(), createdAt: new Date().toISOString() };
      const threads = draft.threadId ? file.collaboration.threads.map(thread => thread.id === draft.threadId ? { ...thread, messages: [...thread.messages, message] } : thread)
        : [...file.collaboration.threads, { id: annotationId(), anchor: draft.anchor, resolved: false, messages: [message] }];
      const next = { ...file, collaboration: { ...file.collaboration, threads } };
      parseGraphFile(JSON.stringify(next)); change(next); setAuthorName(draft.author); setDraft(null);
    } catch (error) { showError(error); }
  };
  const open = (files: FileList | File[], mode: 'open' | 'update' | 'compare' = 'open') => {
    const request = ++requestRef.current;
    setPendingFile(null); setError(null); setLoading(false);
    if (files.length !== 1) { setError('Choose one .tensorviz.json graph file.'); return; }
    const source = files[0];
    if (source.size > GRAPH_FILE_MAX_BYTES) { setError('Graph files must be smaller than 64 MiB.'); return; }
    setLoading(true);
    const reader = new FileReader();
    reader.onload = () => {
      if (request !== requestRef.current) return;
      try {
        const next = parseGraphFile(String(reader.result));
        if (mode === 'compare' && fileRef.current) {
          setComparison(next); comparisonViews.current.before = undefined;
          setSelectedTarget(null); setInitialTarget(undefined); setComparisonSide('after'); setPanel('compare');
        } else if (mode === 'update' && fileRef.current) {
          if (draftRef.current?.body.trim()) throw new Error('Add or discard your draft message before updating the architecture.');
          const updated = parseGraphFile(JSON.stringify(updateArchitecture(fileRef.current, next)));
          clearComparison(); change(updated); setDraft(null); setSelectedTarget(null); setRevision(value => value + 1);
        } else if (dirtyRef.current) setPendingFile(next); else replace(next);
      } catch (error) { showError(error); }
      finally { setLoading(false); }
    };
    reader.onerror = () => {
      if (request !== requestRef.current) return;
      setLoading(false); setError('Could not read this file. Choose it again.');
    };
    reader.readAsText(source);
  };
  const choose = () => inputRef.current?.click();
  const openingView = file?.collaboration.savedViews.find(view => view.id === file.collaboration.openingViewId);
  const displayedFile = comparisonSide === 'before' && comparison ? comparison : file;
  const inferred = displayedFile?.capture?.capture.backend === 'jaxpr';
  const canvasInitialView = comparisonViews.current[comparisonSide] ?? (comparisonSide === 'after' ? openingView?.view : undefined);
  return <main className="graph-file-viewer" onDragOverCapture={event => {
    if (event.dataTransfer.types.includes('Files')) { event.preventDefault(); event.stopPropagation(); }
  }} onDropCapture={event => { event.preventDefault(); event.stopPropagation(); open(event.dataTransfer.files); }}>
    <header className="graph-viewer-header">
      <div><strong>tensorViz</strong><span>Graph viewer</span></div>
      {displayedFile && <div className="graph-viewer-title">{panel === 'compare' && comparison && <strong className="graph-comparison-side">{comparisonSide === 'before' ? 'Before' : 'After'}</strong>}<h1>{displayedFile.document.semantic.name}</h1><span title={`Exported ${displayedFile.exportedAt}`}>{inferred && displayedFile.status === 'traced' ? 'Traced snapshot · inferred shapes, not revalidated here' : statusLabels[displayedFile.status]}</span>
        {displayedFile.warningCount > 0 && <span>{displayedFile.warningCount} resolution warning{displayedFile.warningCount === 1 ? '' : 's'} recorded</span>}
        {displayedFile.capture ? <span>{displayedFile.capture.capture.status === 'complete' ? inferred ? 'Complete capture of the traced program' : 'Complete capture of one execution path' : 'Partial capture · see Captured details for limits'}</span>
          : displayedFile.status !== 'static' && <span>Legacy snapshot · ordered arguments, shared state and capture completeness unavailable</span>}
      </div>}
      <div className="graph-viewer-file-actions">{file && <>
        <span role="status" className="graph-save-state">{unsaved ? 'Unsaved notes' : 'No unsaved notes'}</span>
        <button aria-pressed={panel === 'views'} onClick={() => switchPanel(panel === 'views' ? null : 'views')}>Saved views</button>
        <button aria-pressed={panel === 'walkthrough'} onClick={() => switchPanel(panel === 'walkthrough' ? null : 'walkthrough')}>Walkthrough</button>
        <button aria-pressed={panel === 'discussions'} onClick={() => switchPanel(panel === 'discussions' ? null : 'discussions')}>Discussions</button>
        <button aria-pressed={panel === 'compare'} onClick={() => switchPanel(panel === 'compare' ? null : 'compare')}>Compare</button>
        <button disabled={comparisonSide === 'before'} title="Take architecture from another file and keep the current notes" onClick={() => updateInputRef.current?.click()}>Update architecture</button>
        <button onClick={save}>{comparisonSide === 'before' ? 'Download After graph file' : 'Download graph file'}</button>
      </>}<button onClick={choose}>{file ? 'Open another graph' : 'Open graph file'}</button></div>
      <input ref={inputRef} aria-label="Graph file" type="file" accept=".tensorviz.json,.json,application/json" hidden onChange={event => {
        if (event.target.files?.length) open(event.target.files); event.target.value = '';
      }} />
      <input ref={comparisonInputRef} aria-label="Comparison Before file" type="file" accept=".tensorviz.json,.json,application/json" hidden onChange={event => {
        if (event.target.files?.length) open(event.target.files, 'compare'); event.target.value = '';
      }} />
      <input ref={updateInputRef} aria-label="Replacement architecture file" type="file" accept=".tensorviz.json,.json,application/json" hidden onChange={event => {
        if (event.target.files?.length) open(event.target.files, 'update'); event.target.value = '';
      }} />
    </header>
    {loading && <div className="graph-viewer-notice" role="status">Opening graph…</div>}
    {error && <div className="graph-viewer-notice graph-viewer-error" role="alert">{error}{file && ' The previous graph is still open with your notes.'}</div>}
    {file ? <div className={`graph-viewer-workspace${panel === 'compare' ? ' is-comparing' : ''}`}>
      <GraphFileCanvas ref={canvasRef} key={`${revision}-${comparisonSide}`} file={displayedFile!} initialView={canvasInitialView} initialTarget={initialTarget} focusPanelWidth={panel === 'compare' ? 452 : panel ? 382 : 0} panelOpen={panel !== null} onTargetChange={setSelectedTarget} onDiscuss={target => { setSelectedTarget(target); setPanel('discussions'); }} />
      {panel === 'compare' && <ViewerComparisonPanel before={comparison} after={file} side={comparisonSide} onChoose={chooseComparison} onNavigate={navigateComparison} onClose={closeComparison} />}
      <div hidden={panel !== 'views'}><SavedViewsPanel key={`views-${revision}`} file={file} onChange={change} onClose={() => setPanel(null)} onError={showError}
        capture={() => { if (!canvasRef.current) throw new Error('The canvas is still loading.'); return canvasRef.current.captureView(); }}
        restore={view => canvasRef.current?.restoreView(view)} /></div>
      <div hidden={panel !== 'walkthrough'}><WalkthroughPanel key={`guide-${revision}`} file={file} onChange={change} onClose={() => setPanel(null)} onError={showError}
        capture={() => { if (!canvasRef.current) throw new Error('The canvas is still loading.'); return canvasRef.current.captureView(); }}
        restore={view => canvasRef.current?.restoreView(view)} /></div>
      <div hidden={panel !== 'discussions'}><DiscussionPanel file={file} selected={selectedTarget} draft={draft} onDraft={setDraft} onStart={startDiscussion}
        onSubmit={submitMessage} onFocus={focus} onClose={() => setPanel(null)}
        onResolve={(id, resolved) => change({ ...file, collaboration: { ...file.collaboration, threads: file.collaboration.threads.map(thread => thread.id === id ? { ...thread, resolved } : thread) } })} /></div>
    </div> : <section className="graph-viewer-empty">
      <div className="graph-viewer-mark" aria-hidden="true">◇</div><h1>Explore a saved model</h1>
      <p>Drop a .tensorviz.json file here, or choose one from your computer.</p><button onClick={choose}>Choose graph file</button>
      <p className="graph-viewer-hint">Your file stays in this browser. No upload, account or Python setup.</p>
      <p className="graph-viewer-hint">Create a file with <strong>Export graph</strong> in tensorViz’s live model canvas.</p>
    </section>}
    {pendingFile && <div className="graph-viewer-dialog-backdrop"><section role="alertdialog" aria-modal="true" aria-labelledby="graph-unsaved-title" className="graph-viewer-dialog" onKeyDown={event => {
      if (event.key === 'Escape') setPendingFile(null);
      if (event.key === 'Tab') {
        const buttons = event.currentTarget.querySelectorAll('button');
        const first = buttons[0]; const last = buttons[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    }}>
      <h2 id="graph-unsaved-title">Keep your unsaved notes?</h2><p>You chose “{pendingFile.document.semantic.name}”. Download the current graph file first, or discard your changes in this tab.</p>
      <div className="graph-panel-actions"><button autoFocus onClick={() => setPendingFile(null)}>Keep editing</button><button onClick={() => replace(pendingFile)}>Discard changes and open</button></div>
    </section></div>}
  </main>;
}
