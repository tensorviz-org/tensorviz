import React, { useState } from 'react';
import { GraphFile, parseGraphFile } from '../services/graphFile';
import { annotationId, captureAnchor, GraphTarget, NamedGraphView, SavedGraphView, stepStatus, targetLabel } from '../services/graphCollaboration';

interface Props {
  file: GraphFile;
  onChange: (file: GraphFile) => void;
  capture: () => SavedGraphView;
  restore: (view: SavedGraphView) => void;
  onClose: () => void;
  onError: (error: unknown) => void;
}
export default function SavedViewsPanel({ file, onChange, capture, restore, onClose, onError }: Props): React.ReactElement {
  const { savedViews, openingViewId } = file.collaboration;
  const [activeId, setActiveId] = useState<string | null>(openingViewId);
  const [editId, setEditId] = useState<string | null>(null);
  const active = savedViews.find(view => view.id === activeId);
  const edited = savedViews.find(view => view.id === editId);
  const change = (views: NamedGraphView[], opening = openingViewId) => onChange({ ...file, collaboration: { ...file.collaboration, savedViews: views, openingViewId: opening } });
  const record = () => {
    const view = capture();
    const targets: GraphTarget[] = view.selectedEdge ? [view.selectedEdge] : view.selectedNodeIds.map(nodeId => ({ kind: 'node', nodeId }));
    if (!targets.length) throw new Error('Select a layer, block or real connection before sharing a view.');
    return { view, anchors: targets.map(target => captureAnchor(file, target)), title: targets.map(targetLabel).join(', ').slice(0, 160) };
  };
  const share = () => {
    try {
      if (savedViews.length >= 50) throw new Error('A graph file can contain up to 50 saved views.');
      const item = { id: annotationId(), note: '', ...record() };
      const next = { ...file, collaboration: { ...file.collaboration, savedViews: [...savedViews, item], openingViewId: item.id } };
      parseGraphFile(JSON.stringify(next)); onChange(next); setActiveId(item.id); setEditId(item.id);
    } catch (error) { onError(error); }
  };
  const move = (id: string, direction: number) => {
    const views = [...savedViews]; const from = views.findIndex(view => view.id === id); const to = from + direction;
    if (to < 0 || to >= views.length) return;
    [views[from], views[to]] = [views[to], views[from]]; change(views);
  };
  const update = (id: string, patch: Partial<NamedGraphView>) => change(savedViews.map(view => view.id === id ? { ...view, ...patch } : view));
  return <aside className="graph-viewer-panel" aria-label="Saved views">
    <div className="graph-panel-heading"><h2>Saved views</h2><button aria-label="Close saved views" onClick={onClose}>×</button></div>
    <p className="graph-panel-hint">Select a layer or connection and arrange the graph. Share this view sets where the file opens. Download the graph file and send it to your teammate.</p>
    <button onClick={share}>Share this view</button>
    {savedViews.length === 0 && <p className="graph-panel-hint">No saved views yet.</p>}
    {active && <section aria-label="Current saved view" className="graph-guide-playback">
      <h3>{active.title || 'Untitled view'}</h3>
      {active.note && <p className="graph-note">{active.note}</p>}
      <p className="graph-panel-hint">{openingViewId === active.id ? 'This view opens first for recipients.' : 'Saved view'}</p>
      {stepStatus(file, active) !== 'unchanged' && <p role="status" className="graph-context-warning">Original context: {stepStatus(file, active)}. Review this view against the current architecture.</p>}
      <div className="graph-panel-actions"><button onClick={() => restore(active.view)}>Show saved view</button>
        <button onClick={() => setEditId(editId === active.id ? null : active.id)}>Edit view</button>
        <button onClick={() => change(savedViews, openingViewId === active.id ? null : active.id)}>{openingViewId === active.id ? 'Clear opening view' : 'Make opening view'}</button></div>
    </section>}
    <ul className="graph-guide-list">{savedViews.map(view => <li key={view.id}><button aria-current={activeId === view.id ? 'true' : undefined} onClick={() => { setActiveId(view.id); setEditId(null); restore(view.view); }}>
      {view.title || 'Untitled view'}{openingViewId === view.id ? ' · Opens first' : ''}{stepStatus(file, view) !== 'unchanged' ? ` · ${stepStatus(file, view)}` : ''}
    </button></li>)}</ul>
    {edited && <section className="graph-guide-editor" aria-label="Edit saved view">
      <label>View title<input maxLength={160} value={edited.title} onChange={event => update(edited.id, { title: event.target.value })} /></label>
      <label>View note<textarea rows={4} maxLength={4000} value={edited.note} onChange={event => update(edited.id, { note: event.target.value })} /></label>
      <div className="graph-panel-actions"><button disabled={savedViews[0].id === edited.id} onClick={() => move(edited.id, -1)}>Move view up</button>
        <button disabled={savedViews[savedViews.length - 1].id === edited.id} onClick={() => move(edited.id, 1)}>Move view down</button></div>
      <div className="graph-panel-actions"><button onClick={() => { try { const { view, anchors } = record(); update(edited.id, { view, anchors }); } catch (error) { onError(error); } }}>Update captured view</button>
        <button onClick={() => { change(savedViews.filter(view => view.id !== edited.id), openingViewId === edited.id ? null : openingViewId); setEditId(null); setActiveId(null); }}>Delete view</button>
        <button onClick={() => setEditId(null)}>Finish editing view</button></div>
    </section>}
    <p className="graph-panel-hint">Changes stay in this tab until you download the graph file. A localhost URL does not share your file.</p>
  </aside>;
}
