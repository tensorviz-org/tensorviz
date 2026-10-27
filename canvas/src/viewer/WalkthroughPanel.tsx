import React, { useState } from 'react';
import { GraphFile } from '../services/graphFile';
import { annotationId, captureAnchor, SavedGraphView, stepStatus, WalkthroughStep } from '../services/graphCollaboration';

interface Props {
  file: GraphFile;
  onChange: (file: GraphFile) => void;
  capture: () => SavedGraphView;
  restore: (view: SavedGraphView) => void;
  onClose: () => void;
  onError: (error: unknown) => void;
}
export default function WalkthroughPanel({ file, onChange, capture, restore, onClose, onError }: Props): React.ReactElement {
  const guide = file.collaboration.walkthrough;
  const [editing, setEditing] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeIndex = guide.steps.findIndex(step => step.id === activeId);
  const active = guide.steps[activeIndex];
  const edited = guide.steps.find(step => step.id === editId);
  const change = (steps: WalkthroughStep[], title = guide.title) => onChange({ ...file,
    collaboration: { ...file.collaboration, walkthrough: { title, steps } } });
  const update = (id: string, patch: Partial<WalkthroughStep>) => change(guide.steps.map(step => step.id === id ? { ...step, ...patch } : step));
  const recorded = () => {
    const view = capture();
    if (!view.selectedNodeIds.length) throw new Error('Select a layer or block on the graph before capturing a walkthrough step.');
    return { view, anchors: view.selectedNodeIds.map(nodeId => captureAnchor(file, { kind: 'node', nodeId })) };
  };
  const add = () => {
    try {
      if (guide.steps.length >= 50) throw new Error('A walkthrough can contain up to 50 steps.');
      const saved = recorded(); const id = annotationId();
      change([...guide.steps, { id, title: saved.view.selectedNodeIds.join(', ').slice(0, 160), note: '', ...saved }]);
      setEditId(id); setEditing(true);
    } catch (error) { onError(error); }
  };
  const play = (step: WalkthroughStep) => { setActiveId(step.id); restore(step.view); };
  const move = (id: string, direction: number) => {
    const steps = [...guide.steps]; const index = steps.findIndex(step => step.id === id);
    const destination = index + direction;
    if (destination < 0 || destination >= steps.length) return;
    [steps[index], steps[destination]] = [steps[destination], steps[index]];
    change(steps);
  };
  return <aside className="graph-viewer-panel" aria-label="Architecture walkthrough">
    <div className="graph-panel-heading"><h2>{guide.title || 'Walkthrough'}</h2><button aria-label="Close walkthrough" onClick={onClose}>×</button></div>
    <p className="graph-panel-hint">Follow the author’s views, or pan and explore freely between steps.</p>
    {active ? <section aria-label="Walkthrough playback" className="graph-guide-playback">
      <span>Step {activeIndex + 1} of {guide.steps.length}</span><h3>{active.title || 'Untitled step'}</h3>
      <p className="graph-note">{active.note || 'No note for this step.'}</p>
      {stepStatus(file, active) !== 'unchanged' && <p role="status" className="graph-context-warning">Original context: {stepStatus(file, active)}. This saved view may no longer match the architecture.</p>}
      <div className="graph-panel-actions">
        <button disabled={activeIndex === 0} onClick={() => play(guide.steps[activeIndex - 1])}>Previous</button>
        <button disabled={activeIndex === guide.steps.length - 1} onClick={() => play(guide.steps[activeIndex + 1])}>Next</button>
        <button onClick={() => setActiveId(null)}>Exit walkthrough</button>
      </div><button onClick={() => restore(active.view)}>Show saved view</button>
    </section> : guide.steps.length > 0 ? <button onClick={() => play(guide.steps[0])}>Start walkthrough</button>
      : <p>No walkthrough yet. Select a layer, arrange the canvas and add your first step.</p>}
    {guide.steps.length > 0 && <ol className="graph-guide-list">{guide.steps.map((step, index) => <li key={step.id}>
      <button aria-current={step.id === activeId ? 'step' : undefined} onClick={() => editing ? setEditId(step.id) : play(step)}>
        {index + 1}. {step.title || 'Untitled step'}{stepStatus(file, step) !== 'unchanged' && <small> · {stepStatus(file, step)}</small>}
      </button>
    </li>)}</ol>}
    <div className="graph-panel-actions"><button aria-pressed={editing} onClick={() => setEditing(!editing)}>{editing ? 'Finish editing' : 'Edit walkthrough'}</button><button onClick={add}>Add current view</button></div>
    {editing && <section className="graph-guide-editor" aria-label="Edit walkthrough">
      <label>Walkthrough title<input value={guide.title} maxLength={160} onChange={event => change(guide.steps, event.target.value)} /></label>
      {edited && <>
        <label>Step title<input value={edited.title} maxLength={160} onChange={event => update(edited.id, { title: event.target.value })} /></label>
        <label>Step note<textarea value={edited.note} maxLength={4000} rows={5} placeholder="Explain the design decision behind this view…" onChange={event => update(edited.id, { note: event.target.value })} /></label>
        <button onClick={() => { try { update(edited.id, recorded()); } catch (error) { onError(error); } }}>Update saved view</button>
        <div className="graph-panel-actions"><button disabled={guide.steps[0].id === edited.id} onClick={() => move(edited.id, -1)}>Move up</button><button disabled={guide.steps[guide.steps.length - 1].id === edited.id} onClick={() => move(edited.id, 1)}>Move down</button>
          <button onClick={() => { change(guide.steps.filter(step => step.id !== edited.id)); setEditId(null); if (activeId === edited.id) setActiveId(null); }}>Delete step</button></div>
      </>}
      <p className="graph-panel-hint">Changes stay in this tab until you download the graph file.</p>
    </section>}
  </aside>;
}
