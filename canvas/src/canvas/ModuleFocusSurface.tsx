import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowLeft, ChevronDown, ChevronRight } from 'lucide-react';
import type { ModelNode } from '../types';
import { enterableModuleIds, focusNodeName, moduleFocusPath } from '../services/moduleFocus';
import type { ModuleFocusController } from './useModuleFocus';
import { graphFocus, setTraceDirection } from './graphFocus';
import { selectedEdgeId } from './edgeSelection';
import { closeNodeParameters, parameterTarget } from './nodeParameters';
import { captureFocusFrame, playModuleFocusMotion } from './moduleFocusMotion';
import './moduleFocus.css';

export function ModuleFocusSurface({ focus, nodes, title, selectedId, children }: {
  focus: ModuleFocusController; nodes: ModelNode[]; title: string; selectedId: string | null; children: React.ReactNode;
}) {
  const currentCrumb = useRef<HTMLButtonElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement | null>(null);
  const [menu, setMenu] = useState<string | null>(null);
  const [motion, setMotion] = useState(true);
  const current = useRef(focus); current.current = focus;
  const enterable = React.useMemo(() => enterableModuleIds(nodes), [nodes]);
  const scope = focus.path[focus.path.length - 1];
  const selected = focus.graph.nodes.find(node => node.id === selectedId && !node.data.focusBoundary);
  const returnFocus = useRef<string | undefined>(undefined);
  useLayoutEffect(() => {
    setMenu(null);
    // External search/MCP navigation keeps its originating keyboard focus.
    if (focus.transition) {
      if (focus.active) currentCrumb.current?.focus({ preventScroll: true });
      else focus.editor?.getContainer().focus({ preventScroll: true });
      if (!focus.transition.to || moduleFocusPath(nodes, focus.transition.from).some(node => node.id === focus.transition!.to)) returnFocus.current = focus.transition.anchorId;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus.scopeId, focus.active]);
  useEffect(() => {
    if (focus.transition || !returnFocus.current) return;
    const node = nodes.find(node => node.id === returnFocus.current); returnFocus.current = undefined;
    const name = node?.data.captureLabel ?? node?.data.moduleInstanceId ?? node?.data.moduleTemplateId;
    const button = Array.from(root.current?.querySelectorAll<HTMLButtonElement>('button') ?? []).find(button => button.getAttribute('aria-label') === `Expand ${name}`);
    (button ?? focus.editor?.getContainer())?.focus({ preventScroll: true });
  }, [focus.transition, focus.editor, nodes]);
  // Keep the departing frame on top until destination shapes/camera are ready.
  useLayoutEffect(() => {
    const transition = focus.transition;
    if (!transition?.snapshot || focus.settled === transition.revision || !root.current || !motion) return;
    const cover = document.createElement('div'); cover.className = 'module-flight'; cover.inert = true;
    cover.setAttribute('aria-hidden', 'true');
    const face = transition.snapshot.face.cloneNode(true) as HTMLElement;
    Object.assign(face.style, { left: `${transition.snapshot.frame.x}px`, top: `${transition.snapshot.frame.y}px` });
    cover.append(face); root.current.append(cover);
    const dismiss = () => { cover.remove(); current.current.finishTransition(transition.revision); };
    const timer = setTimeout(dismiss, 1000);
    root.current.addEventListener('pointerdown', dismiss, true);
    root.current.addEventListener('keydown', dismiss, true);
    const shell = root.current;
    return () => { clearTimeout(timer); shell.removeEventListener('pointerdown', dismiss, true); shell.removeEventListener('keydown', dismiss, true); cover.remove(); };
  }, [focus.transition, focus.settled, motion]);
  useLayoutEffect(() => {
    const transition = focus.transition;
    if (!transition || focus.settled !== transition.revision) return;
    const controller = current.current;
    if (!motion || !root.current || !controller.editor || typeof Element.prototype.animate !== 'function') {
      controller.finishTransition(transition.revision); return;
    }
    if (!transition.snapshot) {
      const frame = root.current.querySelector('.module-focus-frame')!;
      const animation = frame.animate([{ opacity: .7 }, { opacity: 1 }], { duration: 140 });
      animation.finished.then(() => current.current.finishTransition(transition.revision), () => undefined);
      return () => { animation.cancel(); current.current.finishTransition(transition.revision); };
    }
    const after = captureFocusFrame(controller.editor, controller.graph, transition.anchorId);
    if (!after) { controller.finishTransition(transition.revision); return; }
    const inward = Boolean(transition.to && (!transition.from || moduleFocusPath(nodes, transition.to).some(node => node.id === transition.from)));
    return playModuleFocusMotion(root.current, transition.snapshot, after, inward, transition.direction, () => current.current.finishTransition(transition.revision));
    // The snapshot belongs to one navigation; selection/evidence updates do not restart it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus.transition, focus.settled, motion]);
  useEffect(() => {
    if (!menu) return;
    const close = (event: PointerEvent) => {
      if (!(event.target as Element).closest('.module-focus-crumb')) setMenu(null);
    };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, [menu]);
  const onKey = (event: React.KeyboardEvent) => {
    if (event.key !== 'Escape' || event.defaultPrevented || (event.target as Element).closest('input,textarea,select,dialog,[contenteditable="true"]')) return;
    if (menu) { event.preventDefault(); event.stopPropagation(); setMenu(null); trigger.current?.focus(); }
    else if (focus.active) {
      event.preventDefault(); event.stopPropagation();
      if (parameterTarget.get()) closeNodeParameters();
      else if (selectedEdgeId.get()) selectedEdgeId.set(null);
      else if (focus.editor && graphFocus(focus.editor).focus.get().direction !== 'local') setTraceDirection(focus.editor, 'local');
      else if (selected || focus.editor?.getSelectedShapeIds().length) focus.clearSelection();
      else focus.back();
    }
  };
  const childCount = focus.graph.nodes.filter(node => !node.data.focusBoundary).length;
  return <div ref={root} className={`module-focus-shell${focus.active ? ' module-focus-active' : ''}`} onKeyDownCapture={onKey}
    data-focus-scope={focus.scopeId ?? ''} data-focus-direction={focus.direction ?? ''}>
    {focus.active && <div className="module-focus-toolbar">
      <nav className="module-focus-path" aria-label="Module path">
        <button aria-label="Back to parent module" onClick={focus.back}><ArrowLeft size={14} />Back</button>
        <button onClick={() => focus.navigate(null)} title={`${title} overview`}>Overview</button>
        {focus.path.map(node => {
          const siblings = nodes.filter(sibling => sibling.data.parentId === node.data.parentId && enterable.has(sibling.id));
          const current = node.id === focus.scopeId;
          const tip = current && scope
            ? `${scope.data.layerType.displayName || scope.data.layerType.name} · ${childCount} direct children`
            : focusNodeName(node);
          return <React.Fragment key={node.id}><ChevronRight size={12} aria-hidden="true" />
            <span className="module-focus-crumb">
              <button ref={current ? currentCrumb : undefined} aria-current={current ? 'page' : undefined} onClick={() => focus.navigate(node.id)} title={tip}>{focusNodeName(node)}</button>
              {siblings.length > 1 && <button aria-label={`Switch sibling of ${focusNodeName(node)}`} aria-expanded={menu === node.id}
                onClick={event => { trigger.current = event.currentTarget; setMenu(menu === node.id ? null : node.id); }}><ChevronDown size={12} /></button>}
              {menu === node.id && <div className="module-focus-siblings" role="group" aria-label="Sibling modules">
                {siblings.map(sibling => <button key={sibling.id} aria-current={sibling.id === node.id ? 'true' : undefined}
                  onClick={() => { setMenu(null); if (sibling.id === node.id) trigger.current?.focus(); else focus.switchSibling(node.id, sibling.id); }}>{focusNodeName(sibling)}</button>)}
              </div>}
            </span>
          </React.Fragment>;
        })}
      </nav>
      {scope && <div className="module-focus-controls">
        <label className="module-focus-orientation">Flow<select aria-label="Module flow direction" value={focus.orientation}
          onChange={event => focus.setOrientation(event.target.value as 'auto' | 'LR' | 'TB')}>
          <option value="auto">Auto · {focus.direction === 'TB' ? 'vertical' : 'horizontal'}</option>
          <option value="LR">Horizontal →</option><option value="TB">Vertical ↓</option>
        </select></label>
      </div>}
    </div>}
    <section className="module-focus-frame" aria-label={scope ? `${focusNodeName(scope)} module` : 'Model graph'}>
      {children}
      {focus.active && <div className="module-focus-options">
        <span>One module in focus · zoom in to explore</span>
        <label><input type="checkbox" checked={motion} onChange={event => setMotion(event.target.checked)} />Motion</label>
      </div>}
    </section>
    <span className="module-focus-announcement" role="status">{focus.active && scope ? `${focusNodeName(scope)} in focus. Only immediate children are shown.` : ''}</span>
  </div>;
}
