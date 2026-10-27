import type { Editor } from 'tldraw';
import type { ModelGraph } from '../types';
import { graphSvgBounds, graphToSvg } from '../services/graphSvg';
import { getCanvasTheme } from '../styles/canvasTheme';
import { prefersReducedMotion } from './reducedMotion';

type Rect = { x: number; y: number; w: number; h: number };
export interface FocusFrameSnapshot { face: HTMLElement; frame: Rect; anchor?: Rect }
let sequence = 0;

/** Inert visual copies never become another editor or own keyboard focus. */
export function captureFocusFrame(editor: Editor, graph: ModelGraph, anchorId?: string): FocusFrameSnapshot | undefined {
  // Bounded input delay: huge scenes use the surface's short fade instead of
  // cloning thousands of offscreen cards or serializing a large SVG on click.
  if (graph.nodes.filter(node => !node.hidden).length > 400) return undefined;
  const canvas = editor.getContainer?.();
  const frame = canvas?.closest<HTMLElement>('.module-focus-frame');
  const shell = frame?.closest<HTMLElement>('.module-focus-shell');
  if (!canvas || !frame || !shell) return undefined;
  const origin = shell.getBoundingClientRect(), rect = frame.getBoundingClientRect();
  const face = frame.cloneNode(true) as HTMLElement;
  face.classList.add('module-flight-face'); face.setAttribute('aria-hidden', 'true'); face.inert = true;
  Object.assign(face.style, { width: `${rect.width}px`, height: `${rect.height}px`, margin: '0', flex: 'none', position: 'absolute' });
  face.querySelectorAll('.module-focus-options,.module-flight,.graph-focus-overlay,.tl-selection__fg').forEach(node => node.remove());
  // SVG overview revokes its previous blob on scene replacement. Give the copy
  // an independent data URL so its image remains valid throughout the flight.
  face.querySelectorAll<HTMLImageElement>('[data-testid="graph-svg-layer"] img').forEach(img => {
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(graphToSvg(graph.nodes, graph.edges, { theme: getCanvasTheme() }))}`;
  });
  const prefix = `module-flight-${++sequence}-`;
  const ids = new Map<string, string>();
  face.querySelectorAll<HTMLElement>('[id]').forEach(node => {
    const old = node.id; node.id = prefix + old; ids.set(old, node.id);
  });
  face.querySelectorAll('*').forEach(item => {
    item.removeAttribute('data-testid');
    for (const attr of Array.from(item.attributes)) if (attr.value.includes('url(#')) {
      item.setAttribute(attr.name, attr.value.replace(/url\(#([^)]*)\)/g, (match, id) => ids.has(id) ? `url(#${ids.get(id)})` : match));
    }
  });
  const bounds = anchorId && graphSvgBounds(graph.nodes).find(box => box.id === anchorId);
  const camera = editor.getCamera(), canvasBox = canvas.getBoundingClientRect();
  return { face, frame: { x: rect.x - origin.x, y: rect.y - origin.y, w: rect.width, h: rect.height },
    ...(bounds ? { anchor: { x: canvasBox.x - origin.x + (bounds.x + camera.x) * camera.z,
      y: canvasBox.y - origin.y + (bounds.y + camera.y) * camera.z, w: bounds.w * camera.z, h: bounds.h * camera.z } } : {}) };
}

/** The same one-world projection as the approved HTML. Only this overlay moves;
 * the real destination and tldraw's measured container stay flat underneath. */
export function playModuleFocusMotion(shell: HTMLElement, before: FocusFrameSnapshot, after: FocusFrameSnapshot,
  inward: boolean, direction: number, done: () => void): () => void {
  const flight = document.createElement('div'); flight.className = 'module-flight';
  flight.inert = true; flight.setAttribute('aria-hidden', 'true'); shell.append(flight);
  const animations: Animation[] = [];
  let finished = false;
  const finish = () => { if (finished) return; finished = true; animations.forEach(animation => animation.cancel()); flight.remove(); done(); };
  const reduced = prefersReducedMotion();
  const duration = reduced ? 140 : direction ? 260 : inward ? 600 : 420;
  const run = (node: HTMLElement, frames: Keyframe[]) => {
    const animation = node.animate(frames, { duration, easing: 'cubic-bezier(.22,.7,.2,1)', fill: 'forwards' });
    animations.push(animation); return animation;
  };
  const place = (snapshot: FocusFrameSnapshot) => Object.assign(snapshot.face.style, { left: `${snapshot.frame.x}px`, top: `${snapshot.frame.y}px` });
  const box = inward ? before.anchor : after.anchor;
  let completion: Animation;
  if (reduced || direction || !box) {
    place(before); place(after); flight.append(before.face, after.face);
    const dx = reduced ? 0 : direction * 60;
    run(before.face, [{ opacity: 1, transform: 'translateX(0)' }, { opacity: 0, transform: `translateX(${-dx}px)` }]);
    completion = run(after.face, [{ opacity: 0, transform: `translateX(${dx}px)` }, { opacity: 1, transform: 'translateX(0)' }]);
  } else {
    const parent = inward ? before : after, child = inward ? after : before;
    const target = child.frame, tx = target.x + target.w / 2, ty = target.y + target.h / 2;
    const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
    const scale = Math.max(target.w / box.w, target.h / box.h);
    flight.style.perspectiveOrigin = `${tx}px ${ty}px`;
    const world = document.createElement('div'); world.className = 'module-flight-world';
    const plane = document.createElement('div'); plane.className = 'module-flight-child';
    plane.style.transform = `translate(${cx - target.w / (2 * scale)}px,${cy - target.h / (2 * scale)}px) scale(${1 / scale})`;
    place(parent); Object.assign(child.face.style, { left: '0', top: '0' });
    plane.append(child.face); world.append(parent.face, plane); flight.append(world);
    const end = `translate3d(${tx - cx}px,${ty - cy}px,${1200 * (1 - 1 / scale)}px)`;
    const reverse = (frames: Keyframe[]) => [...frames].reverse().map(frame => ({ ...frame, offset: 1 - Number(frame.offset) }));
    const parentFade = [{ opacity: 1, offset: 0 }, { opacity: .6, offset: .4 }, { opacity: 0, offset: 1 }];
    const childFade = [{ opacity: 0, offset: 0 }, { opacity: 0, offset: .15 }, { opacity: 1, offset: .8 }, { opacity: 1, offset: 1 }];
    run(parent.face, inward ? parentFade : reverse(parentFade)); run(plane, inward ? childFade : reverse(childFade));
    completion = run(world, inward ? [{ transform: 'translate3d(0,0,0)' }, { transform: end }] : [{ transform: end }, { transform: 'translate3d(0,0,0)' }]);
    const outline = document.createElement('div'); outline.className = 'module-flight-outline'; flight.append(outline);
    const frames = Array.from({ length: 81 }, (_, i) => {
      const u = i / 80, magnify = 1 / (1 - u * (1 - 1 / scale)), morph = Math.min(1, u / .35);
      const w = (box.w + (target.w / scale - box.w) * morph) * magnify;
      const h = (box.h + (target.h / scale - box.h) * morph) * magnify;
      return { offset: u, left: `${tx + (cx - tx) * (1 - u) * magnify - w / 2}px`,
        top: `${ty + (cy - ty) * (1 - u) * magnify - h / 2}px`, width: `${w}px`, height: `${h}px` };
    });
    run(outline, inward ? frames : reverse(frames));
  }
  // Any explicit interaction ends the visual copy before the real action runs.
  const interrupt = () => finish();
  shell.addEventListener('pointerdown', interrupt, true); shell.addEventListener('keydown', interrupt, true);
  completion.finished.then(finish, () => undefined);
  return () => { shell.removeEventListener('pointerdown', interrupt, true); shell.removeEventListener('keydown', interrupt, true); finish(); };
}
