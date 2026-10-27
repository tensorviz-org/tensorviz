import type { FocusDirection, FocusOrientation } from '../services/moduleFocus';

export interface FocusCamera { x: number; y: number; z: number }
export interface ModuleFocusPresentation {
  scopeId: string | null;
  views: Array<{ id: string | null; camera?: FocusCamera; selected: string[]; orientation: FocusOrientation;
    direction?: FocusDirection; layoutKey?: string; positions: Array<[string, { x: number; y: number }]> }>;
}

/** Native retained state is presentation, never a source of graph identities. */
export function validModuleFocusPresentation(value: unknown): value is ModuleFocusPresentation {
  const state = value as ModuleFocusPresentation | undefined;
  const id = (value: unknown) => value === null || typeof value === 'string';
  return Boolean(state && id(state.scopeId) && Array.isArray(state.views) && state.views.every(view =>
    view && id(view.id) && ['auto', 'LR', 'TB'].includes(view.orientation) &&
    (view.direction === undefined || ['LR', 'TB'].includes(view.direction)) &&
    (view.layoutKey === undefined || typeof view.layoutKey === 'string') &&
    (!view.camera || ([view.camera.x, view.camera.y, view.camera.z].every(Number.isFinite) && view.camera.z > 0)) &&
    Array.isArray(view.selected) && view.selected.every(item => typeof item === 'string') &&
    Array.isArray(view.positions) && view.positions.every(entry => Array.isArray(entry) && entry.length === 2 &&
      typeof entry[0] === 'string' && entry[1] && Number.isFinite(entry[1].x) && Number.isFinite(entry[1].y))));
}
