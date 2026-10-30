import { InputShapeUtil } from '../../canvas/InputShapeUtil';
import { LayerShapeUtil } from '../../canvas/LayerShapeUtil';
import { ModuleShapeUtil } from '../../canvas/ModuleShapeUtil';
import { OutputShapeUtil } from '../../canvas/OutputShapeUtil';

jest.mock('../../canvas/graphQueryState', () => ({ graphOverlayScale: { get: () => ({ values: new Map() }) },
  graphSearchResult: { get: () => ({}) }, searchStateFor: () => undefined }));
jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: () => null, T: { number: {}, any: {} }, track: (component: unknown) => component,
}));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => 'inspection' }));

const create = (Util: unknown) => new (Util as new () => Record<string, () => unknown>)();

// Each card outlines itself when selected. Input and Output cards used to keep
// tldraw's blue selection box and resize handles while every other card hid them.
it.each([['Input', InputShapeUtil], ['Output', OutputShapeUtil], ['layer', LayerShapeUtil], ['module', ModuleShapeUtil]] as const)(
  'a selected %s card shows no tldraw selection box or handles', (_name, Util) => {
    const util = create(Util);
    expect(util.hideSelectionBoundsFg()).toBe(true);
    expect(util.hideResizeHandles()).toBe(true);
    expect(util.hideRotateHandle()).toBe(true);
  });

it.each([['Input', InputShapeUtil], ['Output', OutputShapeUtil], ['layer', LayerShapeUtil]] as const)(
  'a %s card keeps the size its content measures', (_name, Util) => {
    expect(create(Util).canResize()).toBe(false);
  });
