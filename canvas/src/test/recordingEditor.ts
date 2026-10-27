/** Headless editor seam for bridge tests. Each batch reads the original records
 * before storing ANY updates, matching Editor._updateShapes (not a sequential
 * merge stub). A regression test checks this contract against the installed API.
 */
export class RecordingEditor {
  shapes = new Map<string, any>();
  batches: any[][] = [];
  readonlyChanges: boolean[] = [];
  readonly = false;
  private ignoreShapeLock = false;
  private afterChange = new Set<(prev: any, next: any) => void>();
  private operationComplete = new Set<(source: string) => void>();
  private beforeDelete = new Set<(shape: any, source: string) => boolean | void>();
  private register = (handlers: Set<any>, handler: any) => {
    handlers.add(handler);
    return () => { handlers.delete(handler); };
  };
  sideEffects = {
    registerAfterChangeHandler: (_type: string, handler: any) => this.register(this.afterChange, handler),
    registerOperationCompleteHandler: (handler: any) => this.register(this.operationComplete, handler),
    registerBeforeDeleteHandler: (_type: string, handler: any) => this.register(this.beforeDelete, handler),
  };
  store = { listen: () => () => {} };
  on = jest.fn();
  off = jest.fn();
  getCamera = () => ({ x: 0, y: 0, z: 1 });
  getViewportScreenBounds = () => ({ x: 0, y: 0, width: 1600, height: 900 });
  updateViewportScreenBounds = jest.fn();
  getCameraOptions = () => ({ zoomSteps: [0.05, 0.1, 0.5, 1, 2, 4] });
  setCameraOptions = jest.fn();
  screenToPage = (point: { x: number; y: number }) => point;
  getShape = (id: string) => this.shapes.get(id);
  getCurrentPageShapes = () => Array.from(this.shapes.values());
  getCurrentPageBounds = () => undefined;
  getSelectedShapeIds = () => [];
  getCanUndo = () => false;
  getCanRedo = () => false;
  getIsReadonly = () => this.readonly;
  updateInstanceState = ({ isReadonly }: { isReadonly: boolean }) => {
    this.readonly = isReadonly;
    this.readonlyChanges.push(isReadonly);
  };
  run = (apply: () => void, options?: { ignoreShapeLock?: boolean; history?: string }) => {
    const previous = this.ignoreShapeLock;
    this.ignoreShapeLock ||= options?.ignoreShapeLock ?? false;
    try { apply(); } finally { this.ignoreShapeLock = previous; }
  };
  createShape = (shape: any) => this.createShapes([shape]);
  createShapes = (shapes: any[]) => {
    if (this.readonly) return;
    shapes.forEach(shape => this.shapes.set(shape.id, { ...shape }));
  };
  updateShape = (partial: any) => this.updateShapes([partial]);
  updateShapes = (partials: any[]) => {
    if (this.readonly) return;
    this.batches.push(partials);
    const updates = partials.flatMap(partial => {
      const previous = this.getShape(partial.id);
      if (!previous || (previous.isLocked && !this.ignoreShapeLock && partial.isLocked !== false)) return [];
      return [{ previous, next: { ...previous, ...partial,
        props: { ...previous.props, ...partial.props } } }];
    });
    updates.forEach(({ next }) => this.shapes.set(next.id, next));
    updates.forEach(({ previous, next }) => this.afterChange.forEach(handler => handler(previous, next)));
  };
  deleteShapes = (ids: string[]) => {
    if (this.readonly) return;
    ids.forEach(id => {
      const shape = this.getShape(id);
      if (!shape || (shape.isLocked && !this.ignoreShapeLock)) return;
      if (Array.from(this.beforeDelete).some(handler => handler(shape, 'user') === false)) return;
      this.shapes.delete(id);
    });
  };
  completeUserOperation = () => this.operationComplete.forEach(handler => handler('user'));
  sendToBack = () => {};
  setCurrentTool = () => {};
  zoomToFit = jest.fn();
  zoomToBounds = jest.fn();
  setCamera = jest.fn();
}
