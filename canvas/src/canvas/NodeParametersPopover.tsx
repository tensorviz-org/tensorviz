/**
 * The parameter editor, anchored to the node it edits.
 *
 * This is the Inspector's Properties tab and nothing else from it: the catalogue-
 * driven fields were the one thing that panel held which exists nowhere else on
 * the canvas (shapes are on the cards, the quick actions are on the toolbar, the
 * degree counts are the edges you can see). The panel itself is gone.
 *
 * Same slot and same rules as the edge-insert picker, for the same reasons
 * documented at length in EdgeInsertLayer: `InFrontOfTheCanvas` paints AFTER the
 * shapes, so the popover cannot disappear behind a card it overlaps; it is
 * outside the camera transform, so points are projected with
 * `editor.pageToViewport` (NOT pageToScreen, which double-counts the container
 * offset) and it subscribes to the camera so it stays glued to its node while the
 * canvas moves.
 *
 * It opens on a drag-free click only — the gesture TldrawCanvas already
 * distinguishes for `onShapeClick` — which is what the docked panel got wrong:
 * tldraw's SelectTool selects on pointer-down, so a selection-driven panel opened
 * every time a card was picked up to be moved.
 *
 * Writes go out as a `set-parameters` builder command, the same one the panel
 * sent, so the IDE still routes the edit into the .py rather than the canvas.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TLShapeId, useEditor, useValue } from 'tldraw';
import { X } from 'lucide-react';
import ParameterFields from '../components/ui2/ParameterFields';
import { commitParameters } from '../components/ui2/parameterValues';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import { inspectorDraftParameters } from '../utils/inspectorDraft';
import { isModelShape } from './canvasShapePolicy';
import { shapeIdToNodeId } from './canvasShapeId';
import { closeNodeParameters, openNodeParameters, parameterTarget } from './nodeParameters';
import { ModelNodeData } from '../types';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { LocalSourceEditor } from './LocalSourceEditor';

/** How long an edit rests before it is written, so dragging a range handle is one request. */
const COMMIT_DEBOUNCE_MS = 450;

interface EditorProps {
  shapeId: string;
  data: ModelNodeData;
  screen: { x: number; y: number };
  /** True when the popover would otherwise run off the right edge of the viewport. */
  flip: boolean;
}

const Editor: React.FC<EditorProps> = ({ shapeId, data, screen, flip }) => {
  const editor = useEditor();
  const nodeId = shapeIdToNodeId(shapeId);
  const [draft, setDraft] = useState<Record<string, string>>(() =>
    inspectorDraftParameters(data.parameters));
  const baselineRef = useRef<Record<string, string>>(draft);

  // Assigned in render rather than depended on: the commit needs the node's real
  // parameter values, but making them an effect DEPENDENCY restarts the debounce
  // on every inference tick, which starves the write entirely.
  const currentRef = useRef<Record<string, unknown>>({});
  currentRef.current = data.parameters || {};

  const pendingRef = useRef<Record<string, string> | null>(null);
  const flush = useCallback(() => {
    const pending = pendingRef.current;
    if (!pending) return;
    pendingRef.current = null;
    const baseline = baselineRef.current;
    baselineRef.current = pending;
    dispatchBuilderCommand('set-parameters', {
      nodeId,
      // Only the keys that MOVED. The rest are taken from the node itself, so
      // typing in one field cannot retype another, and the internal render props
      // the fields hide are not deleted along the way. See parameterValues.ts.
      parameters: commitParameters(currentRef.current, baseline, pending),
    });
  }, [nodeId]);

  useEffect(() => {
    pendingRef.current = null;
    if (JSON.stringify(draft) === JSON.stringify(baselineRef.current)) return;
    pendingRef.current = draft;
    const timer = window.setTimeout(flush, COMMIT_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [draft, flush]);

  // Closing the popover must not swallow an edit still inside the debounce
  // window; the pending write carries its own node id, so it lands on the right
  // card whatever the selection has moved on to.
  //
  // On a timer rather than inline, and that is load-bearing. This popover closes
  // when the selection moves, and `selectedNodeId` is in the dependency array of
  // the effect in ModelBuilder that registers the `builder-command` listener — so
  // that same commit removes the listener and adds it back. React runs the
  // ancestor's cleanup before this unmounting descendant's, so a command
  // dispatched here inline is dispatched into the gap and reaches nobody: the
  // edit was committed, the card never changed, and nothing reported an error.
  // A zero delay puts the dispatch after the commit, where the listener is
  // attached again.
  useEffect(() => () => { window.setTimeout(flush, 0); }, [flush]);

  return (
    <div
      className="model-node-params"
      style={{ left: screen.x, top: screen.y, marginLeft: flip ? -272 : 12 }}
      data-testid="node-parameters"
      // Bare letters are canvas shortcuts, and Escape/Delete reach tldraw itself.
      onKeyDown={(event) => {
        if (event.key === 'Escape') closeNodeParameters();
        event.stopPropagation();
      }}
      // This panel is 260px wide and sits 12px off the card it edits, while the
      // layout puts cards on a 220px pitch at 200px wide — a 20px gap. So it lies
      // squarely on top of the NEXT card. It renders in InFrontOfTheCanvas, whose
      // wrapper calls editor.markEventAsHandled on pointer down (the property
      // TldrawCanvas already documents for the insert picker), so a press aimed at
      // that card reached nobody: with the editor open, the neighbouring card could
      // not be selected at all, by a test or by a person.
      //
      // Hand the gesture to the card the pointer is genuinely over. Presses on this
      // panel's own controls are untouched — only a press that lands on canvas
      // furniture showing through is forwarded.
      onPointerDownCapture={(event) => {
        const target = event.target as HTMLElement | null;
        // A field, a button, the close X: an UNMODIFIED press belongs to the editor.
        //
        // A shift-press does not: it is the canvas multi-select gesture, and no
        // control in this panel uses one. It has to be forwarded even from on top of
        // a field, because that is exactly where the covered card's centre lands —
        // ParameterFields renders every row as a wall-to-wall <label><input/></label>
        // and the head is only ~35px, so the neighbour's midpoint is always over a
        // label. Guarding those too is why the first attempt at this changed nothing.
        if (!event.shiftKey && target?.closest('input, textarea, select, button, label, a, [role="button"]')) return;
        const page = editor.screenToPage({ x: event.clientX, y: event.clientY });
        const hit = editor.getShapeAtPoint(page, { hitInside: true });
        if (!hit || !isModelShape(hit) || hit.id === shapeId) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.shiftKey) {
          // Multi-select: the editor describes one card, so it has nothing to show.
          editor.select(...new Set([...editor.getSelectedShapeIds(), hit.id]));
          closeNodeParameters();
        } else {
          editor.select(hit.id);
          openNodeParameters(hit.id);
        }
      }}
    >
      <div className="model-node-params-head">
        <span className="model-node-params-title">
          {data.layerType?.displayName || nodeId}
        </span>
        <button
          type="button"
          className="model-node-params-close"
          aria-label="Close parameters"
          onClick={closeNodeParameters}
        >
          <X size={12} />
        </button>
      </div>
      <div className="model-node-params-body">
        <ParameterFields
          layerTypeId={data.layerType?.id || ''}
          inputShape={data.inputShape}
          values={draft}
          onChange={setDraft}
        />
      </div>
      <div className="model-node-params-foot">
        <button
          type="button"
          className="ui2-chip"
          onClick={() => setDraft(inspectorDraftParameters(currentRef.current))}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export const NodeParametersPopover: React.FC = () => {
  const editor = useEditor();
  const local = React.useContext(LocalBrowserContext);
  const targetId = useValue(parameterTarget);
  // Outside the camera transform, so this has to re-project when the camera moves
  // or the popover drifts off its node.
  const camera = useValue('node params camera', () => editor.getCamera(), [editor]);
  // Reactive: the popover must close on its own when the node it edits is deleted.
  const shape = useValue(
    'node params shape',
    () => (targetId ? editor.getShape(targetId as TLShapeId) : undefined),
    [editor, targetId],
  );

  useEffect(() => {
    if (!targetId) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeNodeParameters();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [targetId]);

  const placement = useMemo(() => {
    if (!targetId || !shape) return null;
    const bounds = editor.getShapePageBounds(shape.id);
    if (!bounds) return null;
    const viewport = editor.pageToViewport({ x: bounds.maxX, y: bounds.y });
    const width = editor.getViewportScreenBounds().w;
    return { x: viewport.x, y: viewport.y, flip: viewport.x + 272 > width };
    // `camera` is read so this recomputes on pan/zoom; the value itself is unused.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor, shape, targetId, camera]);

  if (!targetId || !shape || !isModelShape(shape) || !placement) return null;
  const data = (shape.props as { nodeData?: ModelNodeData }).nodeData;
  if (!data) return null;

  if (local) return <div className="model-node-params" style={{ left: placement.x, top: placement.y, marginLeft: placement.flip ? -272 : 12 }}>
    <button aria-label="Close parameters" onClick={closeNodeParameters}>Close</button>
    <LocalSourceEditor key={shape.id} nodeId={shapeIdToNodeId(shape.id)} data={data} />
  </div>;

  return (
    <Editor
      // Keyed on the node, so selecting another card re-seeds the draft from that
      // card's values instead of carrying this one's half-typed text across.
      key={shape.id}
      shapeId={shape.id}
      data={data}
      screen={{ x: placement.x, y: placement.y }}
      flip={placement.flip}
    />
  );
};

export default NodeParametersPopover;
