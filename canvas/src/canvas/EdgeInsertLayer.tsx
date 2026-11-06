/**
 * The insert-on-the-wire affordance: a `+` at the hovered edge's midpoint, and an
 * inline picker that opens at that point.
 *
 * These live in tldraw's `InFrontOfTheCanvas` slot, NOT `OnTheCanvas`. The names
 * are not interchangeable: `OnTheCanvasWrapper` renders *inside* `.tl-html-layer`
 * and *before* `ShapesLayer`, so anything there is painted UNDERNEATH every card
 * — which is right for the connection ghost and wrong for an interactive popover.
 * The picker used to sit there and disappeared behind whatever card it overlapped.
 *
 * `InFrontOfTheCanvas` is outside the camera transform, so this is untransformed
 * space: page points are projected with editor.pageToViewport, sizes are plain px
 * (no --tl-scale arithmetic), and the layer subscribes to the camera so the
 * affordance stays glued to its wire while the canvas moves. Its wrapper also
 * calls editor.markEventAsHandled on pointer down/up, so clicks inside the picker
 * cannot reach the canvas and deselect.
 *
 * pageToViewport, NOT pageToScreen. The two differ by the container's offset in
 * the viewport, and `.tl-canvas__in-front` is inset:0 of that container — so a
 * pageToScreen value double-counts it. With the builder's 40px top bar the `+`
 * drew 40px BELOW its wire, and because moving the pointer down to it left the
 * edge's hit radius, the hover cleared and the button unmounted before the press
 * landed: unclickable, at any zoom. Every other overlay here (ValidationSweep,
 * CandidateGhost) already uses pageToViewport.
 *
 * Implementation note that cost a debugging round while prototyping: do NOT
 * re-render on hover. Re-rendering detaches the element under the cursor, so the
 * `+` flickers and cannot be clicked. Hover lives in an atom.
 */
import React, { useEffect, useMemo, useState } from 'react';
import { useEditor, useValue } from 'tldraw';
import { X } from 'lucide-react';
import { LayerType } from '../types';
import { layerTypes } from '../data/layerTypes';
import { categoryColor } from '../data/categoryColor';
import { cancelEdgeInsert, hoveredEdge, insertTarget, useEdgeInsert } from './edgeInsert';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { LocalSourceEditor } from './LocalSourceEditor';
import { graphEdges, graphNodes } from './graphQueryState';
import { hasDeclarationEdit } from '../ide/directSourceEdits';
import { graphIsReadOnly } from './graphAccess';

/** Ops it makes sense to drop into an existing connection: single-input, not IO. */
const INSERTABLE = layerTypes.filter(
  (type) => !['input', 'output', 'module'].includes(type.id) && type.category !== 'loss',
);

/**
 * What an empty query offers.
 *
 * Catalogue order starts conv1d, conv2d, conv3d, convtranspose1d,
 * convtranspose2d — so "click + and see what you can add" used to answer with
 * five convolutions and nothing else. Faithful to the data, useless as a starting
 * point, and none of them is a thing you commonly splice into a finished
 * connection. These are.
 */
const SUGGESTED_IDS = [
  'relu', 'gelu', 'batchnorm2d', 'layernorm', 'dropout',
  'narrow', 'interpolate', 'reshape', 'flatten',
];

const SUGGESTED: LayerType[] = SUGGESTED_IDS
  .map((id) => INSERTABLE.find((type) => type.id === id))
  .filter((type): type is LayerType => Boolean(type));

const MAX_ROWS = 6;

/** Prefix matches before substring ones, so typing "re" offers ReLU before Dropout. */
function rank(query: string): LayerType[] {
  const q = query.trim().toLowerCase();
  if (!q) return SUGGESTED.slice(0, MAX_ROWS);
  const scored: Array<{ type: LayerType; score: number }> = [];
  for (const type of INSERTABLE) {
    const id = type.id.toLowerCase();
    const label = type.displayName.toLowerCase();
    let score = -1;
    if (id === q) score = 0;
    else if (id.startsWith(q)) score = 1;
    else if (label.startsWith(q)) score = 2;
    else if (id.includes(q) || label.includes(q)) score = 3;
    if (score >= 0) scored.push({ type, score });
  }
  // Tie-breaks matter as much as the score. "re" prefix-matches both `relu` and
  // `reflectionpad1d`; alphabetical order alone put three ReflectionPads above
  // ReLU. So prefer a common insert, then the shorter id (a tighter match), then
  // alphabetical.
  const common = new Set(SUGGESTED_IDS);
  scored.sort((a, b) =>
    a.score - b.score
    || Number(common.has(b.type.id)) - Number(common.has(a.type.id))
    || a.type.id.length - b.type.id.length
    || a.type.displayName.localeCompare(b.type.displayName));
  return scored.slice(0, MAX_ROWS).map((entry) => entry.type);
}

interface PickerProps {
  screen: { x: number; y: number };
  edgeId: string;
}

const Picker: React.FC<PickerProps> = ({ screen, edgeId }) => {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const channel = useEdgeInsert();
  const rows = useMemo(() => rank(query), [query]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  const commit = (layerTypeId: string) => {
    channel.current.insert(edgeId, layerTypeId);
    cancelEdgeInsert();
  };

  return (
    <div
      className="model-edge-insert-picker"
      style={{ left: screen.x, top: screen.y }}
      data-testid="edge-insert-picker"
    >
      <div className="model-edge-insert-head">
        <input
          className="model-edge-insert-input"
          autoFocus
          value={query}
          placeholder="Insert a layer…"
          aria-label="Insert a layer into this connection"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'ArrowDown') {
              event.preventDefault();
              setActive((i) => Math.min(i + 1, rows.length - 1));
            } else if (event.key === 'ArrowUp') {
              event.preventDefault();
              setActive((i) => Math.max(i - 1, 0));
            } else if (event.key === 'Enter') {
              event.preventDefault();
              if (rows[active]) commit(rows[active].id);
            } else if (event.key === 'Escape') {
              event.preventDefault();
              cancelEdgeInsert();
            }
            // Bare letters are canvas shortcuts otherwise.
            event.stopPropagation();
          }}
        />
        {/* Escape already closes, but the picker opens from a pointer gesture and
            gave a pointer user no way back out. onMouseDown is prevented so the
            press does not pull focus off the input before the click lands. */}
        <button
          type="button"
          className="model-edge-insert-close"
          aria-label="Close without inserting"
          title="Close (Esc)"
          data-testid="edge-insert-close"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => cancelEdgeInsert()}
        >
          <X size={12} aria-hidden="true" />
        </button>
      </div>
      {!query && <div className="model-edge-insert-hint">Common inserts — type to search all layers</div>}
      {rows.length === 0 ? (
        <div className="model-edge-insert-empty">No layer matches “{query}”</div>
      ) : (
        rows.map((type, index) => (
          <button
            key={type.id}
            type="button"
            className={`model-edge-insert-row${index === active ? ' on' : ''}`}
            /* The row wears the colour its CARD will wear once inserted — the
               same category palette the canvas paints nodes with, via the shared
               categoryColor helper. The selected row used to be cyan whatever it
               was, so the picker said nothing about what kind of block you were
               about to add. */
            style={{ ['--row-accent' as string]: categoryColor(type.category) } as React.CSSProperties}
            onMouseEnter={() => setActive(index)}
            onClick={() => commit(type.id)}
          >
            <span className="model-edge-insert-swatch" aria-hidden="true" />
            <span className="model-edge-insert-name">{type.displayName}</span>
            <span className="model-edge-insert-op">{type.id}</span>
          </button>
        ))
      )}
      <div className="model-edge-insert-foot">↵ insert into this connection</div>
    </div>
  );
};

export const EdgeInsertLayer: React.FC = () => {
  const editor = useEditor();
  const local = React.useContext(LocalBrowserContext);
  const nodes = useValue(graphNodes);
  const edges = useValue(graphEdges);
  const hover = useValue(hoveredEdge);
  const target = useValue(insertTarget);
  // This layer is outside the camera transform, so it has to re-project when the
  // camera moves or the affordance would drift off its wire.
  const camera = useValue('edge insert camera', () => editor.getCamera(), [editor]);

  useEffect(() => {
    if (!target) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') cancelEdgeInsert();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [target]);

  const project = (point: { x: number; y: number }) => {
    const viewport = editor.pageToViewport(point);
    return { x: viewport.x, y: viewport.y };
  };
  // `camera` is read so this recomputes on pan/zoom; the value itself is unused.
  void camera;

  if (local?.sourceEdits) {
    const active = target || hover;
    const edge = edges.find(item => item.id === active?.edgeId);
    const source = nodes.find(item => item.id === edge?.source);
    // Declaration insertion applies to the layer's calls, not an arbitrary branch.
    const singleOutput = source && edges.filter(item => item.source === source.id).length === 1;
    const destination = nodes.find(item => item.id === edge?.target);
    const port = destination?.data.capturePorts?.find(port => port.id === edge?.targetHandle);
    const caps = destination?.data.sourceEditCapabilities?.filter(cap => cap.command === 'insert_unary_op' &&
      port?.valueId === cap.expected.id && port.paths.some(path => JSON.stringify(path) === JSON.stringify(cap.argumentPath)));
    const insertion = destination && caps?.length === 1
      ? { nodeId: destination.id, data: destination.data, capabilityId: caps[0].id } : undefined;
    if (!active || !source || (!insertion && (!singleOutput || !hasDeclarationEdit(source.data)))) return null;
    const point = project(active.point);
    if (target) return <div className="model-edge-insert-picker" style={{ left: point.x, top: point.y }}>
      <button aria-label="Close insertion" onClick={cancelEdgeInsert}>Close</button>
      <p>{insertion ? "Activation insertion uses this connection’s exact source operand." : "Insert after this declaration (all its calls)."}</p>
      <LocalSourceEditor key={source.id} nodeId={source.id} data={source.data} structuralOnly insertion={insertion} />
    </div>;
    return <button className="model-edge-insert-plus" aria-label="Insert layer through source review"
      style={{ left: point.x, top: point.y }} onClick={() => insertTarget.set(active)}>+</button>;
  }
  if (graphIsReadOnly(editor)) return null;
  if (target) return <Picker screen={project(target.point)} edgeId={target.edgeId} />;
  if (!hover) return null;

  const screen = project(hover.point);
  return (
    <button
      type="button"
      className="model-edge-insert-plus"
      data-testid="edge-insert-plus"
      title="Insert a layer into this connection"
      aria-label="Insert a layer into this connection"
      style={{ left: screen.x, top: screen.y }}
      onPointerDown={() => insertTarget.set(hover)}
    >
      +
    </button>
  );
};
