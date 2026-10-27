import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { categories, layerTypes } from '../../data/layerTypes';
import { LayerType } from '../../types';
import {
  ApiService,
  PaperTemplateInventoryResponse,
  PaperTemplateInventoryTemplate,
} from '../../services/apiService';
import {
  WorkspaceClass,
  fetchWorkspaceClasses,
  importPathOf,
  selectWorkspaceClasses,
  workspaceRoot,
} from '../../ide/workspaceClasses';

type AddLayerOptions = {
  parameterOverrides?: Record<string, any>;
  dataOverrides?: Record<string, any>;
};

interface BuilderPaletteV2Props {
  onAddLayer: (layerType: string, options?: AddLayerOptions) => void;
}

const MAX_VISIBLE_TEMPLATE_ITEMS = 28;

export function selectPaletteTemplateItems(
  templateInventory: PaperTemplateInventoryResponse | null,
  query: string
) {
  const q = query.trim().toLowerCase();
  const items = templateInventory?.templates || [];
  const filtered = q
    ? items.filter((row) => {
        const id = row.template_id.toLowerCase();
        const name = row.name.toLowerCase();
        return id.includes(q) || name.includes(q);
      })
    : items;

  return [...filtered]
    .sort((a, b) => {
      if (a.source !== b.source) return a.source === 'dynamic' ? -1 : 1;
      return a.name.localeCompare(b.name);
    })
    .slice(0, MAX_VISIBLE_TEMPLATE_ITEMS);
}

/** Ids that exist so a projected node resolves to an OpSpec, never so a person can
 *  insert one. Kept beside the palette rather than exported from layerTypes.ts: the
 *  registry is right to list them, and this is the one place that must not offer them. */
const PLACEHOLDER_LAYER_IDS = new Set(['python', 'custom_code']);

/**
 * The right-hand hint names the torch symbol the row actually inserts, so it may
 * not be fabricated. Entries in layerTypes are already qualified where they are
 * not nn.* members (`torch.matmul`), and an Input/Output is a graph endpoint with
 * no torch class at all — those get no hint rather than a plausible `nn.Input`.
 */
function torchSymbol(layer: LayerType) {
  if (layer.category === 'io') return '';
  if (layer.name.includes('.')) return layer.name;
  const compact = layer.name.replace(/[^a-zA-Z0-9]/g, '');
  return compact ? `nn.${compact}` : '';
}

/**
 * Layers and blocks differ only in their accent, their hint, and what inserting
 * or dragging them does. Reducing both to one row shape keeps a single renderer
 * — and one flat index space, which is what ↑↓ moves through.
 */
type PaletteRow = {
  key: string;
  name: string;
  detail: string;
  detailMono: boolean;
  hint: string;
  accent: string;
  title: string;
  insert: () => void;
  /**
   * Absent for a row that cannot be dragged. A workspace class is one: the drop
   * protocol carries a layer id and a template id, and neither can say which of
   * the user's classes this is — so it would land as a `custom_code` node with no
   * class behind it. Clicking and ↵ insert it with its provenance intact.
   */
  startDrag?: (event: React.DragEvent) => void;
};

const PANEL_STYLE: React.CSSProperties = {
  width: 620,
  // Viewport-relative, not parent-relative: the drawer that positions this panel
  // is narrower than the panel is, so `maxWidth: 100%` would collapse it.
  maxWidth: 'calc(100vw - 64px)',
  maxHeight: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  borderRadius: 12,
  border: '1px solid var(--ui2-border-strong)',
  background: 'var(--ui2-surface-2)',
  boxShadow: 'var(--ui2-shadow-float)',
};

const HEAD_STYLE: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  padding: '16px 18px',
  borderBottom: '1px solid var(--ui2-rule)',
  cursor: 'text',
};

const QUERY_INPUT_STYLE: React.CSSProperties = {
  flex: '0 1 auto',
  minWidth: 0,
  padding: 0,
  border: 0,
  outline: 'none',
  background: 'transparent',
  fontFamily: 'var(--font-mono)',
  fontSize: 14,
  color: 'var(--ui2-text)',
  caretColor: 'var(--ui2-cyan)',
};

const PROMPT_STYLE: React.CSSProperties = {
  flex: '1 1 auto',
  minWidth: 0,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  fontSize: 14,
  color: 'var(--ui2-text-dim)',
};

const RESULTS_STYLE: React.CSSProperties = {
  flex: '1 1 auto',
  minHeight: 0,
  maxHeight: 400,
  overflowY: 'auto',
  padding: '8px 8px 10px',
};

const FOOT_STYLE: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 16,
  padding: '10px 16px',
  borderTop: '1px solid var(--ui2-rule)',
  background: 'rgba(255, 255, 255, 0.02)',
  fontFamily: 'var(--font-mono)',
  fontSize: 10.5,
  color: 'var(--ui2-text-dim)',
};

const ACCENT_BAR_STYLE: React.CSSProperties = {
  flex: '0 0 auto',
  width: 2,
  height: 22,
  borderRadius: 2,
};

const NAME_STYLE: React.CSSProperties = {
  flex: '0 0 150px',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  fontFamily: 'var(--font-mono)',
  fontSize: 13,
  color: 'var(--ui2-text)',
};

const HINT_STYLE: React.CSSProperties = {
  flex: '0 0 auto',
  fontFamily: 'var(--font-mono)',
  fontSize: 10.5,
  color: 'var(--ui2-text-dim)',
};

const GROUP_HEAD_STYLE: React.CSSProperties = { padding: '10px 10px 8px' };

const SECOND_GROUP_HEAD_STYLE: React.CSSProperties = {
  padding: '16px 10px 8px',
  marginTop: 8,
  borderTop: '1px solid var(--ui2-rule)',
};

export default function BuilderPaletteV2({ onAddLayer }: BuilderPaletteV2Props) {
  const [templateInventory, setTemplateInventory] = useState<PaperTemplateInventoryResponse | null>(null);
  const [templateError, setTemplateError] = useState('');
  // null means "there is no workspace here" — the hosted product, or an IDE whose
  // shell or backend did not answer — and renders NO Workspace section at all.
  // An empty array means a folder that really has no nn.Module classes in it, and
  // says so. The two are different answers and must not collapse into one.
  const [workspaceClasses, setWorkspaceClasses] = useState<WorkspaceClass[] | null>(null);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const mountedRef = useRef(true);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const activeRowRef = useRef<HTMLButtonElement | null>(null);

  const loadTemplates = useCallback(async () => {
    try {
      const inventory = await ApiService.getPaperTemplateInventory();
      if (!mountedRef.current) return;
      setTemplateInventory(inventory);
      setTemplateError('');
    } catch (error) {
      if (!mountedRef.current) return;
      setTemplateError(error instanceof Error ? error.message : 'Failed to load template blocks');
    }
  }, []);

  // The user's own classes (task.md M6). Every failure is the same outcome — no
  // section — so this swallows rather than reports: the palette is a search box
  // the user opened to insert a layer, and an error banner about a scan they did
  // not ask for would be noise in the hosted product, where it can never succeed.
  const loadWorkspaceClasses = useCallback(async () => {
    try {
      const root = await workspaceRoot();
      if (!root) return;
      const classes = await fetchWorkspaceClasses(root);
      if (!mountedRef.current) return;
      setWorkspaceClasses(classes);
    } catch {
      if (!mountedRef.current) return;
      setWorkspaceClasses(null);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void loadTemplates();
    void loadWorkspaceClasses();
    return () => {
      mountedRef.current = false;
    };
  }, [loadTemplates, loadWorkspaceClasses]);

  // The palette is mounted only while it is open, and the builder's window-level
  // keydown handler bails whenever focus sits in an input — so taking focus here
  // is what routes typing and ↑↓/↵ into this list instead of the canvas.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const templateItems = useMemo(() => {
    return selectPaletteTemplateItems(templateInventory, query);
  }, [query, templateInventory]);

  const layerRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    // `python` and `custom_code` are in layerTypes.ts so that every id on the
    // canvas resolves to an OpSpec, not so that anyone can insert one. Picking
    // "Custom Code" out of this list would make a node with no class behind it:
    // the validator refuses it for having no shape, and codegen cannot emit it.
    // The Workspace section below is where a real class comes from (task.md M6).
    const insertable = layerTypes.filter((layer) => !PLACEHOLDER_LAYER_IDS.has(layer.id));
    if (!q) return insertable;
    return insertable.filter((layer) => {
      const title = layer.displayName.toLowerCase();
      const id = layer.id.toLowerCase();
      return title.includes(q) || id.includes(q) || layer.description.toLowerCase().includes(q);
    });
  }, [query]);

  const onLayerDragStart = (event: React.DragEvent, layerType: string) => {
    event.dataTransfer.setData('application/canvas-layer', layerType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const onTemplateDragStart = (event: React.DragEvent, templateId: string, templateName: string) => {
    event.dataTransfer.setData('application/canvas-layer', 'module');
    event.dataTransfer.setData('application/template-id', templateId);
    event.dataTransfer.setData('application/template-name', templateName);
    event.dataTransfer.effectAllowed = 'move';
  };

  // Category order and the 14-per-category cap are preserved from the grouped
  // drawer: the design collapses the heads into one "Layers" group, but the set
  // of layers offered — and the order they appear in — must not change.
  const layerResults = useMemo<PaletteRow[]>(() => {
    return categories.flatMap((category) =>
      layerRows
        .filter((layer) => layer.category === category.id)
        .slice(0, 14)
        .map((layer) => ({
          key: `layer-${layer.id}`,
          name: layer.displayName,
          detail: layer.description,
          detailMono: false,
          hint: torchSymbol(layer),
          accent: 'var(--ui2-cyan)',
          title: layer.description,
          insert: () => onAddLayer(layer.id),
          startDrag: (event: React.DragEvent) => onLayerDragStart(event, layer.id),
        }))
    );
  }, [layerRows, onAddLayer]);

  const blockResults = useMemo<PaletteRow[]>(() => {
    return templateItems.map((template: PaperTemplateInventoryTemplate) => {
      const name = template.name || template.template_id;
      const layers = template.required_nodes;
      return {
        key: `block-${template.template_id}`,
        name,
        // The id, not `input_expectation`: it is unique, it is what the search
        // matches on, and the expectation string repeats across most templates.
        detail: template.template_id,
        detailMono: true,
        hint: layers > 0 ? `${layers} layer${layers === 1 ? '' : 's'}` : '',
        accent: 'var(--ui2-green)',
        title: `${template.name} (${template.template_id})`,
        insert: () =>
          onAddLayer('module', {
            parameterOverrides: { name },
            dataOverrides: { moduleTemplateId: template.template_id },
          }),
        startDrag: (event: React.DragEvent) =>
          onTemplateDragStart(event, template.template_id, name),
      };
    });
  }, [templateItems, onAddLayer]);

  /**
   * The Workspace rows. `source: 'workspace'` in spirit and in fact: these come
   * from the user's folder, not from layerTypes.ts and not from the canonical
   * registry, and inserting one produces a `custom_code` node whose provenance
   * keys (task.md §6.1) say which class it is and where to import it from.
   */
  const workspaceResults = useMemo<PaletteRow[]>(() => {
    if (workspaceClasses === null) return [];
    return selectWorkspaceClasses(workspaceClasses, query).map((entry) => ({
      key: `workspace-${entry.file}#${entry.name}`,
      name: entry.name,
      detail: `${entry.file}:${entry.line}`,
      detailMono: true,
      hint: entry.ctor_required > 0 ? `${entry.ctor_required} required arg${entry.ctor_required === 1 ? '' : 's'}` : '',
      accent: 'var(--cat-module)',
      title: `${entry.name} — ${entry.bases.join(', ') || 'nn.Module'}`,
      insert: () =>
        onAddLayer('custom_code', {
          parameterOverrides: {
            _class: entry.name,
            _import_path: importPathOf(entry.file),
            _source_file: entry.file,
            _source_line: entry.line,
          },
        }),
    }));
  }, [workspaceClasses, query, onAddLayer]);

  const rows = useMemo(
    () => [...layerResults, ...blockResults, ...workspaceResults],
    [layerResults, blockResults, workspaceResults],
  );
  // Clamp rather than reset in an effect, so a shrinking result set can never
  // point ↵ at a row that is no longer on screen.
  const activeAt = rows.length === 0 ? -1 : Math.min(activeIndex, rows.length - 1);

  useEffect(() => {
    activeRowRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [activeAt]);

  const onSearchKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (rows.length === 0) return;
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActiveIndex(Math.min(activeAt + 1, rows.length - 1));
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveIndex(Math.max(activeAt - 1, 0));
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      rows[activeAt]?.insert();
    }
  };

  const renderRow = (row: PaletteRow, index: number) => {
    const active = index === activeAt;
    return (
      <button
        key={row.key}
        // .ui2-palette-item is kept for its width/alignment reset; the row's own
        // look is inline, including transform:none to cancel the class's hover
        // lift. (It also used to be how agent-demo-live.spec.ts found these rows;
        // that spec is deleted, so no test depends on the class any more.)
        className="ui2-palette-item"
        type="button"
        draggable={Boolean(row.startDrag)}
        onDragStart={(event) => row.startDrag?.(event)}
        onClick={() => row.insert()}
        onMouseEnter={() => setActiveIndex(index)}
        ref={active ? activeRowRef : undefined}
        title={row.title}
        style={{
          gap: 12,
          padding: '9px 10px',
          border: 0,
          borderRadius: 8,
          transform: 'none',
          background: active ? 'rgba(255, 255, 255, 0.07)' : 'transparent',
          boxShadow: active ? 'inset 2px 0 0 var(--ui2-text)' : 'none',
        }}
      >
        <span style={{ ...ACCENT_BAR_STYLE, background: row.accent }} />
        <span style={NAME_STYLE}>{row.name}</span>
        <span
          style={{
            flex: '1 1 auto',
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            fontSize: 12.5,
            color: 'var(--ui2-text-dim)',
            fontFamily: row.detailMono ? 'var(--font-mono)' : undefined,
          }}
        >
          {row.detail}
        </span>
        {row.hint ? <span style={HINT_STYLE}>{row.hint}</span> : null}
      </button>
    );
  };

  const trimmedQuery = query.trim();

  return (
    <div style={PANEL_STYLE}>
      <label style={HEAD_STYLE}>
        <Search size={17} style={{ flex: '0 0 auto', color: 'var(--ui2-cyan)' }} />
        <input
          ref={inputRef}
          aria-label="Search layers, blocks and templates"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setActiveIndex(0);
          }}
          onKeyDown={onSearchKeyDown}
          // Sized to the query so the native caret lands where the mockup's does,
          // immediately before the prompt, instead of after an empty text field.
          style={{ ...QUERY_INPUT_STYLE, width: `${Math.max(query.length, 1)}ch` }}
        />
        <span style={PROMPT_STYLE}>Add a layer, block or template</span>
        <span className="ui2-kbd">⇧A</span>
      </label>

      <div className="ui2-scroll-surface" style={RESULTS_STYLE}>
        <section>
          <div className="ui2-eyebrow" style={GROUP_HEAD_STYLE}>Layers</div>
          {layerResults.length === 0 ? (
            <div className="ui2-palette-note" style={{ margin: '0 10px 8px' }}>
              No layers match “{trimmedQuery}”.
            </div>
          ) : (
            layerResults.map((row, index) => renderRow(row, index))
          )}
        </section>

        <section data-testid="template-blocks-section">
          <div className="ui2-eyebrow" style={SECOND_GROUP_HEAD_STYLE}>Blocks</div>
          {templateError ? (
            <div
              className="ui2-palette-note-error"
              style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '0 10px 8px' }}
            >
              <span style={{ flex: '1 1 auto', minWidth: 0 }}>{templateError}</span>
              <button
                className="ui2-chip"
                type="button"
                onClick={() => {
                  setTemplateError('');
                  void loadTemplates();
                }}
              >
                Retry
              </button>
            </div>
          ) : null}
          {!templateError && blockResults.length === 0 ? (
            <div className="ui2-palette-note" style={{ margin: '0 10px 8px' }}>
              {trimmedQuery ? `No blocks match “${trimmedQuery}”.` : 'No blocks found.'}
            </div>
          ) : null}
          {blockResults.map((row, index) => renderRow(row, layerResults.length + index))}
        </section>

        {workspaceClasses !== null && (
          <section data-testid="workspace-classes-section">
            <div className="ui2-eyebrow" style={SECOND_GROUP_HEAD_STYLE}>Workspace</div>
            {workspaceResults.length === 0 ? (
              <div className="ui2-palette-note" style={{ margin: '0 10px 8px' }}>
                {trimmedQuery
                  ? `No workspace classes match “${trimmedQuery}”.`
                  : 'No nn.Module classes in this folder yet.'}
              </div>
            ) : null}
            {workspaceResults.map((row, index) =>
              renderRow(row, layerResults.length + blockResults.length + index),
            )}
          </section>
        )}
      </div>

      <div style={FOOT_STYLE}>
        <span>↑↓ move</span>
        <span>↵ insert after selection</span>
        <span style={{ marginLeft: 'auto' }}>drag to place</span>
      </div>
    </div>
  );
}
