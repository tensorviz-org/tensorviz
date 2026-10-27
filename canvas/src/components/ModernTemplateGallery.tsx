/**
 * ModernTemplateGallery — "Start from an architecture".
 *
 * One dialog with one job: pick a catalog architecture and instantiate it on this
 * machine. The card face is a GraphThumb of the topology the backend actually
 * declares for that entry, so a two-layer starter reads as smaller than a staged
 * CNN before either is opened. An entry whose declared structure we do not have
 * gets no thumbnail at all rather than a generic glyph that would be decoration.
 *
 * Selecting and committing are now separate gestures: a card click selects, the
 * foot commits (double-click commits directly). Previously a card click meant
 * "instantiate now" on the Templates sub-tab and "open a customisation sheet" on
 * the Popular one — the same gesture with two outcomes depending on invisible
 * state.
 *
 * The commit control is labelled "Use Template" because three e2e specs locate it
 * by that phrase; the selected architecture's name sits BESIDE it, not inside it,
 * so the accessible name stays constant as the selection changes.
 *
 * Styling is the shared dialog frame from styles/ui2.tokens.css (`.ui2-dialog*`,
 * `.ui2-tab-underline`, `.ui2-card*`, `.ui2-metric-strip`); only the wrappers
 * those classes do not cover are inline.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Loader2, Search, X } from 'lucide-react';
import { ApiService } from '../services/apiService';
import { ModelGraph } from '../types';
import GraphThumb from './ui2/GraphThumb';
import { formatParams, graphMetrics } from '../services/graphMetrics';

interface TemplateCategory {
  id: string;
  name: string;
  description: string;
  icon: string;
  count: number;
}

interface ModelTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  sub_category: string;
  difficulty: string;
  tags: string[];
  base_config: Record<string, any>;
  customizable_params: Array<{
    name: string;
    type: string;
    description: string;
    [key: string]: any;
  }>;
  /** The template's declared node/edge skeleton. Present on every catalog entry. */
  template_structure?: { nodes?: unknown; edges?: unknown };
}

interface PopularModel {
  id: string;
  name: string;
  full_name: string;
  description: string;
  category: string;
  architecture_type: string;
  parameter_count?: string;
  training_dataset?: string;
  template_id: string;
  config: Record<string, any>;
  typical_applications: string[];
}

interface ModernTemplateGalleryProps {
  onTemplateSelected: (templateData: any) => void;
  onClose: () => void;
}

/** One selectable row of the catalog, flattened across the two kinds the API serves. */
type Entry = {
  /** Unique across kinds — a popular model and its template can share an id. */
  key: string;
  kind: 'popular' | 'template';
  id: string;
  name: string;
  description: string;
  graph: Pick<ModelGraph, 'nodes' | 'edges'> | null;
  /** Layer count + estimated params, only when the declared structure supports them. */
  metrics: { layers: number; params: number } | null;
  /** The backend's own parameter figure, when it reports one. */
  paramLabel?: string;
  haystack: string;
  model?: PopularModel;
  template?: ModelTemplate;
};

const MONO = { fontFamily: 'var(--font-mono)' } as const;

const S = {
  hint: { fontSize: 12.5, lineHeight: '19px', color: 'var(--ui2-text-dim)' },
  search: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '7px 11px',
    background: 'rgba(255,255,255,0.03)', border: '1px solid var(--ui2-border)',
    borderRadius: 8, color: 'var(--ui2-text-dim)',
  },
  searchInput: {
    flex: 1, minWidth: 0, background: 'transparent', border: 'none', outline: 'none',
    color: 'var(--ui2-text)', fontFamily: 'var(--font-ui)', fontSize: 12.5,
  },
  field: {
    padding: '6px 9px', fontSize: 12.5, fontFamily: 'var(--font-mono)',
    background: 'rgba(255,255,255,0.03)', color: 'var(--ui2-text)',
    border: '1px solid var(--ui2-border)', borderRadius: 8, outline: 'none',
  },
  blurb: {
    fontSize: 13, lineHeight: '18px', color: 'var(--ui2-text-dim)',
    display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' as const,
    overflow: 'hidden',
  },
  name: { fontFamily: 'var(--font-ui)', fontSize: 14.5 },
};

const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Unresolved template params arrive as literal `{placeholder}` strings. */
const PLACEHOLDER = /^\{[^}]*\}$/;

function previewGraph(structure?: { nodes?: unknown; edges?: unknown } | null) {
  const nodes = Array.isArray(structure?.nodes) ? structure!.nodes : null;
  if (!nodes || nodes.length === 0) return null;
  const edges = Array.isArray(structure?.edges) ? structure!.edges : [];
  return { nodes, edges } as unknown as Pick<ModelGraph, 'nodes' | 'edges'>;
}

/**
 * Whether a declared structure can carry a layer count and a parameter estimate.
 *
 * Most catalog templates are skeletons: one module node stands in for a whole
 * stage and its parameters are `{placeholder}` strings the instantiate call fills
 * in. Counting those nodes would print "3 layers" for VGG-16 — a number that
 * looks like a fact and is off by fifteen. The starter templates ARE fully
 * resolved chains of real torch modules, so those get the strip and the rest do
 * not.
 */
function hasConcreteStructure(graph: Pick<ModelGraph, 'nodes'>): boolean {
  return graph.nodes.every((node) => {
    if (node.type !== 'layerNode') return false;
    if (node.data?.isGroup || node.data?.type === 'module' || node.data?.type === 'group') return false;
    return Object.values(node.data?.parameters ?? {}).every(
      (value) => typeof value !== 'string' || !PLACEHOLDER.test(value),
    );
  });
}

const ModernTemplateGallery: React.FC<ModernTemplateGalleryProps> = ({
  onTemplateSelected,
  onClose,
}) => {
  const [categories, setCategories] = useState<TemplateCategory[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [templates, setTemplates] = useState<Record<string, ModelTemplate>>({});
  const [popularModels, setPopularModels] = useState<Record<string, PopularModel>>({});
  const [loading, setLoading] = useState(true);
  /** Catalog fetch failed — recoverable, so it renders a Retry. */
  const [loadError, setLoadError] = useState<string | null>(null);
  /** An instantiate call failed — retried by pressing the commit control again. */
  const [actionError, setActionError] = useState<string | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const [query, setQuery] = useState('');
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  /** Non-null while the Configure editor is open; holds the edited config. */
  const [configDraft, setConfigDraft] = useState<Record<string, any> | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const res = await ApiService.request('/api/templates/categories');
        if (!live) return;
        const list: TemplateCategory[] = res.categories ?? [];
        setCategories(list);
        setLoadError(null);
        // Only default the tab when the open one is gone, so a Retry does not
        // yank the user back to the first category.
        setSelectedCategory((curr) =>
          curr && list.some((c) => c.id === curr) ? curr : list[0]?.id ?? '');
        if (list.length === 0) setLoading(false);
      } catch (e) {
        if (live) {
          setLoadError(messageOf(e));
          setLoading(false);
        }
      }
    })();
    return () => { live = false; };
  }, [retryToken]);

  useEffect(() => {
    if (!selectedCategory) return;
    let live = true;
    setLoading(true);
    (async () => {
      try {
        const res = await ApiService.request(`/api/templates/categories/${selectedCategory}`);
        if (!live) return;
        setTemplates(res.templates || {});
        setPopularModels(res.popular_models || {});
        setLoadError(null);
      } catch (e) {
        if (live) {
          setTemplates({});
          setPopularModels({});
          setLoadError(messageOf(e));
        }
      } finally {
        if (live) setLoading(false);
      }
    })();
    return () => { live = false; };
  }, [selectedCategory, retryToken]);

  const entries = useMemo<Entry[]>(() => {
    const templateList = Object.values(templates);
    const build = (
      kind: Entry['kind'],
      id: string,
      name: string,
      description: string,
      structure: { nodes?: unknown; edges?: unknown } | undefined,
      tags: string[],
      extra: Partial<Entry>,
    ): Entry => {
      const graph = previewGraph(structure);
      const derived = graph && hasConcreteStructure(graph) ? graphMetrics(graph) : null;
      return {
        key: `${kind}:${id}`,
        kind,
        id,
        name,
        description,
        graph,
        metrics: derived ? { layers: derived.layers, params: derived.params } : null,
        haystack: [name, description, ...tags].join(' ').toLowerCase(),
        ...extra,
      };
    };

    return [
      // A popular model has no structure of its own — it configures a template.
      // That template's skeleton is the honest preview for it, when the same
      // category shipped it.
      ...Object.values(popularModels).map((model) => build(
        'popular',
        model.id,
        model.name,
        model.description,
        templates[model.template_id]?.template_structure,
        [model.architecture_type, ...model.typical_applications],
        { paramLabel: model.parameter_count, model },
      )),
      ...templateList.map((template) => build(
        'template',
        template.id,
        template.name,
        template.description,
        template.template_structure,
        template.tags,
        { template },
      )),
    ];
  }, [popularModels, templates]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? entries.filter((e) => e.haystack.includes(q)) : entries;
  }, [entries, query]);

  // The foot commits the selection, so the first entry is selected by default and
  // the fallback is derived rather than pushed into state by an effect: an effect
  // leaves the commit control disabled for one frame after the grid appears, and
  // a stale key from the previous category would disable it until a card is
  // clicked.
  const selected = filtered.find((e) => e.key === selectedKey) ?? filtered[0] ?? null;
  const configurable = selected?.kind === 'popular'
    && Object.keys(selected.model?.config ?? {}).length > 0;

  // A draft belongs to one entry's config; carrying it across a selection change
  // would send ResNet's overrides to ViT.
  useEffect(() => { setConfigDraft(null); }, [selected?.key]);

  const commit = useCallback(async (entry: Entry | null, config: Record<string, any> | null) => {
    if (!entry || busy) return;
    setBusy(true);
    setActionError(null);
    try {
      if (entry.kind === 'popular' && entry.model) {
        const model = entry.model;
        const finalConfig = { ...model.config, ...(config ?? {}) };
        const res = await ApiService.request(`/api/popular-models/${model.id}/instantiate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: finalConfig, model_name: model.name }),
        });
        if (!res?.graph) throw new Error('The backend returned no graph for that model.');
        onTemplateSelected({ type: 'popular_model', data: res, model, config: finalConfig });
        return;
      }
      const res = await ApiService.request(`/api/templates/${entry.id}/instantiate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: {}, model_name: `Custom_${entry.id}` }),
      });
      if (!res?.graph) throw new Error('The backend returned no graph for that template.');
      onTemplateSelected({ type: 'template', data: res, templateId: entry.id, config: {} });
    } catch (e) {
      setActionError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }, [busy, onTemplateSelected]);

  const activeCategory = categories.find((c) => c.id === selectedCategory);

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      className="ui2-dialog-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Start from an architecture"
    >
      <div className="ui2-dialog" style={{ width: 980, maxWidth: '100%', maxHeight: '86vh' }}>
        <header className="ui2-dialog-head" style={{ alignItems: 'center' }}>
          <h2 className="ui2-dialog-title">Start from an architecture</h2>
          <div style={{ ...S.search, marginLeft: 'auto', flex: '0 1 300px', minWidth: 150 }}>
            <Search size={14} style={{ flex: '0 0 auto' }} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                entries.length
                  ? `Search ${entries.length} architectures`
                  : 'Search architectures'
              }
              aria-label="Search architectures"
              style={S.searchInput}
            />
          </div>
          <button
            onClick={onClose}
            type="button"
            aria-label="Close"
            className="ui2-icon-button"
            style={{ flex: '0 0 auto' }}
          >
            <X size={16} />
          </button>
        </header>

        {categories.length > 0 && (
          <div
            role="tablist"
            style={{
              display: 'flex', gap: 20, padding: '0 20px', flex: '0 0 auto',
              overflowX: 'auto', borderBottom: '1px solid var(--ui2-rule)',
            }}
          >
            {categories.map((cat) => (
              <button
                onClick={() => { setSelectedCategory(cat.id); setQuery(''); }}
                key={cat.id}
                type="button"
                role="tab"
                aria-selected={cat.id === selectedCategory}
                title={cat.description}
                className="ui2-tab-underline"
                style={{ flex: '0 0 auto' }}
              >
                {cat.name}
              </button>
            ))}
          </div>
        )}

        <div
          className="ui2-dialog-body ui2-scroll-surface"
          style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}
        >
          {loadError && (
            <div
              role="alert"
              className="ui2-panel ui2-notice-error"
              style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '11px 13px' }}
            >
              <AlertTriangle size={15} style={{ color: 'var(--ui2-red)', flex: '0 0 auto', marginTop: 2 }} />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, color: 'var(--ui2-text)' }}>
                  Couldn’t load the architecture catalog
                </div>
                <div
                  style={{
                    ...MONO, fontSize: 11.5, lineHeight: '17px', marginTop: 4,
                    color: 'var(--ui2-text-dim)', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                  }}
                >
                  {loadError}
                </div>
              </div>
              <button
                onClick={() => setRetryToken((t) => t + 1)}
                type="button"
                className="ui2-chip"
                style={{ flex: '0 0 auto' }}
              >
                Retry
              </button>
            </div>
          )}

          {actionError && (
            <div
              role="alert"
              className="ui2-panel ui2-notice-error"
              style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '11px 13px' }}
            >
              <AlertTriangle size={15} style={{ color: 'var(--ui2-red)', flex: '0 0 auto', marginTop: 2 }} />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, color: 'var(--ui2-text)' }}>
                  Couldn’t open that architecture
                </div>
                <div
                  style={{
                    ...MONO, fontSize: 11.5, lineHeight: '17px', marginTop: 4,
                    color: 'var(--ui2-text-dim)', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                  }}
                >
                  {actionError}
                </div>
              </div>
            </div>
          )}

          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, ...S.hint }}>
              <Loader2 size={15} className="spinning" />
              Loading {activeCategory?.name ?? 'catalog'} architectures…
            </div>
          ) : configDraft && selected ? (
            <ConfigEditor
              name={selected.name}
              draft={configDraft}
              onChange={setConfigDraft}
            />
          ) : (
            <>
              {filtered.length === 0 && !loadError && (
                <div style={S.hint}>
                  {query
                    ? `Nothing in ${activeCategory?.name ?? 'this category'} matches “${query}”.`
                    : `${activeCategory?.name ?? 'This category'} has no architectures yet.`}
                </div>
              )}
              {filtered.length > 0 && (
                <div className="ui2-card-grid" style={{ marginTop: 0 }}>
                  {filtered.map((entry) => (
                    <button
                      onClick={() => setSelectedKey(entry.key)}
                      onDoubleClick={() => commit(entry, null)}
                      key={entry.key}
                      type="button"
                      aria-pressed={entry.key === selected?.key}
                      className="ui2-card"
                      style={entry.key === selected?.key
                        ? { borderColor: 'var(--ui2-accent-border)', background: 'rgba(255,255,255,0.05)' }
                        : undefined}
                    >
                      {entry.graph && <GraphThumb graph={entry.graph} height={100} />}
                      <div className="ui2-card-body">
                        <div className="ui2-card-name" style={S.name}>{entry.name}</div>
                        <div style={S.blurb}>{entry.description}</div>
                        {entry.metrics ? (
                          <div className="ui2-metric-strip">
                            <span>
                              {entry.metrics.layers} layer{entry.metrics.layers === 1 ? '' : 's'}
                            </span>
                            {entry.metrics.params > 0 && (
                              <>
                                <span>{formatParams(entry.metrics.params)}</span>
                                <span className="ui2-metric-estimated">est.</span>
                              </>
                            )}
                          </div>
                        ) : entry.paramLabel ? (
                          <div className="ui2-metric-strip">
                            <span>{entry.paramLabel} params</span>
                          </div>
                        ) : null}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        <div className="ui2-dialog-foot">
          <span style={S.hint}>Instantiated locally. No weights are downloaded.</span>
          {selected && (
            <span
              style={{
                ...MONO, marginLeft: 'auto', fontSize: 12, minWidth: 0,
                color: 'var(--ui2-text-dim)', overflow: 'hidden',
                textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}
            >
              {selected.name}
            </span>
          )}
          {configDraft ? (
            <button
              onClick={() => setConfigDraft(null)}
              type="button"
              className="ui2-chip"
              style={{ flex: '0 0 auto', marginLeft: selected ? undefined : 'auto' }}
            >
              Back to gallery
            </button>
          ) : configurable ? (
            <button
              onClick={() => setConfigDraft({ ...(selected?.model?.config ?? {}) })}
              type="button"
              className="ui2-chip"
              style={{ flex: '0 0 auto', marginLeft: selected ? undefined : 'auto' }}
            >
              Configure
            </button>
          ) : null}
          <button
            onClick={() => commit(selected, configDraft)}
            type="button"
            disabled={!selected || busy}
            className="ui2-primary-action"
            style={{ flex: '0 0 auto', marginLeft: selected ? undefined : 'auto' }}
          >
            {busy && <Loader2 size={14} className="spinning" />}
            Use Template
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * The config editor behind "Configure".
 *
 * Only scalar entries are editable: a nested list like `inception_blocks` cannot
 * be typed into a text field, and the previous version rendered it as
 * "[object Object]" and then sent that string back as the value. Non-scalars are
 * named as untouched rather than silently dropped.
 */
const ConfigEditor: React.FC<{
  name: string;
  draft: Record<string, any>;
  onChange: (next: Record<string, any>) => void;
}> = ({ name, draft, onChange }) => {
  const scalars = Object.entries(draft).filter(
    ([, v]) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean',
  );
  const fixed = Object.keys(draft).filter((k) => !scalars.some(([sk]) => sk === k));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div className="ui2-eyebrow">Configure {name}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 12 }}>
        {scalars.map(([key, value]) => (
          typeof value === 'boolean' ? (
            <label key={key} className="ui2-check">
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => onChange({ ...draft, [key]: e.target.checked })}
              />
              <span style={{ ...MONO, fontSize: 11.5, color: 'var(--ui2-text-dim)' }}>{key}</span>
            </label>
          ) : (
            <label key={key} style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <span style={{ ...MONO, fontSize: 11.5, color: 'var(--ui2-text-dim)' }}>{key}</span>
              <input
                type={typeof value === 'number' ? 'number' : 'text'}
                value={value}
                onChange={(e) => onChange({
                  ...draft,
                  [key]: typeof value === 'number' ? Number(e.target.value) : e.target.value,
                })}
                style={S.field}
              />
            </label>
          )
        ))}
      </div>
      {fixed.length > 0 && (
        <div style={S.hint}>
          Structured values stay at their defaults here:{' '}
          <span style={{ ...MONO, fontSize: 11.5 }}>{fixed.join(', ')}</span>
        </div>
      )}
      {scalars.length === 0 && (
        <div style={S.hint}>This architecture exposes no scalar settings.</div>
      )}
    </div>
  );
};

export default ModernTemplateGallery;
