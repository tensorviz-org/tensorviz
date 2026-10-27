import React, { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { X, User, Sparkles, RotateCcw } from 'lucide-react';
import { ensureDefaultProject, listVersions, restoreVersion, GraphVersion } from '../../services/projectsApi';
import * as hostServices from '../../ide/hostServices';

interface HistoryPanelV2Props {
  open: boolean;
  onClose: () => void;
}

function relativeTime(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return '';
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/**
 * Figma-style version timeline: shows who (You / AI) changed the graph and when,
 * with one-click restore. Reads the current graph id from the route so it needs
 * no prop threading; empty until the graph has been saved to the server.
 */
export default function HistoryPanelV2({ open, onClose }: HistoryPanelV2Props) {
  const { graphId } = useParams<{ graphId: string }>();
  const [pid, setPid] = useState<string | null>(null);
  const [versions, setVersions] = useState<GraphVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState<number | null>(null);

  const load = useCallback(async () => {
    if (!graphId) return;
    setLoading(true);
    setError(null);
    try {
      const project = await ensureDefaultProject();
      setPid(project);
      const vs = await listVersions(project, graphId);
      setVersions([...vs].sort((a, b) => b.revision - a.revision)); // newest first
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, [graphId]);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  const restore = async (rev: number) => {
    if (!pid || !graphId) return;
    setRestoring(rev);
    try {
      const currentRevision = versions.reduce((latest, version) => Math.max(latest, version.revision), 0);
      await restoreVersion(pid, graphId, rev, currentRevision);
      // Re-fetch the restored graph into the builder (the route loads on mount).
      hostServices.reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Restore failed');
      setRestoring(null);
    }
  };

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-label="Version history"
      style={{ position: 'absolute', inset: 0, zIndex: 60, display: 'flex', justifyContent: 'flex-end' }}
      onClick={onClose}
    >
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.4)' }} />
      <div
        className="ui2-panel"
        onClick={(e) => e.stopPropagation()}
        style={{ position: 'relative', width: 360, maxWidth: '90vw', height: '100%', display: 'flex', flexDirection: 'column', borderRadius: 0 }}
      >
        <div className="flex items-center gap-2 p-4" style={{ borderBottom: '1px solid var(--ui2-border)' }}>
          <RotateCcw size={15} style={{ color: 'var(--ui2-cyan)' }} />
          <div className="font-medium">Version history</div>
          <button className="ui2-chip ml-auto" onClick={onClose} aria-label="Close"><X size={13} /></button>
        </div>

        <div className="flex-1 overflow-auto p-3 space-y-2">
          {!graphId && (
            <div className="ui2-muted text-sm p-3">Save this graph to start tracking its history.</div>
          )}
          {graphId && loading && <div className="ui2-muted text-sm p-3">Loading…</div>}
          {error && (
            <div className="ui2-panel p-3 text-sm" style={{ borderColor: 'rgba(255,123,114,0.4)' }} role="alert">{error}</div>
          )}
          {graphId && !loading && !error && versions.length === 0 && (
            <div className="ui2-muted text-sm p-3">No versions yet.</div>
          )}
          {versions.map((v, i) => {
            const isAgent = v.author === 'agent';
            return (
              <div key={v.revision} className="ui2-panel p-3 flex items-center gap-3">
                <span
                  className="flex items-center gap-1 text-xs px-2 py-0.5 rounded"
                  style={{ color: isAgent ? 'var(--ui2-cyan)' : 'var(--ui2-text-dim)', border: '1px solid var(--ui2-border)' }}
                  title={isAgent ? 'Edited by AI' : 'Edited by you'}
                >
                  {isAgent ? <Sparkles size={11} /> : <User size={11} />}{isAgent ? 'AI' : 'You'}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-sm truncate">{v.opSummary || (isAgent ? 'AI edit' : 'Edit')}</div>
                  <div className="ui2-muted text-xs">rev {v.revision} · {relativeTime(v.createdAt)}</div>
                </div>
                {i !== 0 && (
                  <button
                    className="ui2-chip"
                    onClick={() => restore(v.revision)}
                    disabled={restoring !== null}
                    title="Restore this version"
                  >
                    {restoring === v.revision ? '…' : 'Restore'}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
