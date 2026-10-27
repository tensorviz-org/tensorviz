import React from 'react';
import { createPortal } from 'react-dom';
import { ListTree } from 'lucide-react';
import CanvasTooltip from '../components/CanvasTooltip';
import type { ModelNodeData } from '../types';
import type { CaptureInspectionQuery } from '../ide/protocol.generated';
import type { CaptureInspectionResult, LocalBrowserCaptureClient } from '../ide/localBrowserCaptureClient';
import './captureDetails.css';

type Reference = NonNullable<ModelNodeData['captureRef']>;
type Place = { entity?: CaptureInspectionQuery['entity']; path: Array<string | number>; offsets: number[] };

/** Temporary read-only detail, separate from the source-edit form and canvas. */
export function CaptureDetails({ reference, label, client, stale = false, buttonLabel = 'View captured details', initialQuery }: {
  reference: Reference; label: string; client: LocalBrowserCaptureClient; stale?: boolean;
  buttonLabel?: string; initialQuery?: Pick<CaptureInspectionQuery, 'entity' | 'path'>;
}) {
  const [open, setOpen] = React.useState(false);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const wasOpen = React.useRef(false);
  React.useEffect(() => {
    if (!open && wasOpen.current) trigger.current?.focus();
    wasOpen.current = open;
  }, [open]);
  return <>
    <CanvasTooltip text="View recorded arguments, outputs and tensor details from this capture. This does not run the model.">
      <button ref={trigger} type="button" className="capture-details-trigger" aria-haspopup="dialog" aria-expanded={open}
        onClick={() => setOpen(true)}><ListTree size={15} aria-hidden="true" />{buttonLabel}</button>
    </CanvasTooltip>
    {open && createPortal(<CaptureDialog reference={reference} label={label} client={client} stale={stale} initialQuery={initialQuery} close={() => setOpen(false)} />, document.body)}
  </>;
}

function CaptureDialog({ reference, label, client, stale, initialQuery, close }: {
  reference: Reference; label: string; client: LocalBrowserCaptureClient; stale: boolean; close(): void;
  initialQuery?: Pick<CaptureInspectionQuery, 'entity' | 'path'>;
}) {
  const dialog = React.useRef<HTMLDialogElement>(null);
  const selected = React.useMemo(() => initialQuery ?? { entity: { kind: reference.kind, id: reference.semanticId }, path: [] }, [initialQuery, reference.kind, reference.semanticId]);
  const [place, setPlace] = React.useState<Place>({ ...selected, offsets: [0] });
  const [result, setResult] = React.useState<CaptureInspectionResult | null>(null);
  React.useEffect(() => { dialog.current?.showModal(); }, []);
  React.useEffect(() => {
    let active = true;
    setResult(null);
    void client.inspect({ captureId: reference.captureId, sourceId: reference.sourceId, ...(place.entity ? { entity: place.entity } : {}),
      path: place.path, offset: place.offsets[place.offsets.length - 1], limit: 25 }).then(value => { if (active) setResult(value); });
    return () => { active = false; client.cancel(); };
  }, [client, reference.captureId, reference.sourceId, place]);
  const go = (path: Place['path'], entity = place.entity) => setPlace({ entity, path, offsets: [0] });
  const page = result?.page;
  const kind = page?.items.find(item => item.key === 'kind')?.value;
  const valueId = page?.items.find(item => item.key === 'id')?.value;
  const start = place.offsets[place.offsets.length - 1];
  return <dialog ref={dialog} className="capture-details" aria-label={`Captured details: ${label}`} onCancel={close} onClose={close}
    onClick={event => { if (event.target === event.currentTarget) close(); }}>
    <div className="capture-details-content">
      <header><div><h2>{label}</h2><p>Captured evidence · read only{stale ? ' · does not verify the current source' : ''}</p></div><button aria-label="Close captured details" onClick={close}>×</button></header>
      <nav aria-label="Capture sections">
        <button onClick={() => setPlace({ ...selected, offsets: [0] })}>Selected record</button>
        {(selected.entity?.kind === 'operation' || selected.entity?.kind === 'invocation') && <>
          <button onClick={() => go(['arguments'], selected.entity)}>Arguments</button>
          <button onClick={() => go(['output'], selected.entity)}>Output</button>
        </>}
        <button onClick={() => setPlace({ path: ['coverage'], offsets: [0] })}>Coverage</button>
        <button onClick={() => setPlace({ path: [], offsets: [0] })}>Whole capture</button>
      </nav>
      <div className="capture-details-path">
        <button disabled={!place.path.length} onClick={() => go(place.path.slice(0, -1))}>← Parent</button>
        <code>{place.entity ? `${place.entity.kind} ${place.entity.id}` : 'Capture'}{place.path.map(key => typeof key === 'number' ? `[${key}]` : `.${key}`).join('')}</code>
      </div>
      <div className="capture-details-body" aria-live="polite" aria-busy={!result}>
        {!result ? <p>Reading captured details…</p> : !page ? <p role="alert">{result.detail}</p> : <>
          {kind === 'value' && typeof valueId === 'string' && <button onClick={() => go([], { kind: 'value', id: valueId })}>Inspect tensor {valueId}</button>}
          {page.node.kind === 'array' || page.node.kind === 'object' ? <>
            <p>{page.total === 0 ? 'Empty container' : `${start + 1}–${start + page.items.length} of ${page.total}`} · {page.node.kind}</p>
            <table><thead><tr><th>Field</th><th>Captured value</th></tr></thead><tbody>{page.items.map(item => <tr key={JSON.stringify(item.key)}>
              <th scope="row"><code>{String(item.key)}</code></th>
              <td>{item.kind === 'object' || item.kind === 'array'
                ? <button onClick={() => go([...place.path, item.key!])}>{item.label || item.kind} <span>· {item.count} {item.kind === 'array' ? 'items' : 'fields'} →</span></button>
                : <code>{JSON.stringify(item.value)}</code>}</td>
            </tr>)}</tbody></table>
          </> : <pre>{JSON.stringify(page.node.value)}</pre>}
        </>}
      </div>
      <footer><small>Evidence covers this capture. Empty or missing records do not prove absence.</small><div>
        <button disabled={!page || place.offsets.length < 2} onClick={() => setPlace({ ...place, offsets: place.offsets.slice(0, -1) })}>Previous page</button>
        <button disabled={!page || page.nextOffset === null} onClick={() => { if (page?.nextOffset != null) setPlace({ ...place, offsets: [...place.offsets, page.nextOffset] }); }}>Next page</button>
      </div></footer>
    </div>
  </dialog>;
}
