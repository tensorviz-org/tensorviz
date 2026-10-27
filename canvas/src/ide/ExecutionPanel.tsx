import React from 'react';
import { CheckCircle2, ChevronDown, Cpu, History, Pencil, Play, RefreshCw, Server, Square, X } from 'lucide-react';
import type { ExecutionReviewAction, ExecutionReviewState, GraphSetupAction, GraphSetupEvidence } from './protocol.generated';
import './ExecutionPanel.css';

const bytes = (n: number | null) => n === null ? 'Unknown' : `${(n / 1024 ** 3).toFixed(1)} GiB`;
const percent = (n: number | null) => n === null ? 'Unknown' : `${Math.round(n)}%`;
function inputRows(serialized: string): Array<{ name: string; value: string; field: (string | number)[] }> {
  try {
    const inputs = JSON.parse(serialized);
    if (inputs.factory) return [{ name: 'Factory', value: `${inputs.factory.file}:${inputs.factory.callable}()`, field: ['inputs', 'factory'] }];
    if (inputs.fixture) return [{ name: 'Fixture', value: inputs.fixture, field: ['inputs', 'fixture'] }];
    const binding = inputs.binding ?? inputs;
    return [...(binding.args ?? []).map((value: unknown, i: number) => ({ name: `Argument ${i + 1}`, value, field: ['inputs', 'args', i] })),
      ...Object.entries(binding.kwargs ?? {}).map(([name, value]) => ({ name, value, field: ['inputs', 'kwargs', name] }))]
      .map(({ name, value, field }: { name: string; value: any; field: (string | number)[] }) => ({ name,
        value: Array.isArray(value?.shape) ? `${value.shape.length ? value.shape.join(' × ') : 'Scalar'} · ${value.dtype}` : JSON.stringify(value),
        field: Array.isArray(value?.shape) ? [...field, 'shape'] : field }));
  } catch { return []; } // Host summaries may be truncated; retain the supplied input data below.
}
/** Local time of a reused result's original run; the date too when it was not today. */
export function reusedTime(capturedAt: number, now = Date.now()): string {
  const then = new Date(capturedAt);
  const time = then.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return then.toDateString() === new Date(now).toDateString() ? time : `${then.toLocaleDateString()} ${time}`;
}
function Usage({ label, value }: { label: string; value: number | null }) {
  return <div><dt>{label}</dt><dd className="execution-usage">{value !== null && <meter aria-label={label} min={0} max={100} value={value} />}<span>{percent(value)}</span></dd></div>;
}
export default function ExecutionPanel({ state, connected, expanded, onExpandedChange, onAction, onRecovery, onRepeat, onRunFresh, repeatDisabled = false, evidence = [], bottomBoundary }: {
  state: ExecutionReviewState; connected: boolean; onAction(action: GraphSetupAction): void; onRecovery?(): void;
  expanded: boolean; onExpandedChange(expanded: boolean): void;
  onRepeat?(): void; repeatDisabled?: boolean;
  /** Run again, bypassing any reused identical result (execution cache). */
  onRunFresh?(): void;
  evidence?: GraphSetupEvidence[];
  bottomBoundary?: HTMLElement | null;
}): React.ReactElement {
  const [now, setNow] = React.useState(Date.now());
  const heading = React.useRef<HTMLHeadingElement>(null);
  const surface = React.useRef<HTMLElement>(null);
  const [availableHeight, setAvailableHeight] = React.useState<number>();
  React.useLayoutEffect(() => {
    const anchor = surface.current?.closest('.local-browser-header') ?? surface.current;
    const measure = () => setAvailableHeight(Math.max(0, Math.min(window.innerHeight, bottomBoundary?.getBoundingClientRect().top ?? window.innerHeight)
      - (anchor?.getBoundingClientRect().bottom ?? 0) - 16));
    measure(); window.addEventListener('resize', measure);
    const observer = new ResizeObserver(measure); if (anchor) observer.observe(anchor);
    if (bottomBoundary) observer.observe(bottomBoundary);
    return () => { window.removeEventListener('resize', measure); observer.disconnect(); };
  }, [bottomBoundary, expanded]);
  const trigger = React.useRef<HTMLButtonElement>(null);
  React.useEffect(() => {
    if (expanded && state.phase === 'review') heading.current?.focus();
  }, [state.id, expanded, state.phase]);
  React.useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const reviewing = state.phase === 'review';
  const running = state.phase === 'running' || state.phase === 'cancelling';
  const pending = reviewing || running;
  const resources = state.resources;
  const devices = Array.from(new Set(['cpu', state.configuredDevice, state.device, ...(state.kind === 'prepare' ? [] : resources?.devices.map(d => d.id) ?? [])]));
  const calls = state.stages.reduce((n, stage) => n + stage.calls, 0);
  const action = (value: Exclude<ExecutionReviewAction['action'], 'configure'>, device = state.device) => onAction({ kind: 'execution', reviewId: state.id, action: value, device });
  const edit = (field: (string | number)[], name: string) => state.profileName && <button type="button" className="execution-edit" disabled={!connected || running}
    title={`Edit profiles.${state.profileName}.${field.join('.')} in tensorviz.toml`} aria-label={`Edit ${name} in tensorviz.toml`}
    onClick={() => onAction({ kind: 'edit_execution', reviewId: state.id, field })}><Pencil size={12} aria-hidden="true" /><span>Edit</span></button>;
  const close = () => { onExpandedChange(false); trigger.current?.focus(); };
  const title = state.kind === 'prepare' ? 'Prepare inputs' : state.kind === 'trace' ? 'Trace inputs' : 'Run model';
  const label = (id: string) => id === 'cpu' ? 'CPU' : id === 'mps' ? 'Apple GPU · mps' : `GPU · ${id}`;
  const inputs = inputRows(state.inputs);
  // A finished invocation whose result was reused: no Python ran for it.
  const reused = state.phase === 'succeeded' ? state.reused : undefined;
  const reusedLabel = reused ? `Reused identical result from ${reusedTime(reused.capturedAt, now)} (no Python ran)` : undefined;
  return <section ref={surface} className="execution-surface" aria-label="Execution target" onKeyDown={event => { if (event.key === 'Escape' && expanded) { close(); event.stopPropagation(); } }}>
    <button ref={trigger} type="button" className="execution-target" aria-label={reusedLabel ? `Execution details. ${reusedLabel}` : 'Execution details'} aria-expanded={expanded} aria-controls="execution-sheet"
      title={`${label(state.device)} · ${resources ? `${resources.remote ? 'SSH · ' : ''}${resources.host}` : 'Checking execution host…'}`}
      onClick={() => onExpandedChange(!expanded)}>
      <Cpu size={14} aria-hidden="true" /><strong>{label(state.device)}</strong>
      <span className="execution-host">{resources ? `${resources.remote ? 'SSH · ' : ''}${resources.host}` : 'Checking execution host…'}</span>
      {reused && <span className="execution-reused-badge" aria-hidden="true"><History size={12} />Reused · no Python ran</span>}
      <ChevronDown size={13} aria-hidden="true" />
    </button>
    {expanded && <div id="execution-sheet" className="execution-sheet" style={{ maxHeight: availableHeight }}
      onWheel={event => event.stopPropagation()} onPointerDown={event => event.stopPropagation()}
      onKeyDown={event => { event.stopPropagation(); if (event.key === 'Escape') close(); }}>
      <header><div><h2 ref={heading} tabIndex={-1}>{title}</h2></div>
        <button type="button" aria-label="Collapse execution details" onClick={close}><X size={16} aria-hidden="true" /></button></header>
      <div className="execution-scroll" role="region" aria-label="Run configuration" tabIndex={0}>
        <label className="execution-device">Device for this run
          <select aria-label="Device for this run" disabled={!reviewing || !connected} value={state.device} onChange={event => action('select', event.target.value)}>
            {devices.map(id => <option key={id} value={id}>{label(id)}{resources?.devices.find(d => d.id === id)?.name ? ` — ${resources.devices.find(d => d.id === id)!.name}` : ''}</option>)}
          </select>
        </label>
        <div className="execution-host-line"><Server size={14} aria-hidden="true" /><span>{resources ? `${resources.remote ? 'Remote SSH' : 'Local Python worker'}: ${resources.host}` : 'Python worker host: checking…'}</span></div>
        {resources && <>
          <dl className="execution-readings execution-hardware"><div><dt>Processor</dt><dd>{resources.cpu}</dd></div><div><dt>Logical CPUs</dt><dd>{resources.cores}</dd></div></dl>
          <div className="execution-resource-heading"><span>System usage</span><button type="button" disabled={!connected || !pending || state.phase === 'cancelling'} onClick={() => action('refresh')} aria-label="Refresh resource readings"><RefreshCw size={12} aria-hidden="true" />Refresh</button></div>
          <dl className="execution-readings">
            <Usage label="CPU usage" value={resources.cpuPercent} />
            <div><dt>RAM available</dt><dd>{bytes(resources.memoryFree)} <small>/ {bytes(resources.memoryTotal)}</small></dd></div>
          </dl>
          <div className="execution-sample"><span>Updated {Math.max(0, Math.floor((now - resources.sampledAt) / 1000))}s ago</span><span title="Other workloads can change these values. This is not a reservation or an estimate of model fit.">Shared resources</span></div>
        </>}
        <div className="execution-workload"><div className="execution-workload-title"><strong>Planned workload</strong><span>{calls} {state.kind === 'trace' ? 'traced' : 'forward'} {calls === 1 ? 'call' : 'calls'}</span></div>
          {state.kind === 'capture' && <div className="execution-source">One call captures shapes and operations.</div>}
          <ol>{state.stages.map(stage => <li key={stage.label}><span>{stage.label}</span><b>{stage.calls}</b></li>)}</ol>
        </div>
        <div className="execution-profile-heading"><strong>Model settings</strong></div>
        <div className="execution-source">{state.profileName ? <><code>tensorviz.toml</code><span>profiles.{state.profileName}</span></> : state.kind === 'prepare' ? 'Input preparation defaults' : 'Resolved execution settings'}</div>
        <dl className="execution-readings execution-mode"><div><dt title="PyTorch module training / evaluation mode (mode)">Mode</dt><dd><code>{state.mode}</code>{edit(['mode'], 'mode')}</dd></div>
          <div><dt title="PyTorch autograd recording (grad_enabled)">Gradients</dt><dd>{state.gradients ? 'On' : 'Off'}{edit(['grad_enabled'], 'gradients')}</dd></div></dl>
        <div className="execution-profile-heading"><strong>Inputs</strong>{edit(['inputs'], 'inputs')}</div>
        {state.kind === 'prepare' ? <div className="execution-source">Shapes pending model construction</div> : <>
          {inputs.length > 0 && <table className="execution-inputs" aria-label="Inputs"><thead><tr><th scope="col">Argument</th><th scope="col">Shape / value</th><th scope="col"><span className="execution-visually-hidden">Edit source</span></th></tr></thead>
            <tbody>{inputs.map(({ name, value, field }) => <tr key={name}><th scope="row">{name}</th><td>{value}</td><td>{edit(field, name)}</td></tr>)}</tbody></table>}
          <div className="execution-source">{state.profileName ? 'Source: saved profile · inputs' : 'Source: resolved input recipe'}</div>
          {evidence.length > 0 && <details className="execution-origin"><summary>How inputs were inferred</summary><ul>{evidence.map((item, i) => <li key={i}>{item.detail}{item.file && <code>{item.file}{item.line ? `:${item.line}` : ''}</code>}</li>)}</ul></details>}
        </>}
        <details><summary>Run settings</summary>
          <dl className="execution-readings execution-settings"><div><dt>Device selection</dt><dd>This run only</dd></div>
            <div><dt>On device error</dt><dd>Stop · no fallback</dd></div>
            <div><dt>Project code</dt><dd title="Imports, constructors and factories can do additional work, including choosing their own device.">May do extra work</dd></div>
            <div><dt>Python</dt><dd><code>{state.interpreter}</code></dd></div>
          </dl>
          {state.kind !== 'prepare' && <details className="execution-recipe"><summary>Input recipe</summary><pre>{state.inputs}</pre></details>}
        </details>
        {reused && <div className="execution-reused"><History size={14} aria-hidden="true" />
          <p><strong>Reused identical result from <time dateTime={new Date(reused.capturedAt).toISOString()}>{reusedTime(reused.capturedAt, now)}</time> (no Python ran)</strong>
            <span>Same saved sources, setup, inputs, environment and tensorViz backend as that run. Timings shown are from that run.</span></p></div>}
        {!reviewing && <p className="execution-feedback" role="status">{state.detail}</p>}
        {running && <div className="execution-progress" role="progressbar" aria-label={state.phase === 'cancelling' ? 'Stopping execution' : 'Execution in progress'} />}
        {!connected && <p role="status">Disconnected. Reconnect to control this invocation.</p>}
      </div>
      <footer><span className="execution-run-count" aria-label={`${calls} ${state.kind === 'trace' ? 'traced' : 'forward'} calls planned`}>{calls} calls</span>{pending ? <>
        <button type="button" disabled={!connected || state.phase === 'cancelling'} onClick={() => action('cancel')}>{running && <Square size={12} aria-hidden="true" />}Cancel</button>
        {/* Starting the run closes the details; the toolbar keeps progress and Cancel. */}
        {reviewing && <button className="execution-run" type="button" disabled={!connected} onClick={() => { action('run'); close(); }}><Play size={13} aria-hidden="true" />{state.kind === 'prepare' ? 'Prepare on CPU' : state.kind === 'trace' ? 'Run trace' : 'Run'}</button>}
      </> : <>{state.phase === 'failed' && onRecovery && <button type="button" disabled={!connected} onClick={() => { close(); onRecovery(); }}>Review setup</button>}<button type="button" onClick={close}><CheckCircle2 size={13} aria-hidden="true" />Close details</button>
        {reused && onRunFresh ? <button className="execution-run" type="button" disabled={!connected || repeatDisabled}
          title="Run the model in Python now, even though an identical result is available"
          onClick={() => { trigger.current?.focus(); onRunFresh(); }}><Play size={13} aria-hidden="true" />Run fresh</button>
          : onRepeat && <button className="execution-run" type="button" disabled={!connected || repeatDisabled} onClick={() => { trigger.current?.focus(); onRepeat(); }}><RefreshCw size={13} aria-hidden="true" />{state.kind === 'prepare' ? 'Prepare again' : 'Run again'}</button>}</>}</footer>
    </div>}
  </section>;
}
