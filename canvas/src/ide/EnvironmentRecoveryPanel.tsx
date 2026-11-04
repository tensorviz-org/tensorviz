import React from 'react';
import { GraphSetupProgress } from './GraphSetupProgress';
import { ArrowLeft, ArrowUpRight, Check, ChevronRight, FolderOpen, LoaderCircle, RotateCw, ShieldCheck, Sparkles, SquareTerminal, TriangleAlert, X } from 'lucide-react';
import type { GraphEnvironmentCandidate, GraphOpeningState, GraphSetupAction } from './protocol.generated';
import { isModelEnvironmentName } from './protocol.generated';
import './EnvironmentRecoveryPanel.css';

function pathText(value: string) {
  return value.split(/([/\\._-])/).map((part, index) => <React.Fragment key={index}>{part}{/^[/\\._-]$/.test(part) && <wbr />}</React.Fragment>);
}
function identityOf(candidate: GraphEnvironmentCandidate) {
  if (candidate.managed) return 'TensorViz Python';
  const parts = (candidate.prefix || candidate.interpreter.replace(/[/\\][^/\\]+$/, '').replace(/[/\\](?:bin|Scripts)$/i, '')).split(/[/\\]/).filter(Boolean);
  return parts.at(-3) === 'model-environments' && /^model-/.test(parts.at(-2) ?? '')
    ? parts.at(-1)! : parts.slice(-2).join('/') || candidate.interpreter;
}
function libraryStatus(candidate: GraphEnvironmentCandidate) {
  return candidate.status === 'libraries_present' ? 'Capture libraries present' : candidate.status === 'missing_libraries'
    ? `Missing: ${candidate.missing?.join(', ') || 'capture libraries'}` : candidate.status === 'incompatible'
      ? `Incompatible: ${candidate.incompatible?.join(', ') || 'capture libraries'}` : candidate.detail || 'Not available';
}
/** One entry per distribution, without environment markers: the capture and model checks can both name torch or numpy. */
function distinct(requirements: string[]) {
  const name = (text: string) => (/^\s*([A-Za-z0-9][A-Za-z0-9._-]*)/.exec(text)?.[1] ?? text).toLowerCase().replace(/[-_.]+/g, '-');
  const items = requirements.map(text => text.split(';')[0].trim());
  return items.filter((item, index) => items.findIndex(other => name(other) === name(item)) === index);
}
/** What this environment still needs for the model, from the opening's static check of every candidate. */
function modelFit(candidate: GraphEnvironmentCandidate) {
  if (!candidate.modelChecked || candidate.status === 'unavailable') return undefined;
  const install = distinct([...candidate.missing ?? [], ...candidate.modelMissing ?? []]);
  const change = distinct([...candidate.incompatible ?? [], ...candidate.modelIncompatible ?? []]);
  return !install.length && !change.length ? 'Ready for this model'
    : [install.length ? `To install: ${install.join(', ')}` : '', change.length ? `To change: ${change.join(', ')}` : ''].filter(Boolean).join(' · ');
}
function matches(candidate: GraphEnvironmentCandidate, query: string) {
  const needle = query.trim().toLowerCase();
  return !needle || [identityOf(candidate), candidate.interpreter, candidate.prefix, candidate.source,
    candidate.python && `py ${candidate.python}`, candidate.torch && `torch ${candidate.torch}`].some(text => text?.toLowerCase().includes(needle));
}

export function EnvironmentRecoveryPanel({ state, connected, standalone, onAction: sendAction, onClose }: {
  state: GraphOpeningState; connected: boolean; standalone?: boolean;
  onAction(action: GraphSetupAction): void; onClose?(): void;
}) {
  const environment = state.environment!;
  // The host recommends only when the saved choice is not the user's own; that environment needs the fewest changes.
  const suggestion = environment.recommendedId ?? environment.selectedId
    ?? environment.candidates.find(candidate => candidate.pythonSelected && candidate.status !== 'unavailable')?.id
    ?? environment.candidates.find(candidate => candidate.status !== 'unavailable')?.id ?? '';
  const [selected, setSelected] = React.useState(suggestion);
  const [manualOpeningId, setManualOpeningId] = React.useState<string>();
  const manual = standalone || manualOpeningId === state.id;
  const [query, setQuery] = React.useState('');
  React.useEffect(() => setQuery(''), [state.id]);
  // Model-checked alternatives stay visible even when the saved choice is recommended or explicit.
  const ranked = environment.candidates.some(candidate => candidate.modelChecked);
  const shown = environment.candidates.filter(item => matches(item, query));
  const defaultName = `${state.target.cls.replace(/[^A-Za-z0-9_-]/g, '').replace(/^[^A-Za-z0-9]+/, '').slice(0, 44) || 'model'}-env`;
  const [name, setName] = React.useState(state.environmentName ?? defaultName);
  const [creatingName, setCreatingName] = React.useState<string>();
  React.useEffect(() => setSelected(suggestion), [state.id, suggestion]);
  React.useEffect(() => { setName(state.environmentName ?? defaultName); setCreatingName(undefined); }, [state.id, state.environmentName, defaultName]);
  const onAction = (action: GraphSetupAction) => { if (action.kind !== 'create_environment') setCreatingName(undefined); sendAction(action); };
  const create = () => { setCreatingName(name); onAction({ kind: 'create_environment', name }); };
  const busy = ['resolving', 'saving', 'capturing'].includes(state.phase);
  const showMethods = !manual && !busy;
  const agentButtonRef = React.useRef<HTMLButtonElement>(null);
  const manualHeadingRef = React.useRef<HTMLHeadingElement>(null);
  React.useEffect(() => {
    if (showMethods) agentButtonRef.current?.focus();
    else if (manual && !standalone && !busy) manualHeadingRef.current?.focus();
  }, [showMethods, manual, standalone, busy, state.id]);
  const bodyRef = React.useRef<HTMLDivElement>(null);
  React.useLayoutEffect(() => {
    // A result must be visible even after choosing an interpreter lower in the list.
    if (!busy && bodyRef.current) bodyRef.current.scrollTop = 0;
  }, [busy, state.id, state.detail, environment.detail, environment.selectedId]);
  const candidate = environment.candidates.find(item => item.id === selected);
  const savedCandidate = environment.candidates.find(item => item.id === environment.selectedId);
  const confirmed = !!candidate && selected === environment.selectedId;
  const failure = state.phase === 'failed' || environment.reason === 'worker_failed';
  const stopped = state.phase === 'cancelled';
  const missing = environment.reason === 'missing_libraries';
  const incompatible = environment.reason === 'incompatible';
  const blocked = missing || incompatible || failure;
  const openingUpdate = state.phase !== 'failed' && state.detail !== environment.detail ? state.detail : undefined;
  const unavailable = state.failure?.code === 'PACKAGE_UNAVAILABLE';
  const repair = (!failure || (!!state.failure?.code && !unavailable)) && confirmed && missing && !!environment.requirements?.length;
  const needsProjectSetup = !failure && missing && !environment.requirements?.length;
  const investigate = confirmed && needsProjectSetup;
  const copy = !failure && confirmed && incompatible && !!environment.compatibilityPackages?.length;
  // Conflicts with the project's own pins can be tried as installed.
  const pinned = !failure && confirmed && incompatible && !!environment.declaredPins?.length && !!savedCandidate;
  const selectable = environment.candidates.some(item => item.status !== 'unavailable');
  const canCreate = !!environment.canCreate && !standalone;
  const createPrimary = canCreate && !failure && (!selectable || (confirmed && blocked && !repair && !copy));
  const sourcePrimary = canCreate && failure && !repair && selected === suggestion;
  const creating = busy ? !!(state.environmentName ?? creatingName) : createPrimary;
  const identity = candidate ? identityOf(candidate) : '';
  const requirements = copy ? environment.compatibilityPackages! : environment.requirements ?? [];
  const torchRequested = repair && requirements.some(requirement => /^torch(?:$|[<>=!~[\s])/.test(requirement));
  const title = busy ? 'Preparing your environment' : stopped ? 'Environment setup stopped' : failure ? 'Environment setup failed' : !selectable ? 'No usable Python found'
    : environment.recommendedId ? 'Choose Python for this model' : repair ? 'Add the missing libraries' : copy ? 'Review library compatibility' : confirmed && incompatible ? 'Environment needs different libraries'
      : confirmed && missing ? 'Model libraries are missing' : 'Choose Python for this model';
  const hint = !connected ? 'Reconnecting. Your model and choice stay here.' : busy ? state.detail : !selectable ? 'None of the detected interpreters can be used.'
    : failure ? 'Choose another Python or retry after fixing the issue.' : repair ? 'Missing libraries in this environment.' : copy ? 'Prepare a separate compatibility copy.'
      : confirmed && blocked ? 'Your choice is saved. This model cannot run with it yet.' : 'Picking one changes nothing until you use it.';
  const primaryAction: GraphSetupAction = sourcePrimary ? { kind: 'source_only' } : createPrimary ? { kind: 'create_environment' } : !selectable ? { kind: standalone ? 'refresh_environments' : 'environment' }
    : repair || copy ? { kind: 'install_dependencies' } : investigate ? { kind: 'agent_help' } : { kind: 'select_environment', candidateId: selected };
  const primaryLabel = busy ? state.phase === 'saving' ? 'Saving…' : state.phase === 'capturing' ? 'Capturing graph…' : 'Preparing environment…'
    : sourcePrimary ? 'Inspect source only' : createPrimary ? 'Create model environment…' : !selectable ? standalone ? 'Refresh list' : 'Browse for Python…'
      : !confirmed ? <>Use <code>{identity}</code>{environment.selectedId ? ' instead' : ''}</> : copy ? 'Review compatibility copy…'
        : repair ? failure || stopped ? 'Retry library setup…' : canCreate ? 'Add libraries here…' : standalone ? 'Install missing libraries' : 'Review missing libraries…'
          : investigate ? standalone ? 'Copy MCP investigation request' : 'Use Agent' : 'Check again with this environment';
  const nameForm = <div className="graph-environment-slip neutral">
    <h2>Create new environment</h2>
    <label className="graph-environment-name-field">Environment name
      <input value={name} maxLength={48} disabled={!connected} onChange={event => setName(event.target.value)} aria-invalid={!isModelEnvironmentName(name)} aria-describedby="environment-name-hint" />
    </label>
    <p id="environment-name-hint">{isModelEnvironmentName(name) ? 'A separate Python for this model. Your current environment stays unchanged.' : 'Use 1–48 letters, numbers, hyphens or underscores; avoid reserved system names.'}</p>
    <details><summary><ChevronRight aria-hidden="true" />Creation details</summary>
      <p>Saved in tensorViz extension storage. Required libraries are reviewed before installation; source builds need approval. PyTorch downloads may be large.</p>
    </details>
    {!createPrimary && <button type="button" disabled={!connected || !isModelEnvironmentName(name)} onClick={create}>{failure && !state.failure?.code ? 'Retry creation' : 'Create model environment…'}</button>}
  </div>;
  const status = busy ? state.detail : failure ? state.detail.split('\n')[0] : stopped ? 'Setup stopped. You can retry when ready.'
    : !confirmed ? (candidate && modelFit(candidate)) ?? 'This Python has not been checked for your model.' : repair ? candidate?.status === 'libraries_present'
      ? 'Model libraries missing · capture tools are installed.' : 'Required libraries missing.' : 'Model check needs attention.';
  const diagnosticSummary = failure || openingUpdate ? (failure ? state.detail : openingUpdate!).split('\n')[0]
    : !canCreate || environment.reason === 'selection_required' ? environment.detail
      : missing ? 'This Python needs libraries to run your model.' : incompatible ? 'This Python’s libraries do not match the model.' : 'Choose Python to continue.';
  // Keep the summary once. The disclosure adds only diagnostic information that is not already visible.
  const diagnosticDetails = Array.from(new Set([
    ...(canCreate || failure || openingUpdate ? [environment.detail] : []),
    ...(failure || openingUpdate ? [state.detail] : []),
  ].map(detail => detail === diagnosticSummary ? '' : detail.startsWith(`${diagnosticSummary}\n`)
    ? detail.slice(diagnosticSummary.length).trim() : detail).filter(Boolean)));
  const methodIssue = failure ? state.detail.split('\n')[0] : undefined;
  const methodDetails = Array.from(new Set([environment.detail, ...(failure ? [state.detail] : [])]
    .map(detail => !methodIssue ? detail : detail === methodIssue ? '' : detail.startsWith(`${methodIssue}\n`)
      ? detail.slice(methodIssue.length).trim() : detail).filter(Boolean)));
  const modelContext = <div className="graph-environment-model" title={`${state.target.cls} · ${state.target.file}`}>
    <span className={`graph-opening-indicator${busy ? ' running' : ''}${!connected ? ' disconnected' : ''}`} /><span>{state.target.cls}</span><code>{state.target.file}</code>
    {onClose && <button type="button" className="graph-opening-close" aria-label="Close environment setup" onClick={onClose}><X aria-hidden="true" /></button>}
  </div>;
  if (showMethods) return <section className="graph-opening-panel graph-environment-panel graph-environment-methods-panel" aria-label="Python environment for model"
    onKeyDown={event => { if (event.key === 'Escape' && onClose) { event.stopPropagation(); onClose(); } }}>
    <header className="graph-environment-header">
      {modelContext}
      <h1>{failure ? 'Project setup needs attention' : stopped ? 'Project setup stopped' : 'Set up this project'}</h1>
      <p id="environment-purpose">Python runs your model to record tensor shapes.</p>
      <GraphSetupProgress state={state} />
    </header>
    <div className="graph-environment-body" ref={bodyRef}>
      {(failure || stopped) && <p className="graph-environment-diagnostic graph-environment-opening-issue" role={failure ? 'alert' : 'status'}>{methodIssue ?? 'Setup stopped. Your model and saved choices are still here.'}</p>}
      <section className="graph-setup-methods" aria-label="Choose a setup method">
        <button ref={agentButtonRef} type="button" data-setup-agent className="graph-setup-agent" disabled={!connected}
          aria-describedby="environment-purpose" onClick={() => onAction({ kind: 'agent_help' })}>
          <Sparkles size={16} aria-hidden="true" /><span className="graph-setup-method-label">Set up project with agent</span><ArrowUpRight size={16} aria-hidden="true" />
        </button>
        {!connected && <p role="status">Reconnecting. Setup is available when the connection returns.</p>}
        <button type="button" className="graph-environment-manual-choice" onClick={() => setManualOpeningId(state.id)}>Set up manually</button>
      </section>
      <details className="graph-environment-setup-details"><summary><ChevronRight aria-hidden="true" />Learn more</summary>
        <p>Your agent checks the project, prepares Python and test inputs, then opens the graph. You review any dependency installation.</p>
        {methodDetails.length > 0 && <>
          <h2>Current setup</h2>
          {methodDetails.map(detail => <p key={detail} className="graph-environment-diagnostic">{detail}</p>)}
        </>}
      </details>
    </div>
    <footer className="graph-environment-footer">
      <div className="graph-environment-actions">
        <button type="button" disabled={!connected} onClick={() => onAction({ kind: 'source_only' })}>Inspect source only</button>
        <button className="graph-environment-cancel" type="button" disabled={!connected} onClick={() => onAction({ kind: 'cancel' })}>Cancel</button>
      </div>
    </footer>
  </section>;
  return <section className="graph-opening-panel graph-environment-panel" aria-label="Python environment for model" aria-busy={busy}
    onKeyDown={event => { if (event.key === 'Escape' && onClose) { event.stopPropagation(); onClose(); } }}>
    <header className="graph-environment-header">
      {modelContext}
      {!standalone && !busy && <button type="button" className="graph-environment-back" onClick={() => setManualOpeningId(undefined)}><ArrowLeft aria-hidden="true" />Setup options</button>}
      <h1 ref={manualHeadingRef} tabIndex={-1}>{title}</h1><p role="status" title={hint}>{hint}</p>
      <GraphSetupProgress state={state} />
    </header>
    <div className="graph-environment-body" ref={bodyRef}>
      {busy ? <div className="graph-environment-slip"><h2>Preparing <code>{state.environmentName ?? creatingName ?? identity}</code></h2></div>
        : <div className={`graph-environment-slip ${failure ? 'error' : blocked ? 'warning' : 'neutral'}`} role={!canCreate && (blocked || openingUpdate) ? 'alert' : undefined}>
          <h2>{failure ? 'Could not finish setup' : (missing || incompatible) && savedCandidate ? <>Model check for <code>{identityOf(savedCandidate)}</code></> : 'Environment check'}</h2>
          <p className={failure || openingUpdate ? 'graph-environment-diagnostic' : undefined}>{diagnosticSummary}</p>
          {(diagnosticDetails.length > 0 || (missing && savedCandidate?.status === 'libraries_present' && !failure)) && <details><summary><ChevronRight aria-hidden="true" />Setup details</summary>
            {diagnosticDetails.map(detail => <p key={detail} className="graph-environment-diagnostic">{detail}</p>)}
            {missing && savedCandidate?.status === 'libraries_present' && !failure && <p>Capture libraries are installed; the missing libraries belong to this model.</p>}
          </details>}
          {!selectable && <p>{canCreate ? 'tensorViz can create a separate environment and continue opening this model.' : standalone ? 'Create an environment for this project, then refresh.' : 'Browse for another interpreter, or create one and refresh.'}</p>}
        </div>}
      {canCreate && !busy && createPrimary && !ranked && nameForm}
      {!busy && <details className="graph-environment-existing" open={!canCreate || ranked}><summary><ChevronRight aria-hidden="true" />{canCreate ? 'Change environment' : `Use existing Python (${environment.candidates.length})`}</summary>
      {!busy && pinned && <div className="graph-environment-slip neutral">
        <h2>Continue with <code>{identityOf(savedCandidate!)}</code></h2>
        <p>Run the model with the libraries already installed. The project’s pins and this environment stay unchanged; if the model fails, its error appears here.</p>
        {environment.replaceable ? <p>Or install <code>{environment.declaredPins!.join(', ')}</code> here. tensorViz first checks that a build exists for Python {savedCandidate!.python ?? 'in this environment'}, then VS Code lists every version change for your approval.</p>
          : savedCandidate!.managed && <p className="graph-environment-note"><TriangleAlert aria-hidden="true" /><span>The tensorViz-managed environment is shared by every model, so its versions are not replaced. Choose your own environment to install the pinned versions.</span></p>}
        <div className="graph-environment-pin-actions">
          <button type="button" disabled={!connected} onClick={() => onAction({ kind: 'use_installed_versions' })}>Try with installed versions</button>
          {environment.replaceable && <button type="button" disabled={!connected} onClick={() => onAction({ kind: 'replace_versions' })}>Install in this environment…</button>}
        </div>
      </div>}
      {!busy && selectable && (repair || copy) && <div className="graph-environment-slip warning">
              <h2>{copy ? 'Align libraries for' : 'Add to'} <code>{identity}</code></h2>
              <p>{requirements.length} requested {requirements.length === 1 ? 'library' : 'libraries'}{copy ? ' in a separate compatibility copy.' : ', added beside what’s installed.'}</p>
              {torchRequested && !candidate?.torch && <p className="graph-environment-note"><TriangleAlert aria-hidden="true" /><span>PyTorch can be large{standalone ? '.' : '; review its exact build.'} Need a specific CUDA build? Choose another environment.</span></p>}
              <details><summary><ChevronRight aria-hidden="true" />{copy ? 'Packages to align' : 'Requested libraries'}</summary><ul>{requirements.map(requirement => <li key={requirement}>{requirement}</li>)}</ul></details>
              <p className="graph-environment-note"><ShieldCheck aria-hidden="true" /><span>{standalone ? copy ? 'Uses your TensorViz plugin permission to prepare this compatibility copy.' : 'Uses your TensorViz plugin permission for this exact, additive install.' : 'VS Code lists exact packages for your approval. Nothing installs before that.'}</span></p>
              {repair && candidate?.torch && <p className="graph-environment-note"><Check aria-hidden="true" /><span>Torch {candidate.torch} and its device build are kept.</span></p>}
            </div>}
      {environment.candidates.length > 1 && <input type="search" className="graph-environment-search" aria-label="Search environments"
        placeholder="Search by name, path or version" value={query} disabled={!connected} onChange={event => setQuery(event.target.value)} />}
      <fieldset disabled={!connected || busy} className="graph-environment-choices"><legend className="graph-setup-sr-only">Detected environments</legend>
        {shown.map(item => { const fit = modelFit(item); return <div key={item.id} className={`graph-environment-entry${selected === item.id ? ' selected' : ''}`}><label className={`graph-environment-choice ${item.status}`}
          data-fit={fit === undefined ? undefined : fit === 'Ready for this model' ? 'ready' : item.incompatible?.length || item.modelIncompatible?.length ? 'change' : 'install'}>
          <input type="radio" name="capture-environment" value={item.id} checked={selected === item.id} disabled={item.status === 'unavailable'} onChange={() => setSelected(item.id)} />
          <span className="graph-environment-name"><strong title={identityOf(item)}>{identityOf(item)}</strong>
            {item.id === environment.selectedId && <small className="graph-environment-active"><Check aria-hidden="true" />Saved</small>}
            {item.id === environment.recommendedId && <small className="graph-environment-recommended">Recommended</small>}</span>
          <span className="graph-environment-version">{item.python && <span>py {item.python}</span>}{item.torch !== undefined && <span>{item.torch ? `torch ${item.torch}` : 'no torch'}</span>}</span>
          <span className="graph-environment-source" title={item.source}>{item.managed ? 'Provided by TensorViz · shared across models' : item.source}{item.pythonSelected && ' · VS Code interpreter'}</span>
          {selected !== item.id && (fit ? <span className={`graph-environment-status${fit === 'Ready for this model' ? ' ready' : ''}`}>{fit}</span>
            : <span className={item.status === 'libraries_present' ? 'graph-setup-sr-only' : 'graph-environment-status'}>{libraryStatus(item)}</span>)}
          {selected === item.id && <span className="graph-environment-facts"><span>Capture libraries</span><span>{libraryStatus(item)}</span>
            {fit && <><span>This model</span><span className={fit === 'Ready for this model' ? 'ready' : 'needs'}>{fit}</span></>}
            {!fit && confirmed && (missing || incompatible) && !failure && !busy && <><span>Model check</span><span>Blocked · see above</span></>}</span>}
        </label>{selected === item.id && <details className="graph-environment-paths"><summary><ChevronRight aria-hidden="true" />Interpreter details</summary><div className="graph-environment-facts"><span>Interpreter</span><code>{pathText(item.interpreter)}</code>
            {item.prefix && <><span>Prefix</span><code>{pathText(item.prefix)}</code></>}
            {[item.implementation, item.abi, item.architecture].some(Boolean) && <><span>Build</span><code>{[item.implementation, item.abi, item.architecture].filter(Boolean).join(' · ')}</code></>}
          </div></details>}</div>; })}
        {!environment.candidates.length && <p>No existing Python environments were detected for this project.</p>}
        {!!environment.candidates.length && !shown.length && <p className="graph-environment-empty">No environments match “{query.trim()}”.</p>}
      </fieldset>
      {canCreate && !createPrimary && !ranked && nameForm}</details>}
      {canCreate && !busy && ranked && nameForm}
      {!busy && <div className="graph-environment-help">
        {needsProjectSetup && <>
          <button type="button" disabled={!connected || busy} onClick={() => onAction({ kind: 'source_only' })}>Inspect source only</button></>}
        {standalone && <details open={!canCreate && (failure || investigate)}><summary><ChevronRight aria-hidden="true" />Investigate with MCP</summary>
          <p>Let your coding agent inspect this environment and opening attempt directly over MCP.</p>
          {(!investigate || createPrimary) && <button type="button" disabled={!connected || busy} onClick={() => onAction({ kind: 'agent_help' })}><SquareTerminal aria-hidden="true" />Copy MCP investigation request</button>}
        </details>}
      </div>}
    </div>
    <footer className="graph-environment-footer">
      <div className={`graph-environment-result${failure ? ' error' : ''}`} role={canCreate && failure ? 'alert' : 'status'} title={status}>
        {failure ? <TriangleAlert aria-hidden="true" /> : busy ? <LoaderCircle className="graph-environment-spinner" aria-hidden="true" /> : <ShieldCheck aria-hidden="true" />}<span>{status}</span>
      </div>
      {busy && <div className="graph-environment-progress" role="progressbar" aria-label="Environment setup in progress" />}
      <p className="graph-environment-selection" role="status"><span>{creating ? 'New' : confirmed ? 'Saved' : candidate ? 'Choice' : 'No environment chosen'}</span>{(creating || candidate) && <><code title={creating ? name : identity}>{creating ? name : identity}</code>{!creating && !confirmed && <span>· not used yet</span>}</>}</p>
      <button className="graph-setup-primary" type="button" disabled={!connected || busy || (createPrimary && !isModelEnvironmentName(name)) || (!createPrimary && !sourcePrimary && selectable && (!candidate || candidate.status === 'unavailable'))}
        onClick={() => createPrimary ? create() : onAction(primaryAction)}>{busy && <LoaderCircle className="graph-environment-spinner" aria-hidden="true" />}{primaryLabel}</button>
      <div className="graph-environment-actions">
        {!standalone && selectable && <button type="button" disabled={!connected || busy} onClick={() => onAction({ kind: 'environment' })}><FolderOpen aria-hidden="true" />Browse…</button>}
        {(selectable || !standalone) && <button type="button" disabled={!connected || busy} onClick={() => onAction({ kind: 'refresh_environments' })}><RotateCw aria-hidden="true" />Refresh</button>}
        <button className="graph-environment-cancel" type="button" disabled={!connected} onClick={() => onAction({ kind: 'cancel' })}>{busy ? 'Stop' : 'Cancel'}</button>
      </div>
    </footer>
  </section>;
}
