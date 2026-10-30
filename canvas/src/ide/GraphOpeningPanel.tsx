import React from 'react';
import { ArrowLeft, ArrowRight, ArrowUpRight, SlidersHorizontal, Sparkles, X } from 'lucide-react';
import type { GraphOpeningState, GraphSetupAction, GraphSetupDraft, GraphSetupField } from './protocol.generated';
import './GraphOpeningPanel.css';
import { EnvironmentRecoveryPanel } from './EnvironmentRecoveryPanel';

type Binding = { args: any[]; kwargs: Record<string, any> };
const emptyBinding = (): Binding => ({ args: [], kwargs: {} });

export function argumentFromJson(value: any): any {
  if (value === null) return { kind: 'none' };
  if (Array.isArray(value)) return { kind: 'list', items: value.map(argumentFromJson) };
  if (typeof value === 'object') return { kind: 'dict', items: Object.fromEntries(Object.entries(value).map(([k, v]) => [k, argumentFromJson(v)])) };
  return typeof value === 'number' && !Number.isInteger(value) ? { kind: 'float', value } : value;
}
function jsonFromArgument(value: any): any {
  if (!value || typeof value !== 'object') return value;
  if (value.kind === 'none') return null;
  if (value.kind === 'float') return value.value;
  if (['list', 'tuple'].includes(value.kind)) return value.items.map(jsonFromArgument);
  if (value.kind === 'dict') return Object.fromEntries(Object.entries(value.items).map(([k, v]) => [k, jsonFromArgument(v)]));
  return value;
}
function boundValue(binding: Binding, field: GraphSetupField, fields: GraphSetupField[]) {
  const position = fields.filter(f => ['positional', 'keyword'].includes(f.kind)).findIndex(f => f.name === field.name);
  return position >= 0 && position < binding.args.length ? binding.args[position] : binding.kwargs[field.name];
}
function bind(binding: Binding, field: GraphSetupField, fields: GraphSetupField[], value: any): Binding {
  const args = [...binding.args], kwargs = { ...binding.kwargs };
  const positional = fields.filter(f => ['positional', 'keyword'].includes(f.kind));
  const index = positional.findIndex(f => f.name === field.name);
  if (field.kind === 'positional' || (index >= 0 && index < args.length)) {
    for (let i = args.length; i < index; i++) {
      if (positional[i].defaultValue === undefined) throw new Error(`Complete ${positional[i].name} first.`);
      args.push(argumentFromJson(positional[i].defaultValue));
    }
    if (value === undefined) {
      if (index !== args.length - 1) throw new Error('Use the advanced binding editor to omit this positional value.');
      args.pop();
    } else args[index] = value;
    delete kwargs[field.name];
  } else if (value === undefined) delete kwargs[field.name];
  else kwargs[field.name] = value;
  return { args, kwargs };
}

function inputFieldsForBinding(declared: GraphSetupField[], binding: Binding): GraphSetupField[] {
  const fields = declared.filter(f => !['varargs', 'kwargs'].includes(f.kind));
  const positionalCount = fields.filter(f => ['positional', 'keyword'].includes(f.kind)).length;
  for (let index = positionalCount; index < binding.args.length; index++) {
    fields.push({ name: `Argument ${index + 1}`, kind: 'positional', required: false });
  }
  for (const name of Object.keys(binding.kwargs)) {
    if (!fields.some(f => f.name === name)) fields.push({ name, kind: 'keyword_only', required: false });
  }
  return fields;
}

function LiteralField({ label, value, placeholder, required, onChange }: {
  label: string; value: any; placeholder?: string; required?: boolean; onChange(value: any): void;
}) {
  const [text, setText] = React.useState(value === undefined ? '' : JSON.stringify(jsonFromArgument(value)));
  React.useEffect(() => setText(value === undefined ? '' : JSON.stringify(jsonFromArgument(value))), [value]);
  return <label>{label}<input value={text} required={required} placeholder={placeholder ?? 'JSON value, e.g. 64 or "name"'}
    onChange={event => {
      const next = event.target.value; setText(next);
      try { onChange(next.trim() ? argumentFromJson(JSON.parse(next)) : undefined); event.target.setCustomValidity(''); }
      catch { event.target.setCustomValidity('Enter a JSON value: number, true/false, quoted text, list or null.'); }
    }} /></label>;
}

type InputFieldProps = { field: GraphSetupField; value: any; onChange(value: any): void };
function InputField({ field, value, onChange }: InputFieldProps) {
  if (['dict', 'tuple', 'list'].includes(value?.kind)) {
    const sequence = value.kind !== 'dict';
    return <fieldset className="graph-setup-input"><legend>{field.name} · {value.kind}</legend>
      {Object.entries(value.items).map(([key, item]) => <InputField key={key}
        field={{ name: sequence ? `${field.name}[${key}]` : `${field.name}.${key}`, kind: 'keyword_only', required: true }}
        value={item} onChange={next => onChange({ ...value, items: sequence
          ? value.items.map((previous: any, index: number) => index === Number(key) ? next : previous)
          : { ...value.items, [key]: next } })} />)}
    </fieldset>;
  }
  return <SingleInputField field={field} value={value} onChange={onChange} />;
}

function SingleInputField({ field, value, onChange }: InputFieldProps) {
  const [kind, setKind] = React.useState(value?.kind === 'tensor' || (value === undefined && field.required) ? 'tensor' : 'literal');
  React.useEffect(() => { if (value !== undefined) setKind(value?.kind === 'tensor' ? 'tensor' : 'literal'); }, [value]);
  const literalKind = React.useRef(value?.kind);
  React.useEffect(() => { if (value !== undefined) literalKind.current = value?.kind; }, [value]);
  const [pending, setPending] = React.useState<any>(value?.kind === 'tensor' ? value : { kind: 'tensor', shape: [], dtype: 'float32', requires_grad: false,
    generator: { kind: 'normal', mean: 0, std: 1, seed: 0 } });
  React.useEffect(() => { if (value?.kind === 'tensor') setPending(value); }, [value]);
  const tensor = value?.kind === 'tensor' ? value : pending;
  const [shape, setShape] = React.useState(value?.kind === 'tensor' ? (value.shape.length ? value.shape.join(', ') : '[]') : '');
  React.useEffect(() => { if (value?.kind === 'tensor') setShape((value.shape.length ? value.shape.join(', ') : '[]')); }, [value?.shape, value?.kind]);
  const changeTensor = (next: any) => { setPending(next); if (shape.trim()) onChange(next); };
  return <div className="graph-setup-input">
    <div className="graph-setup-row"><strong>{field.name}{field.required ? ' *' : ''}</strong>
      <select aria-label={`${field.name} type`} value={kind} onChange={e => { setKind(e.target.value); onChange(undefined); }}>
        <option value="tensor">Tensor</option><option value="literal">Literal / default</option>
      </select></div>
    {kind === 'literal' ? <LiteralField label={`${field.name} value`} value={value} required={field.required}
      placeholder={field.defaultText ? `Default: ${field.defaultText}` : undefined}
      onChange={next => onChange(literalKind.current === 'float' && typeof next === 'number' ? { kind: 'float', value: next } : next)} /> : <>
      <label>{field.name} shape<input value={shape} required={field.required} placeholder="Dimensions, e.g. 1, 3, 256, 256"
        onChange={e => {
          const text = e.target.value; setShape(text);
          const parts = text.trim() ? text.split(',').map(s => Number(s.trim())) : [];
          const valid = text.trim() === '[]' || (text.trim() !== '' && text.split(',').every(s => /^\s*\d+\s*$/.test(s)) && parts.every(Number.isSafeInteger));
          e.target.setCustomValidity(valid || !text.trim() ? '' : 'Enter dimensions separated by commas, or [] for a scalar.');
          if (valid) onChange({ ...tensor, shape: text.trim() === '[]' ? [] : parts });
          else if (!text.trim()) onChange(undefined);
        }} /></label>
      <details className="graph-setup-tensor-options"><summary>{tensor.dtype} · {tensor.generator.kind === 'normal' ? 'Normal values · seed ' + tensor.generator.seed
        : tensor.generator.kind === 'full' ? 'Constant ' + String(tensor.generator.value) : tensor.generator.kind === 'randint'
          ? `Integers ${tensor.generator.low}–${tensor.generator.high} (exclusive)` : tensor.generator.kind} · synthetic</summary>
      <div className="graph-setup-row"><label>{field.name} dtype<select value={tensor.dtype} onChange={e => {
        const dtype = e.target.value;
        changeTensor({ ...tensor, dtype, generator: dtype.includes('float') ? { kind: 'normal', mean: 0, std: 1, seed: 0 }
          : { kind: 'full', value: dtype === 'bool' ? false : 0 } });
      }}>{['float32', 'float16', 'bfloat16', 'float64', 'int64', 'int32', 'int16', 'int8', 'bool', 'uint8'].map(v => <option key={v}>{v}</option>)}</select></label>
      <label>{field.name} values<select value={tensor.generator.kind} onChange={e => changeTensor({ ...tensor,
        generator: e.target.value === 'normal' ? { kind: 'normal', mean: 0, std: 1, seed: 0 }
          : e.target.value === 'randint' ? { kind: 'randint', low: 0, high: 10, seed: 0 } : { kind: 'full', value: tensor.dtype === 'bool' ? false : 0 } })}>
        {tensor.dtype.includes('float') && <option value="normal">Normal · seed 0</option>}
        <option value="full">Constant</option>{/^(u?int)/.test(tensor.dtype) && <option value="randint">Random integers · seed 0</option>}
        {!['normal', 'full', 'randint'].includes(tensor.generator.kind) && <option value={tensor.generator.kind}>{tensor.generator.kind} (saved)</option>}
      </select></label></div>
      {tensor.generator.kind === 'full' && <LiteralField label={`${field.name} constant`} value={tensor.generator.value}
        onChange={v => changeTensor({ ...tensor, generator: { kind: 'full', value: jsonFromArgument(v) } })} />}
      {tensor.generator.kind === 'randint' && <div className="graph-setup-row">{['low', 'high'].map(key => <label key={key}>{field.name} {key}{key === 'high' ? ' (exclusive)' : ''}
        <input type="number" value={tensor.generator[key]} onChange={e => changeTensor({ ...tensor, generator: { ...tensor.generator, [key]: Number(e.target.value) } })} /></label>)}</div>}
      </details>
    </>}
  </div>;
}

function SetupFields({ fields, label, value, resolved, renderField, source, expanded = false }: {
  fields: GraphSetupField[]; label: string; value(field: GraphSetupField): any;
  resolved?(field: GraphSetupField): boolean;
  renderField(field: GraphSetupField): React.ReactNode;
  source?(field: GraphSetupField): React.ReactNode;
  expanded?: boolean;
}) {
  const optional = fields.filter(field => !field.required || resolved?.(field));
  const overrides = optional.filter(field => value(field) !== undefined).length;
  const render = (field: GraphSetupField) => <div key={field.name} className="graph-setup-field">{renderField(field)}{source?.(field)}</div>;
  return <>{fields.filter(field => field.required && !resolved?.(field)).map(render)}
    {optional.length > 0 && <details className="graph-setup-options" open={expanded || undefined}><summary>{label} · {overrides
      ? `${overrides} configured` : `${optional.length} source default${optional.length === 1 ? '' : 's'}`}</summary>
      {optional.map(render)}
    </details>}
  </>;
}

export function initialGraphDraft(state: GraphOpeningState): GraphSetupDraft {
  return state.draft ?? { profile: { kind: 'model', model: state.discovery?.model,
    mode: 'eval', device: 'cpu', grad_enabled: false, init: emptyBinding(), inputs: emptyBinding() } };
}

function SetupEvidence({ state }: { state: GraphOpeningState }) {
  if (!state.evidence?.length) return null;
  return <details className="graph-setup-evidence"><summary>Example setup · sources and assumptions</summary>
    <p>Source values and example choices used to prepare this call. Editing inputs replaces this recipe.</p>
    <ul>{state.evidence.map((item, index) => <li key={index}>
      <strong>{item.field}</strong><span>{item.kind === 'example' ? 'Example choice' : item.kind === 'repository' ? 'Repository call' : 'Source default'}</span>
      <p>{item.detail}{item.file && <small>{item.file}:{item.line}</small>}</p>
    </li>)}</ul>
  </details>;
}

export default function GraphOpeningPanel({ state, connected, onAction, onClose, compact = false, retained = false, standalone = false }: {
  state: GraphOpeningState; connected: boolean; compact?: boolean; retained?: boolean; standalone?: boolean; onAction(action: GraphSetupAction): void; onClose?(): void;
}) {
  const panelRef = React.useRef<HTMLElement>(null);
  const formId = React.useId();
  const previousRect = React.useRef<DOMRect | null>(null);
  React.useLayoutEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const rect = panel.getBoundingClientRect(), previous = previousRect.current;
    if (previous && panel.animate && !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      panel.animate([{ transform: `translate(${previous.x - rect.x}px, ${previous.y - rect.y}px)` }, { transform: 'translate(0, 0)' }],
        { duration: 280, easing: 'cubic-bezier(.2,.8,.2,1)' });
    }
    previousRect.current = rect;
  }, [compact, retained, state.phase]);
  const [draft, setDraft] = React.useState(() => initialGraphDraft(state));
  const directSetup = React.useRef({ init: (draft.profile as any).init, configObject: draft.configObject });
  const customFactory = React.useRef((draft.profile as any).factory);
  const directInputs = React.useRef((draft.profile as any).inputs);
  const customInputs = React.useRef((draft.profile as any).inputs?.factory);
  const [manualSetup, setManualSetup] = React.useState(false);
  React.useEffect(() => setManualSetup(false), [state.id]);
  const [error, setError] = React.useState('');
  const [emptyCallConfirmed, setEmptyCallConfirmed] = React.useState(false);
  const incomingDraft = JSON.stringify(initialGraphDraft(state));
  React.useEffect(() => {
    const next = JSON.parse(incomingDraft);
    setDraft(next); setError(''); setEmptyCallConfirmed(false);
    directSetup.current = { init: next.profile.init, configObject: next.configObject };
    customFactory.current = next.profile.factory;
    directInputs.current = next.profile.inputs?.factory ? emptyBinding() : next.profile.inputs;
    customInputs.current = next.profile.inputs?.factory;
  }, [state.id, incomingDraft]);
  const discovery = state.discovery, profile = draft.profile as any;
  const initialProfile = state.draft?.profile as any;
  const initFields = discovery?.init?.filter(f => !['varargs', 'kwargs'].includes(f.kind) && f.name !== draft.configObject?.parameter) ?? [];
  // The saved recipe provides editable slots, not proof of an unknown signature.
  // Keep them from the incoming draft so clearing a field does not remove its control.
  const inputFields = inputFieldsForBinding(discovery?.inputs ?? [], initialProfile?.inputs?.factory ? emptyBinding() : initialProfile?.inputs ?? emptyBinding());
  const unresolvedInputs = !!discovery && (discovery.inputs === null
    || (discovery.inputs.some(f => ['varargs', 'kwargs'].includes(f.kind)) && !inputFields.some(f => f.required)));
  const hasInputs = !!(profile.inputs?.factory || profile.inputs?.args?.length || Object.keys(profile.inputs?.kwargs ?? {}).length);
  const needsExplicitCall = unresolvedInputs && !hasInputs && !emptyCallConfirmed;
  const needsAdvancedInputs = unresolvedInputs && !hasInputs;
  const needsAdvancedInit = !initialProfile?.factory && discovery?.init === null && !draft.profileName;
  // Use the incoming draft so typing a missing value never collapses its field.
  const needsValue = (key: 'init' | 'inputs', field: GraphSetupField) => !!state.requirements?.some(reason => reason.startsWith(`${key}.${field.name}:`));
  const supplied = (key: 'init' | 'inputs', field: GraphSetupField) => !needsValue(key, field)
    && !(key === 'inputs' && initialProfile?.inputs?.factory)
    && boundValue(initialProfile?.[key] ?? emptyBinding(), field, key === 'inputs' ? inputFields : discovery?.init ?? []) !== undefined;
  const busy = ['resolving', 'saving', 'capturing'].includes(state.phase);
  const showForm = !!discovery && (state.phase === 'needs_setup' || (state.phase === 'failed' && state.failure?.kind === 'setup'));
  const focusedOpening = React.useRef('');
  React.useEffect(() => {
    if (!showForm || compact || !connected) return;
    const key = `${state.id}:${state.phase}:${manualSetup}`;
    if (focusedOpening.current === key) return;
    const fields = Array.from(panelRef.current?.querySelectorAll<HTMLInputElement>('form input[required]') ?? []);
    const target = manualSetup ? fields.find(input => !input.value.trim())
      ?? panelRef.current?.querySelector<HTMLElement>('form select')
      : panelRef.current?.querySelector<HTMLButtonElement>('[data-setup-agent]');
    if (!target) return;
    focusedOpening.current = key;
    let parent = target.parentElement;
    while (parent && parent !== panelRef.current) { if (parent instanceof HTMLDetailsElement) parent.open = true; parent = parent.parentElement; }
    target.focus({ preventScroll: true });
    if (manualSetup) target.scrollIntoView?.({ block: 'nearest' });
  }, [showForm, compact, connected, state.id, state.phase, state.environment?.reason, manualSetup]);
  const update = (changes: Record<string, unknown>) => setDraft(old => ({ ...old,
    ...('init' in changes || 'factory' in changes ? { configDefaults: undefined } : {}),
    profile: { ...old.profile, ...changes } }));
  const setBinding = (key: 'init' | 'inputs', field: GraphSetupField, value: any) => {
    try { update({ [key]: bind(profile[key] ?? emptyBinding(), field, key === 'inputs' ? inputFields : discovery?.init ?? [], value) }); setError(''); }
    catch (e) { setError((e as Error).message); }
  };
  const fieldSource = (field: GraphSetupField, binding?: 'init' | 'inputs') => {
    const evidence = binding ? state.evidence?.filter(item => item.field === `${binding}.${field.name}`
      || item.field.startsWith(`${binding}.${field.name}.`)) : undefined;
    return <div className="graph-setup-source">
      {field.source ? <button type="button" disabled={!connected || busy} title={`Open declaration of ${field.name}`}
        onClick={() => onAction({ kind: 'reveal_setup_source', ...field.source! })}><ArrowUpRight size={12} aria-hidden="true" /><span>{field.source.file}:{field.source.line}</span></button>
        : <span>Declaration not available</span>}
      {field.defaultText && <span>Code default: <code>{field.defaultText}</code></span>}
      {!!evidence?.length && <details><summary>Initial value source</summary>{evidence.map((item, index) => <div key={index}>
        <p>{item.detail}</p>{item.file && item.line ? <button type="button" disabled={!connected || busy}
          onClick={() => onAction({ kind: 'reveal_setup_source', file: item.file!, line: item.line! })}>{item.file}:{item.line}</button> : null}
      </div>)}</details>}
    </div>;
  };
  const button = (kind: GraphSetupAction['kind'], label: string) => <button type="button" disabled={!connected || busy}
    onClick={() => onAction({ kind } as GraphSetupAction)}>{label}</button>;
  const summaryInputs = profile.inputs?.factory ? [`Inputs: ${profile.inputs.factory.callable}()`] : [...(profile.inputs?.args ?? []).map((value: any, index: number) => [String(index + 1), value]),
    ...Object.entries(profile.inputs?.kwargs ?? {})].map(([name, value]: [string, any]) =>
      value?.kind === 'tensor' ? `${name}: ${value.shape?.length ? value.shape.join(' × ') : 'scalar'} · ${value.dtype} · synthetic` : `${name}: ${value?.kind ?? typeof value}`);
  // Keep the opening's order stable while typing; constructor edits still precede dependent inputs.
  const inputsFirst = !profile.factory && !draft.configObject
    && initFields.every(field => !field.required || supplied('init', field));
  const modelSection = discovery && <section key="model" className="graph-setup-section" aria-label="Model configuration"><h2>Create the model</h2>
          <label>Model creation<select aria-label="Model creation" value={profile.factory ? 'python' : 'settings'} onChange={e => {
            if (e.target.value === 'python') {
              directSetup.current = { init: profile.init, configObject: draft.configObject };
              setDraft(old => ({ ...old, configDefaults: undefined, configObject: undefined, profile: { ...old.profile, init: undefined,
                factory: customFactory.current ?? { file: '', callable: '', kwargs: {} } } }));
            } else {
              customFactory.current = profile.factory;
              setDraft(old => ({ ...old, configDefaults: undefined, configObject: directSetup.current.configObject, profile: { ...old.profile,
                factory: undefined, init: directSetup.current.init ?? emptyBinding() } }));
            }
          }}><option value="settings">Class settings</option><option value="python">Python function</option></select></label>
          {!profile.factory && <p className="graph-setup-hint">Sets how {state.target.cls} is created.</p>}
          {profile.factory && <>
            {discovery.factories.length > 0 && <label>Model builder<select value={String(discovery.factories.findIndex(f =>
              f.file === profile.factory.file && f.callable === profile.factory.callable))} onChange={e => {
              const f = discovery.factories[Number(e.target.value)]; if (f) update({ factory: { file: f.file, callable: f.callable, kwargs: {} } });
            }}><option value="-1">Choose a model builder…</option>{discovery.factories.map((f, i) => <option key={i} value={i}>{f.file}:{f.callable}</option>)}</select></label>}
            <label>Model builder file<input required value={profile.factory.file} onChange={e => update({ factory: { ...profile.factory, file: e.target.value } })} placeholder="model_setup.py" /></label>
            <label>Model builder function<input required value={profile.factory.callable} onChange={e => update({ factory: { ...profile.factory, callable: e.target.value } })} placeholder="make_model" /></label>
            <p className="graph-setup-hint">Calls a project function that returns {state.target.cls}. Use this to load trained weights or build a model that needs more than simple settings.</p>
            {!standalone && <button type="button" onClick={() => onAction({ kind: 'factory_template', draft })}>Create starter function</button>}
          </>}
        {!profile.factory && <>
          {draft.configObject && discovery.configs[draft.configObject.index] && <div className="graph-setup-input">
            <SetupFields fields={discovery.configs[draft.configObject.index].fields ?? []} label="Configuration options" source={f => fieldSource(f)}
              resolved={f => state.draft?.configObject?.values[f.name] !== undefined}
              value={f => draft.configObject!.values[f.name]} renderField={f => <LiteralField key={f.name} label={`${f.name}${f.required ? ' *' : ''}`} required={f.required}
              placeholder={f.defaultText ? `Default: ${f.defaultText}` : undefined}
              value={draft.configObject!.values[f.name] === undefined ? undefined : argumentFromJson(draft.configObject!.values[f.name])}
              onChange={v => { const values = { ...draft.configObject!.values }; if (v === undefined) delete values[f.name]; else values[f.name] = jsonFromArgument(v);
                setDraft({ ...draft, configObject: { ...draft.configObject!, values } }); }} />} /></div>}
          <SetupFields fields={initFields.map(f => needsValue('init', f) ? { ...f, required: true } : f)} label="Model settings" source={f => fieldSource(f, 'init')} value={f => boundValue(profile.init ?? emptyBinding(), f, discovery.init!)}
            resolved={f => supplied('init', f)}
            renderField={f => <LiteralField key={f.name} required={f.required}
              label={`${f.name}${f.required ? ' *' : ''}`} placeholder={f.defaultText ? `Default: ${f.defaultText}` : undefined}
              value={boundValue(profile.init ?? emptyBinding(), f, discovery.init!)} onChange={v => setBinding('init', f, v)} />} />
          {draft.configDefaults && <p>Uses declared configuration defaults. Edit the saved Python helper to change them.</p>}
          {discovery.init === null && !draft.configDefaults && <p>{draft.profileName
            ? 'Model settings are saved in this profile. Use Advanced setup to change them.'
            : 'Choose Python function above when your project has a model builder. Explicit bindings are also available in Advanced setup.'}</p>}
        </>}
        </section>;
  const inputSection = discovery && <section key="inputs" className="graph-setup-section" aria-label="Test data"><h2>Test data</h2>
        <p className="graph-setup-hint">A sample passed to {profile.framework === 'jax' ? 'your callable to trace' : 'forward() to capture'} operations and tensor shapes.</p>
        <label>Input source<select aria-label="Input source" value={profile.inputs?.factory ? 'python' : 'entered'} onChange={e => {
          if (e.target.value === 'python') {
            directInputs.current = profile.inputs;
            update({ inputs: { factory: customInputs.current ?? { file: state.target.file, callable: '', kwargs: {} } } });
          } else {
            customInputs.current = profile.inputs.factory;
            update({ inputs: directInputs.current?.factory ? emptyBinding() : directInputs.current ?? emptyBinding() });
          }
        }}><option value="entered">Enter values / generate tensors</option><option value="python">Python function</option></select></label>
        {!profile.inputs?.factory && <p className="graph-setup-hint">Set shapes below to generate sample tensors, or choose a literal value. These are test inputs, not your dataset.</p>}
        {profile.inputs?.factory ? <div className="graph-setup-input">
          <label>Input function file<input required value={profile.inputs.factory.file} onChange={e => update({ inputs: { factory: { ...profile.inputs.factory, file: e.target.value } } })} placeholder="model.py" /></label>
          <label>Input function<input required value={profile.inputs.factory.callable} onChange={e => update({ inputs: { factory: { ...profile.inputs.factory, callable: e.target.value } } })} placeholder="graph_inputs" /></label>
          <p className="graph-setup-hint">Use this for real data, tokenizers or structured inputs that simple tensors cannot represent. Calls your function before capture.</p>
          <details><summary>Function return format</summary><p>Return a dictionary of input names and values, or an explicit args/kwargs binding. Function options are in Advanced setup.</p></details>
        </div> : <SetupFields fields={inputFields} label="Input settings" source={f => fieldSource(f, 'inputs')} value={f => boundValue(profile.inputs ?? emptyBinding(), f, inputFields)}
          expanded={retained || state.phase === 'failed' || (unresolvedInputs && hasInputs)}
          resolved={f => supplied('inputs', f)}
          renderField={f => <InputField key={f.name} field={f}
            value={boundValue(profile.inputs ?? emptyBinding(), f, inputFields)} onChange={v => setBinding('inputs', f, v)} />} />}
        {discovery.inputs?.length === 0 && !hasInputs && <p>This forward has no declared inputs.</p>}
        {needsAdvancedInputs && <div className="graph-setup-unresolved"><p>This forward has dynamic or unavailable arguments. Set its explicit inputs in Advanced setup.</p>
          {!hasInputs && <label className="graph-setup-checkbox"><input type="checkbox" checked={emptyCallConfirmed}
            onChange={e => setEmptyCallConfirmed(e.target.checked)} />Run this call with no arguments</label>}</div>}
        {discovery.constraints.map((c, i) => <p className="graph-setup-hint" key={i}>{c.input}: axis {c.axis} must match {c.size ?? c.initParameter} · {c.file}:{c.line}</p>)}
        </section>;
  if (state.environment && state.environment.reason !== 'ready' && (state.phase !== 'cancelled' || !state.cancellationReason)) return <EnvironmentRecoveryPanel
    state={state} connected={connected} standalone={standalone} onAction={onAction} onClose={onClose} />;
  const environment = state.environment?.candidates.find(candidate => candidate.id === state.environment?.selectedId);
  const setupHelp = <section className="graph-setup-help" aria-label="Setup help and environment">
    {!busy && (showForm && manualSetup ? <div className="graph-setup-manual-heading">
      <button type="button" onClick={() => setManualSetup(false)}><ArrowLeft size={14} aria-hidden="true" /><span>Setup methods</span></button>
      <div><strong>Manual setup</strong><span>Required fields *</span></div>
    </div> : <section className="graph-setup-methods" aria-label="Choose a setup method">
      {showForm && <h2>Choose a setup method</h2>}
      <button type="button" data-setup-agent className="graph-setup-agent" disabled={!connected}
        aria-label={standalone ? 'Copy setup request' : 'Set up with agent'} aria-describedby={`${formId}-agent-help`}
        onClick={() => onAction({ kind: 'agent_help', ...(showForm ? { draft } : {}) })}>
        <Sparkles size={16} aria-hidden="true" /><span><span className="graph-setup-method-label">{standalone ? 'Copy setup request' : 'Set up with agent'}</span><small>Recommended</small></span><ArrowUpRight size={16} aria-hidden="true" />
      </button>
      <p id={`${formId}-agent-help`}>{standalone ? 'Copy your model context and edits to share with your coding agent.' : 'Opens chat to suggest settings and test inputs from your code.'}</p>
      {showForm && <>
        <div className="graph-setup-or"><span>or</span></div>
        <button type="button" className="graph-setup-manual-choice" disabled={!connected}
          aria-controls={formId} aria-describedby={`${formId}-manual-help`} onClick={() => setManualSetup(true)}>
          <SlidersHorizontal size={16} aria-hidden="true" /><span>Set up manually</span><ArrowRight size={16} aria-hidden="true" />
        </button>
        <p id={`${formId}-manual-help`}>Choose model settings and test inputs yourself.</p>
      </>}
    </section>)}
    <div className="graph-setup-environment">
      <div className="graph-setup-guidance-heading"><strong>Python environment</strong>
        {!standalone && button('environment', 'Change')}</div>
      {environment ? <><div className="graph-setup-environment-values"><span>{state.environmentName ?? environment.prefix?.split(/[/\\]/).filter(Boolean).pop() ?? 'Selected Python'}</span>
        <code>Python {environment.python ?? 'unknown'}{environment.torch ? ` · Torch ${environment.torch}` : ''}</code></div>
        <details><summary>Interpreter path</summary><code>{environment.interpreter}</code></details></>
        : <p className="graph-setup-hint">Details will be checked before execution.</p>}
    </div>
  </section>;
  return <section ref={panelRef} className={`graph-opening-panel${compact ? ' graph-opening-panel-compact' : ' graph-model-setup-panel'}`} aria-label="Open captured graph" aria-busy={busy}
    onKeyDown={event => { if (event.key === 'Escape' && onClose) { event.stopPropagation(); onClose(); } }}>
    {!compact && onClose && <button type="button" className="graph-opening-close" aria-label="Close inputs" title="Close inputs (Escape)" onClick={onClose}><X size={16} aria-hidden="true" /></button>}
    {compact ? <>
      <button className="graph-run-summary" type="button" disabled={!connected} onClick={() => onAction({ kind: 'change_inputs' })} aria-label="Change inputs">
        <span className="graph-run-summary-main"><strong>{state.phase === 'failed' ? 'Partial capture' : profile.framework === 'jax' ? 'JAX traced graph' : 'Captured graph'}</strong>
          <span>{!state.draft ? 'Setup details unavailable' : summaryInputs.length ? summaryInputs.slice(0, 2).join(' · ') + (summaryInputs.length > 2 ? ` · +${summaryInputs.length - 2} inputs` : '')
            : profile.kind === 'script' ? 'Script invocation' : 'No forward arguments'}</span>
          <small>{[profile.device, profile.mode === 'train' ? 'Training' : profile.mode === 'eval' ? 'Evaluation' : '', draft.profileName].filter(Boolean).join(' · ')}</small>
        </span><span className="graph-run-edit">Edit inputs</span>
      </button>
      <SetupEvidence state={state} />
      {state.phase === 'failed' && <p role="alert">{state.detail}</p>}
    </> : <>
    <header className="graph-setup-header">
    <div className="graph-opening-heading"><span className={busy ? 'graph-opening-indicator running' : 'graph-opening-indicator'} />
      <div><div className="graph-opening-model"><strong>{state.target.cls}</strong><code>{state.target.file}</code></div></div></div>
        <h1>{busy ? state.phase === 'capturing' ? profile.framework === 'jax' ? 'Tracing the saved call' : 'Capturing one forward call' : state.phase === 'saving' ? 'Saving this setup' : 'Preparing your graph'
          : state.phase === 'needs_setup' ? 'Set up this model'
            : state.phase === 'failed' ? 'Capture needs attention' : state.phase === 'ready' ? 'Waiting for the captured graph' : 'Opening cancelled'}</h1>
        {state.phase !== 'needs_setup' && <p role="status">{state.detail}</p>}
    </header>
    <div className="graph-setup-body">
    {retained && <p className="graph-previous-capture" role="status">{state.phase === 'failed' && state.hasCapture
      ? 'Partial capture remains visible up to the failure.' : 'Previous capture stays visible until you run again.'}</p>}
    {state.requirements?.length ? <ul role="alert">{state.requirements.map((r, i) => <li key={i}>{r}</li>)}</ul> : null}
    {!connected && <p role="status">Reconnecting to {standalone ? 'TensorViz' : 'VS Code'}. Your setup stays here.</p>}
    {setupHelp}
    {showForm && discovery && <form key={state.id} id={formId} hidden={!manualSetup} onInvalidCapture={e => {
      let parent = (e.target as HTMLElement).parentElement;
      while (parent && parent !== e.currentTarget) { if (parent instanceof HTMLDetailsElement) parent.open = true; parent = parent.parentElement; }
    }} onSubmit={e => {
      e.preventDefault(); if (e.currentTarget.reportValidity() && !error && !needsExplicitCall) onAction({ kind: 'submit', draft });
    }}>
      <fieldset disabled={!connected || busy}><legend className="graph-setup-sr-only">Model settings and test data</legend>
        {discovery.examples.length > 0 && <label>Use an example from your code<select defaultValue="" onChange={e => {
          if (!e.target.value) return;
          const example = discovery.examples[Number(e.target.value)];
          setDraft(old => ({ ...old, configDefaults: undefined, configObject: undefined, profile: { ...old.profile, factory: undefined,
            init: example.init, inputs: example.inputs ?? emptyBinding() } }));
        }}><option value="">Choose an example…</option>{discovery.examples.map((ex, i) => <option key={i} value={i}>{ex.file}:{ex.line}{ex.inputs ? ' · model and inputs' : ' · constructor only'}</option>)}</select></label>}
        {inputsFirst ? [inputSection, modelSection] : [modelSection, inputSection]}
        {profile.framework === 'jax' ? <p className="graph-setup-hint">JAX tracing infers tensor shapes. Flax initialization uses the saved seed.</p> : null}
        <details className="graph-setup-options graph-setup-run-options"><summary><span>Run settings</span><small>{profile.device} · {profile.mode === 'train' ? 'Training' : 'Evaluation'} · {profile.grad_enabled ? 'Gradients enabled' : 'No gradients'}</small></summary>
        <div className="graph-setup-row"><label>Mode<select disabled={profile.framework === 'jax'} value={profile.mode} onChange={e => update({ mode: e.target.value })}><option value="eval">Evaluation</option><option value="train">Training</option></select></label>
          <label>Device<input required value={profile.device} onChange={e => update({ device: e.target.value })} placeholder={profile.framework === 'jax' ? 'cpu or cuda:0' : 'cpu, mps or cuda:0'} /></label>
          <label className="graph-setup-checkbox"><input type="checkbox" disabled={profile.framework === 'jax'} checked={profile.grad_enabled} onChange={e => update({ grad_enabled: e.target.checked })} />Record gradients</label></div>
        </details>
        <details className="graph-setup-options" open={needsAdvancedInputs || needsAdvancedInit
          || (initialProfile?.factory && (!initialProfile.factory.file || !initialProfile.factory.callable))
          || (state.draft?.configObject && !state.draft.configObject.parameter) ? true : undefined}><summary>Advanced setup</summary>
          {!profile.factory && <>
            {discovery.configs.some(config => config.fields) && <label>Configuration type<select value={draft.configObject ? String(draft.configObject.index) : ''} onChange={e => {
              const index = Number(e.target.value);
              setDraft(old => ({ ...old, configDefaults: undefined, configObject: e.target.value === '' ? undefined
                : { index, parameter: discovery.configs[index].parameter ?? '', values: {} } }));
            }}><option value="">Use class parameters</option>
              {discovery.configs.map((config, i) => config.fields && <option key={i} value={i}>{config.cls} · {config.file}</option>)}
            </select></label>}
            {draft.configObject && <label>Model parameter for this config<input required value={draft.configObject.parameter}
              onChange={e => setDraft({ ...draft, configObject: { ...draft.configObject!, parameter: e.target.value } })} /></label>}
          </>}
          <details open={needsAdvancedInputs || needsAdvancedInit ? true : undefined}><summary>Explicit bindings and limits</summary>
          <p>Edit the explicit profile as JSON. Bindings use args and kwargs; tensor values have kind, shape, dtype, requires_grad and generator. No expressions are evaluated.</p>
          <textarea aria-label="Advanced profile" key={JSON.stringify(profile)} defaultValue={JSON.stringify(profile, null, 2)} rows={14}
            onBlur={e => { try { const next = JSON.parse(e.target.value); if (!next || Array.isArray(next) || typeof next !== 'object') throw new Error(); setDraft(old => ({ ...old, profile: next,
                configDefaults: JSON.stringify(next.init) === JSON.stringify(old.profile.init) && !next.factory ? old.configDefaults : undefined })); setError(''); }
              catch { setError('Advanced setup must be a JSON object.'); } }} />
          </details>
        </details>
        {error && <p role="alert">{error}</p>}
      </fieldset>
    </form>}
    {!busy && <details className="graph-setup-options"><summary>Saved setup and source evidence</summary>
      <div className="graph-setup-actions">{button('edit_config', standalone ? 'Copy config path' : 'Edit tensorviz.toml')}</div>
      {draft.profileName && <p className="graph-setup-hint">Profile: <strong>{draft.profileName}</strong></p>}
      <SetupEvidence state={state} />
    </details>}
    {!busy && !showForm && <div className="graph-setup-actions">{button('change_inputs', 'Change inputs')}{button('retry', 'Retry')}</div>}
    {!busy && <details className="graph-setup-recovery"><summary>More options</summary><div className="graph-setup-actions">
      {!standalone && button('save_sources', 'Save & Open')}
      {button('source_only', 'Inspect source only')}
    </div></details>}
    {busy && <button type="button" disabled={!connected} onClick={() => onAction({ kind: 'cancel' })}>Cancel</button>}
    {!standalone && !busy && discovery?.parents.length ? <details className="graph-setup-parents"><summary>Use an enclosing model’s inputs</summary><p>The enclosing model supplies this block’s inputs. Its captured graph will focus this block.</p>
      {discovery.parents.map((p, i) => <button type="button" key={i} disabled={!connected} onClick={() => onAction({ kind: 'parent', index: i })}>{p.cls} → {p.attribute} · {p.file}:{p.line}</button>)}</details> : null}
    {discovery?.truncated && <p>Discovery reached its size limit. Additional setup may exist in your repository.</p>}
    </div>
    {showForm && manualSetup && <footer className="graph-setup-submit">
      <button className="graph-setup-primary" type="submit" form={formId} disabled={!connected || busy || !!error || needsExplicitCall}>Open graph</button>
      <p className="graph-setup-hint">Saves to tensorviz.toml, then runs the model to capture the graph.</p>
    </footer>}
    </>}
  </section>;
}
