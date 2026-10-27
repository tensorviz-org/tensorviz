import React from 'react';
import { layerTypes } from '../data/layerTypes';
import { ModelNode } from '../types';
import { LocalBrowserContext } from './localBrowserContext';
import { LocalBrowserSourceEdit, SourceEditValue } from './protocol.generated';
import { SourceEditOutcome } from './localBrowserSourceEditClient';
import './SourceEditForm.css';
import { CapturedSourceEditForm } from './CapturedSourceEditForm';

const addable = layerTypes.filter(layer => ['linear', 'conv2d', 'relu', 'gelu', 'dropout', 'batchnorm2d', 'layernorm'].includes(layer.id));
function literal(text: string): SourceEditValue {
  try { return JSON.parse(text); } catch { return text; }
}

export function SourceEditForm({ nodes, compact = false }: { nodes: ModelNode[]; compact?: boolean }) {
  return nodes.some(node => !!node.data.captureRef)
    ? <CapturedSourceEditForm nodes={nodes} compact={compact} /> : <DeclaredSourceEditForm nodes={nodes} />;
}

function DeclaredSourceEditForm({ nodes }: { nodes: ModelNode[] }) {
  const context = React.useContext(LocalBrowserContext);
  const [operation, setOperation] = React.useState<'set_param' | 'add_layer' | 'remove_layer' | 'group_layers'>(nodes.length > 1 ? 'group_layers' : 'set_param');
  const [param, setParam] = React.useState('');
  const [value, setValue] = React.useState('');
  const [name, setName] = React.useState('');
  const [layerId, setLayerId] = React.useState('relu');
  const [parameters, setParameters] = React.useState<Record<string, string>>({});
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<SourceEditOutcome | null>(null);
  React.useEffect(() => () => context?.sourceEdits?.cancel(), [context?.sourceEdits]);
  const layer = addable.find(item => item.id === layerId)!;
  const parameterNames = Object.keys(nodes[0]?.data.parameters ?? {}).filter(key => !key.startsWith('_') && key !== 'name');
  const selectionKey = nodes.map(node => node.id).join('\n');
  React.useEffect(() => {
    const values = nodes[0]?.data.parameters ?? {};
    const first = Object.keys(values).find(key => !key.startsWith('_') && key !== 'name') ?? '';
    setParam(first); setValue(first ? JSON.stringify(values[first]) : '');
    if (nodes.length > 1) setOperation('group_layers');
    // Selection identity, rather than each canvas render, resets the form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionKey]);
  React.useEffect(() => {
    setParameters(Object.fromEntries(layer.parameters.filter(p => p.name !== 'name').map(p => [p.name, JSON.stringify(p.defaultValue ?? null)])));
  }, [layer]);
  const connected = context?.status?.phase === 'connected' && context.status.hostLink === 'connected';
  const ready = !!context?.sourceEdits && connected && context.page.syncState !== 'stale' && nodes.length > 0 && !pending;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready || !context?.sourceEdits) return;
    let edit: LocalBrowserSourceEdit;
    if (operation === 'group_layers') edit = { kind: operation, nodeIds: nodes.map(node => node.id), name };
    else if (operation === 'remove_layer') edit = { kind: operation, nodeId: nodes[0].id };
    else if (operation === 'add_layer') edit = { kind: operation, afterNodeId: nodes[0].id, nodeId: name, layerType: layer.name,
      params: Object.fromEntries(Object.entries(parameters).map(([key, text]) => [key, literal(text)])) };
    else edit = { kind: operation, nodeId: nodes[0].id, param, value: literal(value) };
    setResult(null); setPending(true);
    void context.sourceEdits.propose(edit).promise.then(outcome => { setResult(outcome); setPending(false); });
  };
  return <section className="source-edit-form" aria-label="Source edit proposal">
    <p>Select layers on the canvas, propose a change, then review and apply its Python diff in VS Code.</p>
    <p>Editing a shared class definition affects every instance of that class.</p>
    <p className="source-edit-selection">{nodes.length ? nodes.map(node => node.id).join(', ') : 'Select a layer to begin.'}</p>
    <form onSubmit={submit}>
      <fieldset disabled={!ready}>
        <label>Change<select aria-label="Source operation" value={operation} onChange={event => setOperation(event.target.value as typeof operation)}>
          <option value="set_param" disabled={nodes.length !== 1}>Edit parameter</option>
          <option value="add_layer" disabled={nodes.length !== 1}>Add layer after selection</option>
          <option value="remove_layer" disabled={nodes.length !== 1}>Delete layer</option>
          <option value="group_layers" disabled={nodes.length < 2}>Group selection as a block</option>
        </select></label>
        {operation === 'set_param' && <>
          <label>Parameter<select aria-label="Parameter" value={param} onChange={event => { setParam(event.target.value); setValue(JSON.stringify(nodes[0]?.data.parameters?.[event.target.value])); }}>
            {parameterNames.map(key => <option key={key}>{key}</option>)}
          </select></label>
          <label>Value<input aria-label="Parameter value" value={value} onChange={event => setValue(event.target.value)} required /></label>
          <small>Use a number, true/false, text, or an array such as [3, 3].</small>
        </>}
        {(operation === 'group_layers' || operation === 'add_layer') && <label>{operation === 'group_layers' ? 'Block name' : 'Layer name'}
          <input aria-label="New module name" value={name} onChange={event => setName(event.target.value)} placeholder={operation === 'group_layers' ? 'features' : 'activation'} pattern="[A-Za-z_][A-Za-z0-9_]*" maxLength={64} required />
        </label>}
        {operation === 'group_layers' && <p>Grouping changes parameter paths and checkpoint keys. Consecutive layers with literal constructor arguments are supported; ambiguous chains are refused.</p>}
        {operation === 'remove_layer' && <p>The proposal removes the layer and passes its input onward. Review the resulting model behavior in the diff.</p>}
        {operation === 'add_layer' && <>
          <label>Layer type<select aria-label="New layer type" value={layerId} onChange={event => setLayerId(event.target.value)}>
            {addable.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}
          </select></label>
          {Object.entries(parameters).map(([key, text]) => <label key={key}>{key}<input aria-label={`New layer ${key}`} value={text} onChange={event => setParameters(values => ({ ...values, [key]: event.target.value }))} /></label>)}
        </>}
        <button type="submit" disabled={!ready || (operation !== 'group_layers' && nodes.length !== 1) || (operation === 'group_layers' && nodes.length < 2) || (operation === 'set_param' && !param)}>Review in VS Code</button>
      </fieldset>
    </form>
    {pending && <div role="status"><p>Preparing the source diff. Continue the review in VS Code.</p><button onClick={() => context?.sourceEdits?.cancel()}>Cancel source proposal</button></div>}
    {result && <p role="status" data-status={result.status}>{result.detail}</p>}
    {!connected && <p role="status">Source proposals pause until VS Code reconnects.</p>}
  </section>;
}
