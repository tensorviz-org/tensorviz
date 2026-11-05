import React from 'react';
import './localSourceEditor.css';
import { ModelNodeData } from '../types';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { addSourceEdit, constructorParameters, DIRECT_LAYERS, hasDeclarationEdit, insertUnarySourceEdit, parameterSourceEdit, removeSourceEdit, sourceEditLimitation } from '../ide/directSourceEdits';
import { parseParameterInput } from '../components/ui2/parameterValues';
import { LocalBrowserSourceEdit, SourceEditValue } from '../ide/protocol.generated';

/** All actions prepare a host-owned draft. No canvas graph mutation occurs here. */
export function LocalSourceEditor({ nodeId, data, structuralOnly = false, insertion }: {
  nodeId: string; data: ModelNodeData; structuralOnly?: boolean;
  insertion?: { nodeId: string; data: ModelNodeData; capabilityId: string };
}) {
  const local = React.useContext(LocalBrowserContext);
  const [detail, setDetail] = React.useState('Changes open a source review. Accept to write and refresh.');
  const [busy, setBusy] = React.useState(false);
  const [layer, setLayer] = React.useState(insertion ? 'relu' : 'dropout');
  const [name, setName] = React.useState('new_layer');
  const [params, setParams] = React.useState('{"p":0.5}');
  const propose = async (prepare: () => LocalBrowserSourceEdit) => {
    if (!local?.sourceEdits) { setDetail('Connect to VS Code to review source edits.'); return; }
    try {
      const edit = prepare();
      setBusy(true); setDetail('Review the source draft in VS Code or use Canvas Accept.');
      const result = await local.sourceEdits.propose(edit).promise;
      setDetail(result.detail);
    } catch (error) { setDetail(error instanceof Error ? error.message : 'The source edit could not be prepared.'); }
    finally { setBusy(false); }
  };
  const parameters = constructorParameters(data);
  const declaration = hasDeclarationEdit(data);
  const deletable = declaration || data.sourceEditCapabilities?.some(cap => cap.command === 'delete_unary_op');
  const disabled = busy || !!local?.reviewing || !!local?.page.greyed;
  return <section className="local-source-editor" aria-label="Edit model source" onKeyDown={event => event.stopPropagation()}>
    {!structuralOnly && parameters.map(param => <label key={param}>{param}
      <input aria-label={param} defaultValue={String(data.parameters[param] ?? '')} disabled={disabled}
        onBlur={event => {
          const value = parseParameterInput(event.target.value) as SourceEditValue;
          if (JSON.stringify(value) !== JSON.stringify(data.parameters[param])) void propose(() => parameterSourceEdit(nodeId, data, param, value));
        }} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
    </label>)}
    {!structuralOnly && !parameters.length && <p>{sourceEditLimitation(data)}</p>}
    <details><summary>{insertion ? "Blocks · insert on this operand" : "Blocks · insert after selected layer"}</summary>
      {declaration || insertion ? <>
        <select aria-label="Layer to insert" value={layer} disabled={disabled} onChange={event => {
          setLayer(event.target.value);
          setParams(event.target.value === 'linear' ? '{"in_features":8,"out_features":8}' : event.target.value === 'layernorm' ? '{"normalized_shape":[8]}' : event.target.value === 'dropout' ? '{"p":0.5}' : '{}');
        }}>{(declaration ? DIRECT_LAYERS : ['relu', 'gelu', 'silu', 'sigmoid', 'tanh']).map(id => <option key={id}>{id}</option>)}</select>
        {!(insertion && ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'].includes(layer)) && <><label>Attribute name<input aria-label="New layer name" value={name} disabled={disabled} onChange={event => setName(event.target.value)} /></label>
        <label>Constructor params (JSON)<input aria-label="New layer parameters" value={params} disabled={disabled} onChange={event => setParams(event.target.value)} /></label></>}
        <button disabled={disabled} onClick={() => void propose(() => {
          if (insertion && ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'].includes(layer)) return insertUnarySourceEdit(insertion.nodeId, insertion.data, insertion.capabilityId, layer);
          const values = JSON.parse(params);
          if (!values || Array.isArray(values) || typeof values !== 'object') throw new Error('Enter a JSON object of constructor parameters.');
          return addSourceEdit(nodeId, data, layer, name, values);
        })}>Review insertion</button>
      </> : <p>{sourceEditLimitation(data)}</p>}
    </details>
    <button disabled={disabled || !deletable} onClick={() => void propose(() => removeSourceEdit(nodeId, data))}>Review deletion</button>
    <p role="status">{detail}</p>
  </section>;
}
