import React from 'react';
import type { ModelNode } from '../types';
import { LocalBrowserContext } from './localBrowserContext';
import { ACTIVATION_NAMES, isActivationName, isSourceNumber, type SourceNumber,
  type ReplaceActivationEdit, type LocalBrowserSourceEdit } from './protocol.generated';
import type { SourceEditOutcome } from './localBrowserSourceEditClient';
import { CandidateReviewPanel } from './CandidateReviewPanel';

export function parseSourceNumber(text: string, kind: SourceNumber['kind']): SourceNumber | null {
  const value = text.trim();
  if (!value || value.length > 4096) return null;
  if (kind === 'int') {
    const literal = { kind, value };
    return isSourceNumber(literal) ? literal : null;
  }
  if (!/^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?![\s\S])/.test(value)) return null;
  const number = Number(value);
  const literal = { kind, value: Object.is(number, -0) ? '-0' : number };
  return isSourceNumber(literal) ? literal : null;
}

const limitation: Record<string, string> = {
  derived_operation: 'This event cannot be edited independently of its authored expression.',
  ambiguous_binding: 'The captured event has no unique authored expression.',
  unsupported_expression: 'This source expression needs a recipe that is not supported yet.',
  unsupported_activation_expression: 'This functional activation needs a known input and literal options. Expanded or computed options need a source edit in VS Code.',
  ambiguous_constructor: 'This module has no unique verified constructor location. Edit its source directly in VS Code.',
  ambiguous_constructor_arguments: 'Expanded constructor arguments need a source edit in VS Code.',
  unproven_constructor: 'This constructor cannot yet be verified for a numeric argument edit.',
  no_constructor_literals: 'This constructor has no supported numeric literal or verified configuration definition. Edit its source directly in VS Code.',
  unsupported_activation_constructor: 'This activation constructor needs supported literal options. Expanded or computed options need a source edit in VS Code.',
  unproven_activation_import: 'The Torch namespace is missing, shadowed or rebound. Edit this call directly in VS Code.',
  unsafe_import_insertion: 'This source has no supported location for the required import. Edit the import directly in VS Code.',
  incomplete_or_mutating_observation: 'This failed or mutating observation does not support the source recipe.',
  source_unavailable: 'The verified source needed for this edit is unavailable.',
  source_binding_mismatch: 'The source location no longer matches the saved expression.',
  stale_source: 'The source changed. Refresh and validate the current saved model.',
  source_limit: 'The source exceeds the edit resolver’s current limits.',
  capability_limit: 'The graph reached its edit-capability limit.',
  unsupported_source_scope: 'Reconnection needs a supported straight-line forward method. Edit this control flow in VS Code.',
  unsupported_source_expression: 'An expression in this forward method has no source-flow recipe yet.',
  incomplete_source_flow: 'This capture cannot prove every call and value needed for reconnection.',
  mutating_source_scope: 'This forward scope changes tensors in place, so available replacement values cannot be proven.',
  ambiguous_source_flow: 'The source assignments do not have an exact match to the captured values.',
  ambiguous_source_scope: 'Not every execution of this expression has a proven source scope.',
  ambiguous_source_parameters: 'The captured call does not explicitly bind every forward parameter.',
  source_flow_limit: 'This forward method exceeds the reconnection proof limits.',
  no_named_tensor_operands: 'This expression has no supported named tensor operand to reconnect.',
  unproven_unary_namespace: 'A new activation cannot be added through this file’s mutable or dynamic namespace.',
  unsupported_unary_deletion: 'This activation needs a supported single-line call with a named input to be removed.',
};

export function CapturedSourceEditForm({ nodes, compact = false }: { nodes: ModelNode[]; compact?: boolean }) {
  const context = React.useContext(LocalBrowserContext);
  const node = nodes.length === 1 ? nodes[0] : undefined;
  const ref = node?.data.captureRef;
  const capabilities = node?.data.sourceEditCapabilities?.filter(cap =>
    ((cap.command === 'set_literal' && cap.recipe === 'binary_scalar_v1') ||
      (cap.command === 'set_constructor_literal' && cap.recipe === 'constructor_scalar_v1') ||
      (cap.command === 'set_configuration_literal' && cap.recipe === 'configuration_scalar_v1') ||
      (cap.command === 'reconnect_argument' && cap.recipe === 'straight_line_tensor_v1') ||
      (cap.command === 'insert_unary_op' && cap.recipe === 'straight_line_unary_insert_v1') ||
      (cap.command === 'delete_unary_op' && cap.recipe === 'straight_line_unary_delete_v1') ||
      (cap.command === 'replace_activation' && ['functional_activation_v1', 'module_activation_v1'].includes(cap.recipe))) &&
    cap.target.kind === ref?.kind && cap.target.captureId === ref?.captureId &&
    cap.target.sourceId === ref.sourceId && cap.target.semanticId === ref.semanticId) ?? [];
  const [selectedId, setSelectedId] = React.useState<string>();
  const capability = capabilities.find(cap => cap.id === selectedId) ?? capabilities[0];
  const numeric = capability?.command === 'set_literal' || capability?.command === 'set_constructor_literal' || capability?.command === 'set_configuration_literal';
  const initialKind = numeric ? capability.expected.kind : 'int';
  const initialText = numeric ? String(capability.expected.value) : '';
  const initialActivation = capability?.command === 'replace_activation' ? capability.expected : 'relu';
  const initialTensor = capability?.command === 'reconnect_argument' ? capability.expected.id : '';
  const compatibleActivations = capability?.command === 'replace_activation'
    ? capability.compatibleActivations ?? ACTIVATION_NAMES : ACTIVATION_NAMES;
  const [kind, setKind] = React.useState<SourceNumber['kind']>('int');
  const [text, setText] = React.useState('');
  const [activation, setActivation] = React.useState<ReplaceActivationEdit['value']>('relu');
  const [tensorId, setTensorId] = React.useState('');
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<SourceEditOutcome | null>(null);
  const generation = React.useRef(0);
  React.useEffect(() => {
    const current = ++generation.current;
    setKind(initialKind); setText(initialText); setActivation(initialActivation); setTensorId(initialTensor);
    setResult(null); setPending(false);
    return () => { generation.current = current + 1; context?.sourceEdits?.cancel(); };
  }, [capability?.id, initialKind, initialText, initialActivation, initialTensor, context?.sourceEdits]);
  const connected = context?.status?.phase === 'connected' && context.status.hostLink === 'connected';
  const displayed = (context?.page.projection?.document as { semantic?: { metadata?: { capture?: { id?: string; sourceId?: string } } } } | undefined)?.semantic?.metadata?.capture;
  const current = !!capability && displayed?.id === capability.target.captureId && displayed?.sourceId === capability.target.sourceId;
  const ready = !!context?.sourceEdits && connected && context.page.syncState !== 'stale' && current && !pending;
  const value = parseSourceNumber(text, kind);
  const tensor = capability?.command === 'reconnect_argument' ? capability.choices.find(choice => choice.value.id === tensorId) : undefined;
  const valid = capability?.command === 'replace_activation' || capability?.command === 'insert_unary_op'
    ? compatibleActivations.includes(activation) : capability?.command === 'delete_unary_op' ? true
      : capability?.command === 'reconnect_argument' ? !!tensor : !!value;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready || !valid || !capability || !node || !context?.sourceEdits) return;
    let edit: LocalBrowserSourceEdit;
    const common = { nodeId: node.id, capabilityId: capability.id, target: capability.target, scope: 'source_expression' as const };
    if (capability.command === 'set_literal') {
      if (!value) return;
      edit = { ...common, kind: 'set_literal', argumentPath: capability.argumentPath, expected: capability.expected, value };
    } else if (capability.command === 'set_constructor_literal') {
      if (!value) return;
      edit = { ...common, kind: 'set_constructor_literal', sourceArgumentPath: capability.sourceArgumentPath, expected: capability.expected, value };
    } else if (capability.command === 'set_configuration_literal') {
      if (!value) return;
      edit = { ...common, kind: 'set_configuration_literal', scope: 'configuration_definition',
        sourceArgumentPath: capability.sourceArgumentPath, expected: capability.expected, value };
    } else if (capability.command === 'reconnect_argument') {
      if (!tensor) return;
      edit = { ...common, kind: 'reconnect_argument', argumentPath: capability.argumentPath, expected: capability.expected, value: tensor.value };
    } else if (capability.command === 'insert_unary_op') {
      edit = { ...common, kind: 'insert_unary_op', argumentPath: capability.argumentPath, expected: capability.expected, value: activation };
    } else if (capability.command === 'delete_unary_op') {
      edit = { ...common, kind: 'delete_unary_op', argumentPath: capability.argumentPath, expected: capability.expected, value: capability.input.value };
    } else {
      edit = { ...common, kind: 'replace_activation', expected: capability.expected, value: activation };
    }
    const current = generation.current;
    setPending(true); setResult(null);
    void context.sourceEdits.propose(edit).promise.then(outcome => {
      if (current === generation.current) { setResult(outcome); setPending(false); }
    });
  };
  if (!capability) return <section className="source-edit-form" aria-label="Source edit proposal">
    <p role="status">This captured selection has no supported source-edit recipe yet. {nodes.length > 1
      ? 'Select one captured operation or module call to see its supported edits.'
      : limitation[node?.data.sourceEditLimitation ?? ''] ?? 'A source location enables navigation; it does not make a derived operation independently editable.'}</p>
    <button type="button" disabled>Review in VS Code</button>
  </section>;
  const scope = capability.scope;
  return <section className="source-edit-form" aria-label="Source edit proposal">
    <details open={!compact}><summary>Source scope and effects</summary>
    <p>Change {capability.command === 'set_literal' ? 'the numeric second operand'
      : capability.command === 'set_constructor_literal' ? `a numeric argument to ${capability.constructor}`
        : capability.command === 'set_configuration_literal' ? `the definition of ${capability.symbol}`
        : capability.command === 'reconnect_argument' ? `operand ${Number(capability.argumentPath[1]) + 1} (${capability.expectedName})`
        : capability.command === 'insert_unary_op' ? `operand ${Number(capability.argumentPath[1]) + 1} by wrapping ${capability.expectedName} in an activation`
        : capability.command === 'delete_unary_op' ? `the ${capability.activation} call to use ${capability.input.name} directly`
        : `the ${capability.expected} activation`} in <strong>{(capability.command === 'set_configuration_literal' ? capability.binding : capability.expression).anchors[0].scope}</strong>.</p>
    {(capability.command === 'set_constructor_literal' || capability.command === 'set_configuration_literal') && <p>Constructor changes can affect tensor shapes and checkpoint compatibility. Other layers are not adjusted automatically.</p>}
    {capability.command === 'reconnect_argument' && <p>Choose a tensor available at this expression. Only the selected operand changes, even when both operands use the same tensor. Availability does not guarantee compatible dimensions.</p>}
    {capability.command === 'insert_unary_op' && <p>This adds a nonmutating activation to only the selected operand. Other uses of the tensor stay unchanged.</p>}
    {capability.command === 'delete_unary_op' && <p>This removes the activation call and uses its original input <strong>{capability.input.name}</strong>. Surrounding assignments and other tensor uses stay intact.</p>}
    {capability.recipe === 'module_activation_v1' && <p>This replaces the module constructor. Its attribute names and forward calls stay unchanged.</p>}
    {capability.command === 'replace_activation' && capability.options?.inplace != null && <p>
      The authored <code>inplace={capability.options.inplace ? 'True' : 'False'}</code> option stays unchanged. Only compatible activation types are offered.
      {capability.options.inplace && ' Input mutation is preserved, including its effect on aliases.'}
    </p>}
    {capability.command === 'replace_activation' && capability.options?.approximate != null && <p>
      The authored <code>approximate="{capability.options.approximate}"</code> option stays unchanged. Only GELU accepts this option; another activation type needs a source edit in VS Code.
    </p>}
    {(capability.command === 'replace_activation' || capability.command === 'insert_unary_op') && capability.importPlan && <p>The review also adds a uniquely named Torch import. Existing imports and their other uses stay unchanged.</p>}
    {scope.kind === 'configuration_definition' ? <>
      <p>This changes <strong>all references to this configuration definition</strong>, including uses not shown in this run. Constructor references stay intact.</p>
      <p>{scope.knownSourceReferences} known source references; {scope.observedConstructorSites} observed constructor sites.</p>
      <p>Definition: {capability.binding.anchors[0].file}:{capability.binding.anchors[0].startLine} ({capability.binding.anchors[0].scope}).</p>
    </> : <p>This changes <strong>every execution of this source expression</strong>, including calls not shown in this run.</p>}
    <p>Observed in {scope.observedOperations} operation{scope.observedOperations === 1 ? '' : 's'}, {scope.observedInvocations} call{scope.observedInvocations === 1 ? '' : 's'} and {scope.observedModuleInstances} module instance{scope.observedModuleInstances === 1 ? '' : 's'}.</p>
    {node?.data.sourceEditLimitation === 'capability_limit' && <p>Some editable fields are unavailable because this graph reached its edit-capability limit.</p>}
    </details>
    <form onSubmit={submit}>
      <fieldset disabled={!ready}>
        {capabilities.length > 1 && capability.command !== 'set_constructor_literal' && capability.command !== 'set_configuration_literal' &&
          <label>Captured edit<select aria-label="Captured edit" value={capability.id}
            onChange={event => { if (!pending) setSelectedId(event.target.value); }}>
            {capabilities.map(cap => <option key={cap.id} value={cap.id}>{cap.command === 'reconnect_argument'
              ? `Reconnect operand ${Number(cap.argumentPath[1]) + 1} (${cap.expectedName})`
              : cap.command === 'insert_unary_op' ? `Insert activation on operand ${Number(cap.argumentPath[1]) + 1} (${cap.expectedName})`
              : cap.command === 'delete_unary_op' ? `Remove ${cap.activation} (use ${cap.input.name})`
              : cap.command === 'set_literal' ? 'Change numeric second operand' : 'Replace activation'}</option>)}
          </select></label>}
        {(capability.command === 'set_constructor_literal' || capability.command === 'set_configuration_literal') && <label>Constructor argument<select aria-label="Constructor argument" value={capability.id}
          onChange={event => { if (!pending) setSelectedId(event.target.value); }}>
          {capabilities.map(cap => (cap.command === 'set_constructor_literal' || cap.command === 'set_configuration_literal') &&
            <option key={cap.id} value={cap.id}>{cap.sourceArgumentPath[0] === 'args'
              ? `Argument ${Number(cap.sourceArgumentPath[1]) + 1}` : cap.sourceArgumentPath[1]} = {cap.command === 'set_configuration_literal'
                ? `${cap.symbol} (${String(cap.expected.value)}; edit definition)` : String(cap.expected.value)}</option>)}
        </select></label>}
        {numeric ? <><label>Number type<select aria-label="Literal number type" value={kind} onChange={event => setKind(event.target.value as SourceNumber['kind'])}>
          <option value="int">Integer</option><option value="float">Float</option>
        </select></label>
        <label>New literal<input aria-label="New literal" value={text} onChange={event => setText(event.target.value)} maxLength={4096} required /></label>
        {!value && <p>Enter a {kind === 'int' ? 'decimal integer, such as 2 or -12' : 'finite number, such as 0.5 or -0.0'}.</p>}</>
          : capability.command === 'reconnect_argument' ? <label>Available tensor<select aria-label="Available tensor" value={tensorId}
            onChange={event => setTensorId(event.target.value)}>
            {capability.choices.map(choice => <option key={choice.value.id} value={choice.value.id}>
              {choice.name} — defined on line {choice.binding.anchors[0].startLine}{choice.value.id === capability.expected.id ? ' (current)' : ''}
            </option>)}
          </select></label>
          : capability.command === 'delete_unary_op' ? null : <label>New activation<select aria-label="New activation" value={activation} onChange={event => {
            if (isActivationName(event.target.value) && compatibleActivations.includes(event.target.value)) setActivation(event.target.value);
          }}>{compatibleActivations.map(name => <option key={name} value={name}>{name}</option>)}</select></label>}
        <button type="submit" disabled={!ready || !valid}>Review in VS Code</button>
      </fieldset>
    </form>
    <p>Review the shared source change in VS Code. Save and refresh to capture the updated model.</p>
    {pending && <div role="status"><p>Preparing the source diff. Continue the review in VS Code.</p><button onClick={() => context?.sourceEdits?.cancel()}>Cancel source proposal</button></div>}
    {result && <p role="status" data-status={result.status}>{result.detail}</p>}
    {(pending || result) && <CandidateReviewPanel embedded />}
    {!current && !pending && <p>This source snapshot is no longer displayed. Close this form and select a current captured operation to prepare another edit.</p>}
    {!connected && <p role="status">Source proposals pause until VS Code reconnects.</p>}
  </section>;
}
