import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { addSourceEdit, constructorParameters, parameterSourceEdit, removeSourceEdit, insertUnarySourceEdit } from '../../ide/directSourceEdits';
import { LocalSourceEditor } from '../../canvas/LocalSourceEditor';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { ModelNodeData } from '../../types';

const target = { version: 1, captureId: 'capture', sourceId: 'source', kind: 'invocation', semanticId: 'fc' };
const cap: any = { command: 'set_constructor_literal', id: 'constructor-out', constructor: 'Linear',
  target, expected: { kind: 'int', value: '8' }, sourceArgumentPath: ['args', 1],
  expression: { kind: 'constructor', quality: 'exact' }, scope: { kind: 'source_expression' } };
const data = { layerType: { id: 'linear' }, parameters: { out_features: 8 }, captureRef: target,
  sourceEditCapabilities: [cap] } as unknown as ModelNodeData;

test('constructor parameter proposals retain the exact host capability identity', () => {
  expect(constructorParameters(data)).toContain('out_features');
  expect(parameterSourceEdit('fc', data, 'out_features', 16)).toEqual({ kind: 'set_constructor_literal',
    nodeId: 'fc', capabilityId: cap.id, target, expected: cap.expected, scope: 'source_expression',
    sourceArgumentPath: ['args', 1], value: { kind: 'int', value: '16' } });
  expect(() => parameterSourceEdit('fc', data, 'out_features', 1.5)).toThrow('same type');
});
test('safe declaration add and delete use existing proposal kinds', () => {
  for (const layer of ['linear', 'dropout', 'layernorm', 'relu']) {
    expect(addSourceEdit('fc', data, layer, 'new_layer', { p: 0.5 })).toEqual({ kind: 'add_layer',
      afterNodeId: 'fc', nodeId: 'new_layer', layerType: layer, params: { p: 0.5 } });
  }
  expect(removeSourceEdit('fc', data)).toEqual({ kind: 'remove_layer', nodeId: 'fc' });
  const deletion = { command: 'delete_unary_op', id: 'delete', target, expected: { kind: 'value', id: 'x' }, scope: { kind: 'source_expression' }, argumentPath: ['args', 0], input: { value: { kind: 'value', id: 'input' } } };
  expect(removeSourceEdit('act', { ...data, sourceEditCapabilities: [deletion] } as any)).toEqual({
    kind: 'delete_unary_op', nodeId: 'act', capabilityId: 'delete', target, expected: deletion.expected, scope: 'source_expression', argumentPath: deletion.argumentPath, value: deletion.input.value });
});
test('navigation-only nodes explain why direct editing is unavailable', () => {
  const unsupported = { ...data, sourceEditCapabilities: [], sourceEditLimitation: 'Dynamic constructor cannot be edited safely.' };
  expect(() => parameterSourceEdit('fc', unsupported, 'out_features', 16)).toThrow('Dynamic constructor');
  expect(() => addSourceEdit('fc', unsupported, 'dropout', 'drop', {})).toThrow('Dynamic constructor');
  expect(() => removeSourceEdit('fc', unsupported)).toThrow('Dynamic constructor');
});
test('local inspector proposes parameters, add and deletion and displays the host refusal', async () => {
  const propose = jest.fn((_edit: any) => ({ promise: Promise.resolve({ status: 'refused', detail: 'EditApplier refused this dynamic call.' }) }));
  render(<LocalBrowserContext.Provider value={{ sourceEdits: { propose }, page: {} } as any}>
    <LocalSourceEditor nodeId="fc" data={data} />
  </LocalBrowserContext.Provider>);
  fireEvent.change(screen.getByLabelText('out_features'), { target: { value: '16' } });
  fireEvent.blur(screen.getByLabelText('out_features'));
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('EditApplier refused'));
  expect(propose.mock.calls[0][0].kind).toBe('set_constructor_literal');
  fireEvent.click(screen.getByText('Review insertion'));
  await waitFor(() => expect(propose).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByText('Review deletion')).not.toBeDisabled());
  fireEvent.click(screen.getByText('Review deletion'));
  await waitFor(() => expect(propose).toHaveBeenCalledTimes(3));
  expect(propose.mock.calls.map(call => call[0].kind)).toEqual(['set_constructor_literal', 'add_layer', 'remove_layer']);
});

test('activation insertion preserves the exact operand capability', () => {
  const insertion = { command: 'insert_unary_op', id: 'insert', target, expected: { kind: 'value', id: 'x' }, scope: { kind: 'source_expression' }, argumentPath: ['args', 0] };
  expect(insertUnarySourceEdit('act', { ...data, sourceEditCapabilities: [insertion] } as any, 'insert', 'relu')).toEqual({
    kind: 'insert_unary_op', nodeId: 'act', capabilityId: 'insert', target, expected: insertion.expected,
    scope: 'source_expression', argumentPath: insertion.argumentPath, value: 'relu' });
});

test('Dropout insertion explains eval hiding and the unchanged-calls control', () => {
  render(<LocalBrowserContext.Provider value={{ page: {} } as any}>
    <LocalSourceEditor nodeId="fc" data={data} />
  </LocalBrowserContext.Provider>);
  expect(screen.getByText(/In eval captures, Dropout is recorded/)).toHaveTextContent('More tools → Show unchanged calls (N)');
  fireEvent.change(screen.getByLabelText('Layer to insert'), { target: { value: 'layernorm' } });
  expect(screen.queryByText(/In eval captures, Dropout is recorded/)).not.toBeInTheDocument();
});
