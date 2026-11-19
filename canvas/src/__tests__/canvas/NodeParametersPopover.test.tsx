import React from 'react';
import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/react';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { NodeParametersPopover as ParameterPopover, TemplateEditingContext } from '../../canvas/NodeParametersPopover';
import { closeNodeParameters, openNodeParameters, parameterTarget } from '../../canvas/nodeParameters';

let mockEditor: any;
let mockShape: any;
const mockSourceEditor = jest.fn();
const NodeParametersPopover = () => <TemplateEditingContext.Provider value={true}><ParameterPopover /></TemplateEditingContext.Provider>;

jest.mock('tldraw', () => ({
  atom: (_name: string, initial: unknown) => {
    let value = initial;
    return { get: () => value, set: (next: unknown) => { value = next; } };
  },
  useValue: (...args: any[]) => typeof args[0] === 'string' ? args[1]() : args[0].get(),
  useEditor: () => mockEditor,
}));
jest.mock('../../canvas/LocalSourceEditor', () => ({
  LocalSourceEditor: (props: unknown) => { mockSourceEditor(props); return <div data-testid="local-source-editor" />; },
}));
jest.mock('../../components/ui2/ParameterFields', () => ({
  __esModule: true,
  default: () => <input aria-label="Template parameter" />,
}));

beforeEach(() => {
  jest.clearAllMocks();
  closeNodeParameters();
  mockShape = { id: 'shape:fc', type: 'layer', props: { nodeData: {
    layerType: { id: 'linear', displayName: 'Linear' }, parameters: { out_features: 8 },
    sourceLocation: { file: 'model.py', line: 4 },
  } } };
  mockEditor = {
    getCamera: () => ({ x: 0, y: 0, z: 1 }),
    getShape: () => mockShape,
    getShapePageBounds: () => ({ maxX: 200, y: 100 }),
    pageToViewport: (point: object) => point,
    getViewportScreenBounds: () => ({ w: 800 }),
  };
});

test.each([undefined, { propose: jest.fn() }])('local source editing never renders a popup, with sourceEdits=%p', sourceEdits => {
  openNodeParameters(mockShape.id);
  const view = render(<LocalBrowserContext.Provider value={{ page: {}, sourceEdits } as any}>
    <NodeParametersPopover />
  </LocalBrowserContext.Provider>);
  expect(view.container).toBeEmptyDOMElement();
  expect(mockSourceEditor).not.toHaveBeenCalled();
});

test.each([
  ['layer', { focusBoundary: { direction: 'in', externalNodeId: 'external' } }],
  ['module-node', { focusBoundary: { direction: 'out', externalNodeId: 'external' } }],
  ['input-node', {}],
  ['output-node', {}],
])('a targeted %s boundary never opens an editor (%p)', (type, data) => {
  mockShape.type = type;
  mockShape.props.nodeData = { ...mockShape.props.nodeData, ...data };
  openNodeParameters(mockShape.id);
  const view = render(<NodeParametersPopover />);
  expect(view.container).toBeEmptyDOMElement();
  expect(mockSourceEditor).not.toHaveBeenCalled();
});

test('the nonlocal template editor stays closed until explicitly targeted', () => {
  const view = render(<NodeParametersPopover />);
  expect(screen.queryByTestId('node-parameters')).toBeNull();
  openNodeParameters(mockShape.id);
  view.rerender(<NodeParametersPopover />);
  expect(screen.getByTestId('node-parameters')).toBeVisible();
  expect(screen.getByLabelText('Template parameter')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Close parameters' }));
  view.rerender(<NodeParametersPopover />);
  expect(screen.queryByTestId('node-parameters')).toBeNull();
});

test.each([undefined, false])('an explicit target cannot edit without a canvas capability (%p)', capability => {
  openNodeParameters(mockShape.id);
  render(capability === undefined ? <ParameterPopover /> : <TemplateEditingContext.Provider value={capability}>
    <ParameterPopover />
  </TemplateEditingContext.Provider>);
  expect(screen.queryByTestId('node-parameters')).not.toBeInTheDocument();
});

test('clicking a card underneath an explicit template popup selects it without opening another editor', () => {
  mockEditor.screenToPage = (point: object) => point;
  mockEditor.getShapeAtPoint = () => ({ id: 'shape:other', type: 'layer' });
  mockEditor.select = jest.fn();
  openNodeParameters(mockShape.id);
  render(<NodeParametersPopover />);
  fireEvent.pointerDown(screen.getByTestId('node-parameters'), { clientX: 250, clientY: 150 });
  expect(mockEditor.select).toHaveBeenCalledWith('shape:other');
  expect(parameterTarget.get()).toBeNull();
});
