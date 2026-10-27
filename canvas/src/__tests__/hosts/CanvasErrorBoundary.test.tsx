import { fireEvent, render, screen } from '@testing-library/react';
import { CanvasErrorBoundary } from '../../ide/CanvasErrorBoundary';

let mockCanvasFails = false;
jest.mock('../../ide/LocalBrowserApp', () => () => {
  if (mockCanvasFails) throw new TypeError("Cannot read properties of undefined (reading 'length')");
  return require('react').createElement('p', null, 'Canvas ready');
});
jest.mock('../../canvas/TldrawCanvas', () => ({ CanvasAssetContext: require('react').createContext(undefined) }));
jest.mock('@tldraw/assets/selfHosted', () => ({ getAssetUrls: () => ({}) }));

let quiet: jest.SpyInstance;
beforeEach(() => { quiet = jest.spyOn(console, 'error').mockImplementation(() => {}); });
afterEach(() => { quiet.mockRestore(); mockCanvasFails = false; });

test('an uncaught Canvas error shows what failed instead of a blank view, and Reload remounts the Canvas', () => {
  let fail = true;
  function Canvas() {
    if (fail) throw new TypeError("Cannot read properties of undefined (reading 'length')");
    return <p>Graph ready</p>;
  }
  render(<CanvasErrorBoundary><Canvas /></CanvasErrorBoundary>);
  expect(screen.getByRole('heading', { name: 'The Canvas stopped after an error' })).toBeVisible();
  expect(screen.getByRole('alert')).toHaveTextContent("Cannot read properties of undefined (reading 'length')");
  expect(screen.getByText('Error details')).toBeVisible();
  expect(screen.getByText(/^TypeError: Cannot read properties/)).toBeInTheDocument();
  fail = false;
  fireEvent.click(screen.getByRole('button', { name: 'Reload Canvas' }));
  expect(screen.getByText('Graph ready')).toBeVisible();
});

test('the extension Canvas entry is wrapped, so a crash in any view stays visible', () => {
  mockCanvasFails = true;
  const IdeApp = require('../../ide/main').default;
  render(<IdeApp />);
  expect(screen.getByRole('heading', { name: 'The Canvas stopped after an error' })).toBeVisible();
  mockCanvasFails = false;
  fireEvent.click(screen.getByRole('button', { name: 'Reload Canvas' }));
  expect(screen.getByText('Canvas ready')).toBeVisible();
});
