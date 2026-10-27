import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { CanvasMinimap } from '../../canvas/CanvasMinimap';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';

const mockEditor = {
  getViewportPageBounds: () => ({ width: 100, height: 80, center: { x: 50, y: 40 } }),
  centerOnPoint: jest.fn(),
  zoomIn: jest.fn(),
  zoomOut: jest.fn(),
  zoomToFit: jest.fn(),
  markEventAsHandled: jest.fn(),
};

jest.mock('tldraw', () => ({
  useEditor: () => mockEditor,
  DefaultMinimap: () => <canvas aria-label="Minimap" />,
}));

beforeEach(() => jest.clearAllMocks());

test('the local canvas starts with a compact overview and opens the existing minimap on demand', () => {
  render(<LocalBrowserContext.Provider value={{ page: EMPTY_LOCAL_BROWSER_PAGE_STATE }}><CanvasMinimap /></LocalBrowserContext.Provider>);
  expect(screen.queryByLabelText('Minimap')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Show minimap' }));
  expect(screen.getByLabelText('Minimap')).toBeVisible();
});

test('keyboard navigation pans by a fraction of the current viewport and supports zoom and fit', () => {
  render(<CanvasMinimap />);
  const navigation = screen.getByRole('group', { name: 'Minimap navigation' });
  fireEvent.keyDown(navigation, { key: 'ArrowRight' });
  expect(mockEditor.centerOnPoint).toHaveBeenLastCalledWith({ x: 70, y: 40 }, expect.anything());
  fireEvent.keyDown(navigation, { key: 'ArrowUp' });
  expect(mockEditor.centerOnPoint).toHaveBeenLastCalledWith({ x: 50, y: 24 }, expect.anything());
  fireEvent.keyDown(navigation, { key: '+' });
  fireEvent.keyDown(navigation, { key: '-' });
  fireEvent.keyDown(navigation, { key: 'Home' });
  expect(mockEditor.zoomIn).toHaveBeenCalledTimes(1);
  expect(mockEditor.zoomOut).toHaveBeenCalledTimes(1);
  expect(mockEditor.zoomToFit).toHaveBeenCalledTimes(1);
});

test('Escape collapses the overview and restores focus to its toggle', () => {
  render(<CanvasMinimap />);
  const navigation = screen.getByRole('group', { name: 'Minimap navigation' });
  navigation.focus();
  fireEvent.keyDown(navigation, { key: 'Escape' });
  expect(screen.queryByRole('group', { name: 'Minimap navigation' })).not.toBeInTheDocument();
  const show = screen.getByRole('button', { name: 'Show minimap' });
  expect(show).toHaveFocus();
  expect(show).toHaveAttribute('aria-expanded', 'false');
  fireEvent.click(show);
  expect(screen.getByRole('group', { name: 'Minimap navigation' })).toBeVisible();
});

test('modified arrow keys do not hijack browser or operating-system shortcuts', () => {
  render(<CanvasMinimap />);
  fireEvent.keyDown(screen.getByRole('group', { name: 'Minimap navigation' }), { key: 'ArrowLeft', altKey: true });
  expect(mockEditor.centerOnPoint).not.toHaveBeenCalled();
});

// DefaultMinimap ends its drag with native body listeners, outside React.
test.each(['pointerup', 'pointercancel'])('%s reaches minimap drag cleanup on the document body', type => {
  const finishDrag = jest.fn();
  document.body.addEventListener(type, finishDrag);
  try {
    render(<CanvasMinimap />);
    fireEvent(screen.getByLabelText('Minimap'), new Event(type, { bubbles: true }));
    expect(finishDrag).toHaveBeenCalledTimes(1);
  } finally {
    document.body.removeEventListener(type, finishDrag);
  }
});

test('minimap presses stay isolated from canvas selection gestures', () => {
  const canvasPointerDown = jest.fn();
  render(<div onPointerDown={canvasPointerDown}><CanvasMinimap /></div>);
  fireEvent.pointerDown(screen.getByLabelText('Minimap'));
  expect(canvasPointerDown).not.toHaveBeenCalled();
});

test('minimap release is marked handled without becoming a canvas click', () => {
  render(<CanvasMinimap />);
  fireEvent.pointerUp(screen.getByLabelText('Minimap'));
  expect(mockEditor.markEventAsHandled).toHaveBeenCalledTimes(1);
});
