import React from 'react';
import { act, fireEvent, screen } from '@testing-library/react';

let mockContext: any = { displayMode: 'inline', availableDisplayModes: ['inline', 'fullscreen'], theme: 'dark' };
const mockContextListeners = new Set<() => void>();
const mockApp: any = {
  connect: jest.fn(),
  sendSizeChanged: jest.fn(),
  getHostContext: () => mockContext,
  requestDisplayMode: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
};
function hostContextChanged() {
  mockApp.onhostcontextchanged();
  for (const listener of mockContextListeners) listener();
}
let mockOptions: unknown;
jest.mock('@modelcontextprotocol/ext-apps', () => ({ App: function (_info: unknown, _capabilities: unknown, options: unknown) { mockOptions = options; return mockApp; },
  applyDocumentTheme: jest.fn(), applyHostStyleVariables: jest.fn() }));
jest.mock('../../styles/fonts', () => ({}));
jest.mock('../../styles/canvasTheme', () => ({ configureCanvasTheme: jest.fn(), setCanvasHostTheme: jest.fn() }));
jest.mock('@tldraw/assets/selfHosted', () => ({ getAssetUrls: () => ({}) }));
jest.mock('../../canvas/TldrawCanvas', () => ({ CanvasAssetContext: require('react').createContext({}) }));
jest.mock('../../ide/LocalBrowserApp', () => ({ __esModule: true,
  default: ({ hostControls }: any) => <div>{hostControls}</div> }));
jest.mock('../../plugin/panelTransport', () => ({ createPanelTransport: jest.fn() }));
jest.mock('../../plugin/panelAssets', () => ({ panelAssetFetch: (_files: unknown, fetch: unknown) => fetch }));

test('restoring panels never covers the chat; fullscreen requires a click and follows the host exit', async () => {
  mockApp.addEventListener.mockImplementation((_event: string, listener: () => void) => mockContextListeners.add(listener));
  mockApp.removeEventListener.mockImplementation((_event: string, listener: () => void) => mockContextListeners.delete(listener));
  mockApp.connect.mockResolvedValue(undefined);
  mockApp.requestDisplayMode.mockImplementation(async ({ mode }: { mode: string }) => {
    mockContext = { ...mockContext, displayMode: mode };
    hostContextChanged();
    return { mode };
  });
  document.body.innerHTML = '<div id="root"></div>';
  window.fetch = jest.fn();
  await act(async () => { require('../../plugin/panel'); });
  expect(mockOptions).toEqual({ autoResize: false });
  expect(mockApp.sendSizeChanged).toHaveBeenLastCalledWith({ height: 600 });
  const themeHandler = mockApp.onhostcontextchanged;
  await act(async () => { mockApp.ontoolresult({ _meta: { panelId: 'restored-panel' } }); });
  expect(mockApp.onhostcontextchanged).toBe(themeHandler);
  expect(mockContextListeners.size).toBe(1);
  expect(mockApp.requestDisplayMode).not.toHaveBeenCalled();
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Full screen' })); });
  expect(mockApp.requestDisplayMode).toHaveBeenLastCalledWith({ mode: 'fullscreen' });
  expect(screen.getByRole('button', { name: 'Exit full screen' })).toBeInTheDocument();
  await act(async () => {
    mockContext = { ...mockContext, displayMode: 'inline' };
    hostContextChanged();
  });
  expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
  await act(async () => {
    mockContext = { ...mockContext, containerDimensions: { width: 700, maxHeight: 480 } };
    hostContextChanged();
  });
  expect(mockApp.sendSizeChanged).toHaveBeenLastCalledWith({ height: 480 });
  await act(async () => { mockApp.ontoolresult({ _meta: { panelId: 'another-restored-panel' } }); });
  expect(mockApp.onhostcontextchanged).toBe(themeHandler);
  expect(mockApp.removeEventListener).toHaveBeenCalledTimes(1);
  expect(mockContextListeners.size).toBe(1);
  expect(mockApp.requestDisplayMode).toHaveBeenCalledTimes(1);
  await act(async () => {
    mockContext = { ...mockContext, availableDisplayModes: ['inline'] };
    hostContextChanged();
  });
  expect(screen.queryByRole('button', { name: 'Full screen' })).not.toBeInTheDocument();
});
