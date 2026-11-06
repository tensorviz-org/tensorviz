import React from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { PROTOCOL_VERSION } from '../../ide/protocol.generated';
import { configureCanvasTheme, getCanvasTheme, setCanvasThemePreference } from '../../styles/canvasTheme';

test('host services remain available without accepting retired graph traffic', () => {
  const postMessage = jest.fn();
  const originalParent = window.parent;
  Object.defineProperty(window, 'parent', { value: { postMessage }, configurable: true });
  const commands = jest.fn();
  window.addEventListener('builder-command', commands);
  const interval = jest.spyOn(window, 'setInterval').mockReturnValue(1);
  try {
    localStorage.clear(); configureCanvasTheme('host');
    const bridge = require('../../ide/bridge') as typeof import('../../ide/bridge');
    const PublishGraphButton = require('../../ide/PublishGraphButton').default;
    const changed = jest.fn();
    const unsubscribe = bridge.subscribeBridge(changed);
    const removed = jest.fn();
    bridge.subscribeBridge(removed)();
    render(React.createElement(PublishGraphButton, { disabled: false }));
    fireEvent.click(screen.getByRole('button', { name: 'Export graph' }));
    expect(commands).toHaveBeenLastCalledWith(expect.objectContaining({ detail: { command: 'export-graph-file' } }));
    commands.mockClear();
    bridge.startBridge();
    expect(postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'ready' }), '*');
    const receive = (data: unknown, origin = 'vscode-webview://shell', source: Window | null = window.parent) => window.dispatchEvent(new MessageEvent('message', {
      source, origin, data,
    }));
    receive({ type: 'canvasTheme', theme: 'light' });
    receive({ type: 'hostReady', protocolVersion: PROTOCOL_VERSION, theme: 'light' }, 'vscode-webview://shell', null);
    expect(bridge.bridgeActive()).toBe(false);
    expect(changed).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Export graph' })).toBeInTheDocument();
    expect(getCanvasTheme()).toBe('dark');
    receive({ type: 'projection', projection: { file: 'old.py' } });
    act(() => receive({ type: 'hostReady', protocolVersion: PROTOCOL_VERSION, theme: 'light' }));
    expect(changed).toHaveBeenCalledTimes(1);
    expect(removed).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Publish graph' }));
    expect(commands).toHaveBeenLastCalledWith(expect.objectContaining({ detail: { command: 'share-graph-file' } }));
    expect(screen.queryByRole('button', { name: 'Export graph' })).not.toBeInTheDocument();
    commands.mockClear();
    receive({ type: 'hostReady', protocolVersion: PROTOCOL_VERSION, theme: 'light' });
    expect(changed).toHaveBeenCalledTimes(1);
    unsubscribe();
    expect(getCanvasTheme()).toBe('light');
    receive({ type: 'hostReady', protocolVersion: PROTOCOL_VERSION, theme: 'dark' }, 'https://other.example');
    receive({ type: 'canvasTheme', theme: 'dark' }, 'https://other.example');
    receive({ type: 'canvasTheme', theme: 'dark' }, 'vscode-webview://shell', null);
    receive({ type: 'canvasTheme', theme: 'invalid' });
    expect(getCanvasTheme()).toBe('light');
    receive({ type: 'canvasTheme', theme: 'dark' });
    expect(getCanvasTheme()).toBe('dark');
    setCanvasThemePreference('light');
    receive({ type: 'canvasTheme', theme: 'dark' });
    expect(getCanvasTheme()).toBe('light');
    receive({ type: 'projectDocument', document: { nodes: [] } });
    receive({ type: 'candidatePreview', candidate: {} });
    receive({ type: 'focusNode', nodeId: 'old-node' });
    expect(commands).not.toHaveBeenCalled();
    bridge.postToHost({ type: 'showCommandPalette' });
    expect(postMessage).toHaveBeenLastCalledWith({ type: 'showCommandPalette' }, 'vscode-webview://shell');
  } finally {
    cleanup();
    window.removeEventListener('builder-command', commands);
    Object.defineProperty(window, 'parent', { value: originalParent, configurable: true });
    interval.mockRestore();
    localStorage.clear();
  }
});
