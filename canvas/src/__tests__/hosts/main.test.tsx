import React from 'react';
import { render, screen } from '@testing-library/react';

const mockAssets = { fonts: { code: './viewer/assets/fonts/code.woff2' } };
const mockGetAssetUrls = jest.fn((_options: unknown) => mockAssets);
jest.mock('@tldraw/assets/selfHosted', () => ({ getAssetUrls: (options: unknown) => mockGetAssetUrls(options) }));
jest.mock('../../canvas/TldrawCanvas', () => ({ CanvasAssetContext: require('react').createContext(undefined) }));

jest.mock('../../ide/LocalBrowserApp', () => ({
  __esModule: true,
  default: ({ createTransport }: { createTransport?: unknown }) =>
    <div>{createTransport ? 'Native Canvas' : 'Browser Canvas'}</div>,
}));
jest.mock('../../ide/nativeCanvasTransport', () => ({ createNativeCanvasTransport: jest.fn() }));
jest.mock('../../ide/bridge', () => ({ startBridge: jest.fn() }));
jest.mock('../../config/apiBase', () => ({ setLocalApiToken: jest.fn() }));
// Any fallback to the retired builder is a regression, even for an expired URL.
jest.mock('../../App', () => { throw new Error('Extension loaded the retired builder'); });

afterEach(() => { window.history.replaceState(null, '', '/'); });

test.each([
  ['/?view=canvas', 'Native Canvas'],
  ['/#session=ws-1&credential=test', 'Browser Canvas'],
  ['/', 'Browser Canvas'],
  ['/#/builder/old-saved-graph', 'Browser Canvas'],
  ['/?view=compare', 'Browser Canvas'],
])('%s mounts %s without a legacy fallback', (url, expected) => {
  window.history.replaceState(null, '', url);
  jest.isolateModules(() => {
    const Main = require('../../ide/main').default;
    render(<Main />);
  });
  expect(screen.getByText(expected)).toBeInTheDocument();
});

test('the packaged Canvas uses drawing assets already shipped with the viewer instead of the CDN', () => {
  const environment = process.env.NODE_ENV;
  Object.assign(process.env, { NODE_ENV: 'production' });
  mockGetAssetUrls.mockClear();
  try {
    jest.isolateModules(() => {
      require('../../ide/main');
      expect(mockGetAssetUrls).toHaveBeenCalledWith({ baseUrl: './viewer/assets' });
    });
  } finally { Object.assign(process.env, { NODE_ENV: environment }); }
});
