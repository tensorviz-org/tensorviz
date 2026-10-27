const mockRender = jest.fn();
const mockAssets = { fonts: { code: './assets/fonts/code.woff2' } };
jest.mock('react-dom/client', () => ({ __esModule: true,
  default: { createRoot: () => ({ render: mockRender }) },
}));
jest.mock('../../styles/fonts', () => ({}));
jest.mock('../../ide/LocalBrowserApp', () => ({ __esModule: true, default: () => null }));
jest.mock('@tldraw/assets/selfHosted', () => ({ getAssetUrls: () => mockAssets }));
jest.mock('../../canvas/TldrawCanvas', () => ({ CanvasAssetContext: require('react').createContext(null) }));

beforeEach(() => {
  localStorage.clear();
  mockRender.mockReset();
  document.body.innerHTML = '<div id="root"></div>';
});
afterEach(() => localStorage.clear());

function openBrowser() {
  let theme!: typeof import('../../styles/canvasTheme');
  let mountedTheme: string | undefined;
  jest.isolateModules(() => {
    theme = require('../../styles/canvasTheme');
    // A previous embedded view must not choose the standalone browser's theme.
    theme.configureCanvasTheme('host');
    mockRender.mockImplementation(() => { mountedTheme = theme.getCanvasTheme(); });
    require('../../plugin/index');
  });
  expect(mockRender).toHaveBeenCalledTimes(1);
  const assetProvider = mockRender.mock.calls[0][0].props.children;
  expect(assetProvider.props.value).toBe(mockAssets);
  expect(assetProvider.props.children.props.hostKind).toBe('standalone');
  return { theme, mountedTheme };
}

test('the plugin browser selects Paper before mounting Canvas and ignores embedded host themes', () => {
  const { theme, mountedTheme } = openBrowser();
  expect(mountedTheme).toBe('light');
  expect(document.documentElement.dataset.tvTheme).toBe('light');
  theme.setCanvasHostTheme('dark');
  expect(theme.getCanvasTheme()).toBe('light');
});

test('the plugin browser restores its own choice and Default clears only that browser preference', () => {
  localStorage.setItem('tensorviz.canvas.theme.browser', 'dark');
  localStorage.setItem('tensorviz.canvas.theme.host', 'light');
  const { theme, mountedTheme } = openBrowser();
  expect(mountedTheme).toBe('dark');
  expect(theme.getCanvasThemePreference()).toBe('dark');
  theme.setCanvasThemePreference('default');
  expect(theme.getCanvasTheme()).toBe('light');
  expect(localStorage.getItem('tensorviz.canvas.theme.browser')).toBeNull();
  expect(localStorage.getItem('tensorviz.canvas.theme.host')).toBe('light');
});
