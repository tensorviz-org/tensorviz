import { readFileSync } from 'fs';
import { join } from 'path';

const css = (file: string) => readFileSync(join(__dirname, '../../ide', file), 'utf8');

test('labeled toolbar controls are not forced into an icon-sized box', () => {
  const rules = css('LocalBrowserApp.css');
  expect(rules).toContain('.local-canvas-control:not(:has(.local-canvas-control-label)):has(> svg:only-child)');
  expect(rules).toContain('.local-canvas-control:has(.local-canvas-control-label) { width: auto; }');
});

test('the chat button sits on the left so it clears the tldraw watermark', () => {
  const rules = css('CanvasAgentChat.css');
  expect(rules).toMatch(/\.canvas-agent-bubble, \.canvas-agent-chat \{[^}]*left: 12px/);
  expect(rules).not.toMatch(/\.canvas-agent-bubble, \.canvas-agent-chat \{[^}]*right: 12px/);
});
