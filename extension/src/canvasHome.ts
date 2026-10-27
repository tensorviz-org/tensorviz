import { randomUUID } from 'node:crypto';
import type { CanvasHomeSetup } from './canvasHomeSetup';

export interface CanvasHomeModel {
  name: string;
  file: string;
  uri: string;
}

export interface CanvasHomeState {
  hasWorkspace: boolean;
  indexed: boolean;
  models: CanvasHomeModel[];
  setup?: CanvasHomeSetup;
  pending?: string;
  error?: string;
  retryAvailable?: boolean;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** A local starting screen: no service connection, remote assets or command URIs. */
export function canvasHomeParts(state: CanvasHomeState): { style: string; body: string; script: string } {
  const models = state.models.map((model, index) => `<li><button class="model" data-model="${index}" ${state.pending ? 'disabled' : ''}>
    <span class="model-mark" aria-hidden="true">◇</span>
    <span class="model-label"><strong>${escapeHtml(model.name)}</strong><span class="file">${escapeHtml(model.file)}</span></span>
    </button></li>`).join('');
  const guidance = !state.hasWorkspace ? 'Open your model folder, or an empty folder to try a starter model.'
    : !state.indexed ? 'Open a Python model file, then choose Open as Graph. Discovered models will appear here.'
    : 'No PyTorch models were found by the last scan. Open a model file to choose one.';
  return { style: `
:root { color-scheme: light dark; --accent: var(--vscode-textLink-foreground, #3794ff); }
* { box-sizing: border-box; }
body { margin: 0; color: var(--vscode-foreground); background: var(--vscode-sideBar-background);
  font: var(--vscode-font-size, 13px)/1.6 var(--vscode-font-family, sans-serif); }
main { min-height: 100vh; display: flex; flex-direction: column; padding: 28px 24px 18px;
  max-width: 460px; margin: auto; }
.brand { display: flex; align-items: center; gap: 9px; font-size: 15px; font-weight: 600; }
.brand svg { width: 22px; height: 22px; color: var(--accent); flex-shrink: 0; }
.intro { margin: auto 0; padding: 24px 0 32px; }
.art { position: relative; margin: 0 -16px 24px; overflow: hidden; }
.art::before { content: ''; position: absolute; inset: -24px -8px;
  background: radial-gradient(circle, color-mix(in srgb, var(--accent) 22%, transparent) 1px, transparent 1px) 0 0 / 18px 18px;
  mask-image: radial-gradient(ellipse, black, transparent 72%); }
.network { position: relative; display: block; width: 100%; height: auto; }
.wires { fill: none; stroke: var(--accent); stroke-width: 1; opacity: .28; }
.signal { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: round;
  stroke-dasharray: 7 93; animation: flow 7s linear infinite; }
.signal.second { animation-delay: -2.3s; }
.signal.third { animation-delay: -4.6s; }
.nodes { fill: var(--vscode-editor-background); stroke: var(--accent); stroke-width: 1.2; }
.node-detail { fill: none; stroke: var(--accent); stroke-width: 1; opacity: .5; }
@keyframes flow { from { stroke-dashoffset: 100; } to { stroke-dashoffset: 0; } }
body[data-motion="paused"] .signal { animation-play-state: paused; }
@media (prefers-reduced-motion: reduce) { .signal { animation: none; stroke-dasharray: none; opacity: .3; } }
h1 { font-size: clamp(21px, 7vw, 28px); line-height: 1.2; letter-spacing: -.6px; font-weight: 600; margin: 0 0 12px; }
p { margin: 0; color: var(--vscode-descriptionForeground); }
.guidance { margin-bottom: 24px; }
.actions { display: flex; flex-direction: column; gap: 10px; }
.starter-hint { font-size: 12px; text-align: center; }
.canvas-home { height: 100%; overflow: auto; }
button:disabled { cursor: wait; opacity: .6; }
.setup { margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--vscode-widget-border, transparent); }
.setup strong { font-size: 13px; }
.setup p, .setup .file { margin-top: 5px; font-size: 12px; }
.setup-controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.setup-controls button { padding: 5px 9px; border-radius: 4px; color: var(--vscode-foreground); background: var(--vscode-button-secondaryBackground); }
.setup-progress { display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 50%; background: var(--accent); }
button { font: inherit; cursor: pointer; border: 1px solid var(--vscode-button-border, transparent); }
.action { display: flex; align-items: center; justify-content: center; gap: 8px; min-height: 38px;
  width: 100%; padding: 7px 12px; border-radius: 5px; color: var(--vscode-button-foreground);
  background: var(--vscode-button-background); font-weight: 500; }
.action svg { width: 17px; height: 17px; flex-shrink: 0; }
.action:hover { background: var(--vscode-button-hoverBackground); }
.secondary { color: var(--vscode-button-secondaryForeground); background: var(--vscode-button-secondaryBackground); }
.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
button:focus-visible { outline: 2px solid var(--vscode-focusBorder); outline-offset: 3px; }
ul { list-style: none; margin: 0 0 24px; padding: 0; }
.model { display: flex; align-items: center; gap: 12px; width: 100%; text-align: left; padding: 12px 4px;
  color: var(--vscode-foreground); background: transparent; border: 0;
  border-bottom: 1px solid var(--vscode-widget-border, transparent); }
.model:hover { background: var(--vscode-list-hoverBackground); }
.model-mark { color: var(--accent); font-size: 24px; flex-shrink: 0; }
.model-label { min-width: 0; }
strong, .file { display: block; overflow-wrap: anywhere; }
strong { font-weight: 500; }
.file { color: var(--vscode-descriptionForeground); font-size: 12px; }
.has-models .intro { margin: 0; padding-top: 28px; }
.has-models h1 { font-size: 21px; }
footer { display: flex; flex-direction: column; align-items: flex-start; gap: 8px;
  color: var(--vscode-descriptionForeground); font-size: 11px; }
[hidden] { display: none !important; }
@media (max-width: 260px) { main { padding: 20px 16px 16px; } }
@media (max-height: 520px) { .intro { margin: 0; padding-top: 20px; } .art { max-width: 230px; margin: 0 auto 20px; } }
body.vscode-high-contrast .wires, body.vscode-high-contrast-light .wires { opacity: .7; }
`, body: `<main class="${models ? 'has-models' : 'empty'}">
<header class="brand"><svg aria-hidden="true" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6">
  <path d="M8 7h16M8 7v18m0-9h16m0-9v18M8 25h16"/>
  <circle cx="8" cy="7" r="3.2"/><circle cx="24" cy="7" r="3.2"/>
  <circle cx="8" cy="25" r="3.2"/><circle cx="24" cy="25" r="3.2"/><circle cx="24" cy="16" r="3.2"/>
</svg>tensorViz</header>
<section class="intro" aria-labelledby="home-title">
${!models ? `<div class="art" aria-hidden="true">
<svg class="network" viewBox="0 0 320 190" focusable="false">
  <g class="wires">
    <path d="M40 95C68 95 66 35 96 35M40 95H96M40 95C68 95 66 155 96 155"/>
    <path d="M128 35C160 35 160 65 192 65M128 35C164 35 156 125 192 125"/>
    <path d="M128 95C160 95 160 65 192 65M128 95C160 95 160 125 192 125"/>
    <path d="M128 155C164 155 156 65 192 65M128 155C160 155 160 125 192 125"/>
    <path d="M224 65C256 65 250 95 280 95M224 125C256 125 250 95 280 95"/>
  </g>
  <path class="signal" pathLength="100" d="M40 95C68 95 66 35 96 35M128 35C160 35 160 65 192 65M224 65C256 65 250 95 280 95"/>
  <path class="signal second" pathLength="100" d="M40 95H96M128 95C160 95 160 125 192 125M224 125C256 125 250 95 280 95"/>
  <path class="signal third" pathLength="100" d="M40 95C68 95 66 155 96 155M128 155C160 155 160 125 192 125M224 125C256 125 250 95 280 95"/>
  <g class="nodes">
    <rect x="8" y="79" width="32" height="32" rx="8"/><rect x="280" y="79" width="32" height="32" rx="8"/>
    <rect x="96" y="19" width="32" height="32" rx="6"/><rect x="96" y="79" width="32" height="32" rx="6"/>
    <rect x="96" y="139" width="32" height="32" rx="6"/>
    <rect x="192" y="49" width="32" height="32" rx="6"/><rect x="192" y="109" width="32" height="32" rx="6"/>
  </g>
  <g class="node-detail">
    <path d="M18 90h12M18 95h12M18 100h12M290 90h12M290 95h12M290 100h12"/>
    <path d="M106 35h12m-6-6v12M106 95h12m-6-6v12M106 155h12m-6-6v12"/>
    <path d="M202 69l4-8 4 8h4M202 129l4-8 4 8h4"/>
  </g>
</svg></div>` : ''}
<h1 id="home-title">Explore your model</h1>
${models ? '<p class="guidance">Choose a model to see its layers and connections.</p><ul aria-label="Discovered models">' + models + '</ul>' : '<p class="guidance">' + guidance + '</p>'}
<div class="actions">
<button class="action" data-action="open" ${state.pending ? 'disabled' : ''}><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
  <rect x="3" y="9" width="6" height="6" rx="1.5"/><rect x="15" y="3" width="6" height="6" rx="1.5"/>
  <rect x="15" y="15" width="6" height="6" rx="1.5"/><path d="M9 12h3V6h3M12 12v6h3"/>
</svg>Open as Graph</button>
<button class="action secondary" data-action="pr" ${state.pending ? 'disabled' : ''}>Open PR</button>
${state.hasWorkspace && !models ? `<button class="action secondary" data-action="starter" ${state.pending ? 'disabled' : ''}>Try a starter model</button>
<p class="starter-hint">Create an MLP or small CNN, then choose Open as Graph.</p>` : ''}
${!state.hasWorkspace ? '<button class="action secondary" data-action="folder">Open Folder</button>' : ''}
</div>
${state.setup || state.pending || state.error || state.retryAvailable ? `<section class="setup" aria-label="tensorViz setup" aria-live="polite" aria-busy="${!!state.pending || state.setup?.phase === 'checking'}">
<strong>${state.pending ? '<span class="setup-progress" aria-hidden="true"></span>' + escapeHtml(state.pending)
  : state.error ? 'Setup needs attention' : state.setup?.phase === 'checking' ? '<span class="setup-progress" aria-hidden="true"></span>Checking tensorViz runtime…'
  : state.setup?.phase === 'ready' ? 'tensorViz is ready' : 'tensorViz setup'}</strong>
<p>${escapeHtml(state.error ?? state.setup?.detail ?? (state.retryAvailable ? 'Continue your selected model or review when setup is complete.' : 'Preparing tensorViz. Your selected action will continue when setup is ready.'))}</p>
${!state.pending ? `<div class="setup-controls">
${state.retryAvailable ? '<button data-action="retry">' + (state.error ? 'Retry opening' : 'Continue opening') + '</button>' : ''}
${state.error ? '<button data-action="log">Show log</button>' : ''}
${state.setup?.phase === 'untrusted' ? '<button data-action="trust">Manage Workspace Trust</button>' : `
${state.setup?.phase === 'needs_setup' ? '<button data-action="setup">Set up tensorViz</button>' : ''}
<button data-action="check">Check again</button>${['unavailable', 'needs_setup'].includes(state.setup?.phase ?? '') && !state.error ? '<button data-action="log">Show log</button>' : ''}`}
</div>` : ''}
</section>` : ''}
</section>
<footer><p>Runtime checks are automatic. Downloads and library changes ask for approval.</p>
</footer></main>`, script: `
document.addEventListener('click', event => {
  const button = event.target.closest('[data-action], [data-model]');
  if (!button || button.disabled) return;
  vscode.postMessage(button.dataset.model !== undefined
    ? { type: 'homeModel', index: Number(button.dataset.model) }
    : { type: 'homeAction', action: button.dataset.action });
});
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const updateHomeMotion = () => {
  document.body.dataset.motion = reduced.matches || document.hidden || document.getElementById('canvas-home').hidden ? 'paused' : 'playing';
};
reduced.addEventListener('change', updateHomeMotion);
document.addEventListener('visibilitychange', updateHomeMotion);
window.addEventListener('message', event => {
  const frame = document.getElementById('app');
  if (frame && event.source === frame.contentWindow) return;
  if (event.data?.type === 'canvasHomeState' && typeof event.data.html === 'string') {
    const home = document.getElementById('canvas-home');
    const action = home.contains(document.activeElement) ? document.activeElement.dataset.action : undefined;
    home.innerHTML = event.data.html;
    if (action) home.querySelector('[data-action="' + CSS.escape(action) + '"]:not(:disabled)')?.focus({ preventScroll: true });
    updateHomeMotion();
  }
  if (event.data?.type === 'canvasContent') {
    document.getElementById('canvas-home').hidden = !!event.data.visible;
    if (frame) { frame.hidden = !event.data.visible; frame.setAttribute('aria-hidden', String(!event.data.visible)); }
    updateHomeMotion();
  }
});
updateHomeMotion();
vscode.postMessage({ type: 'homeReady' });` };
}

export function canvasHomeHtml(state: CanvasHomeState): string {
  const nonce = randomUUID(), home = canvasHomeParts(state);
  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>tensorViz</title>
<style>${home.style}</style></head><body data-motion="paused"><div id="canvas-home" class="canvas-home">${home.body}</div>
<script nonce="${nonce}">const vscode = acquireVsCodeApi();${home.script}</script></body></html>`;
}
