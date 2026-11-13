import { isCanvasAgentSettings, type CanvasAgentProvider, type CanvasAgentSelections, type CanvasAgentSettings } from './protocol.generated';

const KEY = 'tensorviz.explanation.settings.v1';

/** Only model IDs and effort preferences. Never provider credentials or graph content. */
export function readExplanationSettings(): CanvasAgentSelections {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(KEY) ?? '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([provider, settings]) =>
      (provider === 'codex' || provider === 'claude') && isCanvasAgentSettings(settings)));
  } catch { return {}; }
}

export function saveExplanationSettings(previous: CanvasAgentSelections, provider: CanvasAgentProvider,
  settings: CanvasAgentSettings | undefined): CanvasAgentSelections {
  const next = { ...previous };
  if (settings === undefined) delete next[provider];
  else if (isCanvasAgentSettings(settings)) next[provider] = { ...settings };
  try { window.localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* Keep the choice for this visit. */ }
  return next;
}
