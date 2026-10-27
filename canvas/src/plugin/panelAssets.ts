/** Serve only our bundled translations without a fetch connection in sandboxed MCP Apps. */
export function panelAssetFetch(files: Record<string, string>, fallback: typeof fetch): typeof fetch {
  const translations = new Set(Object.entries(files)
    .filter(([name]) => name.startsWith('translations/'))
    .map(([, url]) => url));
  const prefix = 'data:application/json;base64,';
  return async (input, init) => {
    if (typeof input !== 'string' || !translations.has(input) || !input.startsWith(prefix)
      || (init?.method && init.method.toUpperCase() !== 'GET')) return fallback(input, init);
    init?.signal?.throwIfAborted();
    const bytes = Uint8Array.from(atob(input.slice(prefix.length)), character => character.charCodeAt(0));
    return new Response(bytes, { headers: { 'Content-Type': 'application/json' } });
  };
}
