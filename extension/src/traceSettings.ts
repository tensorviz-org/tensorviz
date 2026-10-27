/** How the sidebar decides whether a save may execute the model.

`tensorviz.traceOnSave` remains in the manifest so an old workspace value still
loads, and its deprecation text says a save does not execute. This reader never
turns that value into a capture. */
export function browserTraceSettings(read: (section: string) => unknown): { onSave: false; tier: 'real' | 'fake' } {
  return { onSave: false, tier: read('traceTier') === 'fake' ? 'fake' : 'real' };
}
