/** Saved model validation is enabled unless the user explicitly opts out. */
export function browserTraceSettings(read: (section: string) => unknown): { onSave: boolean; tier: 'real' | 'fake' } {
  return { onSave: read('traceOnSave') !== false, tier: read('traceTier') === 'fake' ? 'fake' : 'real' };
}
