/** Saved model validation is enabled unless the user explicitly opts out. */
export function browserTraceSettings(read: (section: string) => unknown): { onSave: boolean; tier: 'real' | 'fake' | 'auto' } {
  const tier = read('traceTier');
  return { onSave: read('traceOnSave') !== false, tier: tier === 'fake' || tier === 'auto' ? tier : 'real' };
}
