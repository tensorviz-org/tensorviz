export const checkpointFixture = () => {
  const registered = { kind: 'parameter', shape: [2, 3], dtype: 'torch.float32', objectGroup: 0,
    storageGroup: 0, storageOffset: 0, stride: [3, 1], unavailable: null };
  return { version: 1, sourceDigest: 'a'.repeat(64), checkpoint: { file: 'weights.pt', digest: 'b'.repeat(64), strict: true },
    scope: 'registered_state_before_load', loaderSemantics: 'not_inferred', loadStatus: 'failed', complete: true,
    registeredComplete: true, storedComplete: true, omissions: [],
    entries: [{ name: 'fc.weight', registered, stored: { ...registered, kind: 'stored', shape: [2, 4], stride: [4, 1] },
      presence: 'both', shape: 'different', dtype: 'same' }] };
};
