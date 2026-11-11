import type { LocalBrowserTargetIdentity } from './protocol.generated';

/** Mirrors the graph service's revision identity; interpretation is never portable graph data. */
export async function nodeExplanationRevision(target: LocalBrowserTargetIdentity, projectionDigest?: string): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify([
    target.documentVersion, target.sourceDigest, target.backendRevision, projectionDigest,
  ]));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
