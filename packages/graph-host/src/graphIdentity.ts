import type { LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';

/** Full six-field identity equality: a page and this window are looking at the
 *  same graph state only when every dimension agrees — a rolled revision or
 *  digest is a different answer to "what is on screen". */
export function sameTargetIdentity(
  asked: LocalBrowserTargetIdentity,
  current: LocalBrowserTargetIdentity,
): boolean {
  return (
    asked.workspaceFolder === current.workspaceFolder &&
    asked.file === current.file &&
    asked.cls === current.cls &&
    asked.documentVersion === current.documentVersion &&
    asked.sourceDigest === current.sourceDigest &&
    asked.backendRevision === current.backendRevision
  );
}

