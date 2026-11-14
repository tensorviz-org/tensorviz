import fs from 'node:fs/promises';

// Fixture disposal is separate from the strict runtime-session lifecycle checks.
// Codex's optional marketplace clone may finish writing while its temporary
// home is removed. Retry transient filesystem races, never swallow exhaustion.
export async function removeUnixFixture(root, rm = fs.rm) {
  await rm(root, {recursive: true, force: true, maxRetries: 10, retryDelay: 100});
}
