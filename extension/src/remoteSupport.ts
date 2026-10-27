/** Workspace services run beside the files. Only desktop SSH is supported remotely. */
export function unsupportedRemote(remoteName: string | undefined): string | undefined {
  if (!remoteName || remoteName === 'ssh-remote') return undefined;
  return `tensorViz supports local desktop VS Code and Remote SSH. Remote kind "${remoteName}" is not supported.`;
}

/** Only private desktop forwarding is supported. Never approve a public tunnel,
 * credentials, or a path-based proxy by reducing its URI to an origin. */
export function loopbackOrigin(value: string): string {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
    || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('tensorViz requires a private loopback HTTP endpoint for desktop SSH forwarding.');
  }
  return url.origin;
}
