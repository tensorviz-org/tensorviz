/**
 * The builder command channel.
 *
 * A window CustomEvent rather than a prop, because the senders and the handler
 * have no common ancestor that could carry one: ModelBuilder listens, and the
 * toolbar (BuilderPageV2) and the on-node parameter editor (a component mounted
 * inside tldraw's own overlay slot) both send. It was defined inside
 * BuilderPageV2 until the parameter editor moved onto the node; importing it
 * from there would have pulled the whole page — and tldraw with it — into the
 * canvas module graph.
 */
export type BuilderCommandPayload = Record<string, unknown>;

export function dispatchBuilderCommand(command: string, payload?: BuilderCommandPayload): void {
  window.dispatchEvent(new CustomEvent('builder-command', { detail: { command, payload } }));
}
