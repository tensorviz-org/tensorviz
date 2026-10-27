import { GraphMcpServer } from './mcpServer';

/** An explicit, window-owned desktop endpoint. The shared remote-agent endpoint
 * keeps its existing ownership/takeover contract and never acquires desktop origins. */
export class McpForwarding {
  private endpoint?: GraphMcpServer;
  private generation = 0;
  private address?: string;
  constructor(private readonly create: () => GraphMcpServer,
    private readonly resolveOrigin: (origin: string) => Promise<string>) {}

  get url(): string | undefined { return this.address; }
  get sessionCount(): number { return this.endpoint?.sessionCount ?? 0; }

  async open(): Promise<string> {
    const generation = ++this.generation;
    const previous = this.endpoint;
    this.endpoint = undefined;
    this.address = undefined;
    await previous?.close();
    const current = () => generation === this.generation;
    if (!current()) throw new Error('MCP forwarding was cancelled.');
    const endpoint = this.create();
    this.endpoint = endpoint;
    try {
      const internal = await endpoint.start();
      if (!current()) throw new Error('MCP forwarding was cancelled.');
      const origin = await this.resolveOrigin(new URL(internal.url).origin);
      if (!current()) throw new Error('MCP forwarding was cancelled.');
      endpoint.allowForwardedOrigin(origin);
      return this.address = `${origin}/mcp`;
    } catch (error) {
      await endpoint.close();
      if (this.endpoint === endpoint) this.endpoint = undefined;
      throw error;
    }
  }

  close(): Promise<void> {
    this.generation++;
    const endpoint = this.endpoint;
    this.endpoint = undefined;
    this.address = undefined;
    return endpoint?.close() ?? Promise.resolve();
  }
}
