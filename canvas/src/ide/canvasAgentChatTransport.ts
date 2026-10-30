import type { ChatTransport, UIMessage, UIMessageChunk } from 'ai';
import type { CanvasAgentAction, CanvasAgentSettings, CanvasAgentState } from './protocol.generated';

export type AgentUIMessage = UIMessage<{ settings?: CanvasAgentSettings }>;
export const agentUIMessages = (state: CanvasAgentState): AgentUIMessage[] => state.messages.map(message => ({
  id: message.id, role: message.role, parts: [{ type: 'text', text: message.text }], metadata: { settings: message.settings },
}));

/** AI SDK transport over the authenticated host bridge. The host remains the history authority. */
export class CanvasAgentChatTransport implements ChatTransport<AgentUIMessage> {
  private active?: { requestId: string; controller: ReadableStreamDefaultController<UIMessageChunk>; answerId?: string; text: string; cleanup(): void };
  constructor(private readonly send: (action: CanvasAgentAction) => boolean) {}
  get streaming(): boolean { return !!this.active; }

  accept(state: CanvasAgentState): void {
    const active = this.active;
    if (!active) return;
    const userIndex = state.messages.findIndex(message => message.id === active.requestId);
    if (userIndex === -1) {
      if (state.ack === active.requestId && state.status !== 'running') {
        active.controller.error(new Error(state.activity || 'The host could not accept this message.'));
        active.cleanup(); this.active = undefined;
      }
      return;
    }
    const answer = state.messages[userIndex + 1];
    if (answer?.role === 'assistant') {
      if (!active.answerId) {
        active.answerId = answer.id;
        active.controller.enqueue({ type: 'start', messageId: answer.id, messageMetadata: { settings: answer.settings } });
        active.controller.enqueue({ type: 'text-start', id: answer.id });
      }
      if (answer.text.startsWith(active.text) && answer.text.length > active.text.length) {
        active.controller.enqueue({ type: 'text-delta', id: answer.id, delta: answer.text.slice(active.text.length) });
        active.text = answer.text;
      }
    }
    if (state.status !== 'running') {
      if (active.answerId) active.controller.enqueue({ type: 'text-end', id: active.answerId });
      active.controller.enqueue({ type: 'finish' }); active.controller.close();
      active.cleanup(); this.active = undefined;
    }
  }

  async sendMessages(options: Parameters<ChatTransport<AgentUIMessage>['sendMessages']>[0]): Promise<ReadableStream<UIMessageChunk>> {
    const action = (options.body as { action?: CanvasAgentAction } | undefined)?.action;
    if (this.active || options.trigger !== 'submit-message' || action?.action !== 'send') throw new Error('Wait for the current reply before sending.');
    return new ReadableStream({
      start: controller => {
        const abort = () => {
          this.send({ action: 'stop', requestId: action.requestId });
          if (this.active?.controller === controller) { controller.close(); this.active.cleanup(); this.active = undefined; }
        };
        this.active = { requestId: action.requestId, controller, text: '',
          cleanup: () => options.abortSignal?.removeEventListener('abort', abort) };
        options.abortSignal?.addEventListener('abort', abort, { once: true });
        if (options.abortSignal?.aborted) { abort(); return; }
        if (!this.send(action)) {
          this.active.cleanup(); this.active = undefined;
          controller.error(new Error('Canvas is reconnecting. Your draft is saved; try again when connected.'));
        }
      },
      cancel: () => {
        this.send({ action: 'stop', requestId: action.requestId });
        this.active?.cleanup(); this.active = undefined;
      },
    });
  }

  async reconnectToStream(): Promise<null> { return null; } // Host snapshots restore missed text on reconnect.
  disconnect(): void {
    this.active?.controller.close(); this.active?.cleanup(); this.active = undefined;
  }
}
