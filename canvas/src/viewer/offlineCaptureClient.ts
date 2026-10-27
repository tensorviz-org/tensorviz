import type { components } from '../types/semanticContracts.generated';
import type { LocalBrowserCaptureClient } from '../ide/localBrowserCaptureClient';
import { CaptureReadError, inspectCapture } from '../ide/captureReader.generated';

/** Only accepts capture records already validated by the graph-file importer.
 * No network, source navigation, Python execution or editing capability. */
export function createOfflineCaptureClient(capture: components['schemas']['CaptureSnapshot']): LocalBrowserCaptureClient {
  let disposed = false;
  return {
    async inspect(query) {
      if (disposed) return { page: null, detail: 'This graph file has been closed.' };
      try { return { page: inspectCapture(capture, query), detail: 'Saved capture evidence.' }; }
      catch (error) { return { page: null, detail: error instanceof CaptureReadError ? error.message : 'This saved capture could not be inspected.' }; }
    },
    cancel() { /* Reads are synchronous; the dialog owns asynchronous result cancellation. */ },
    dispose() { disposed = true; },
  };
}
