/** Host-only preparation: Python owns detachment; raw retained evidence and the
 * launch credential never pass to the view. No model execution or source reads. */
import Ajv from 'ajv';
import { isDeepStrictEqual } from 'node:util';
import schema from '@tensorviz/contracts/semanticSchema.generated';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES } from '@tensorviz/contracts/protocol';

export type PortableCapture = components['schemas']['PortableCapture'];
export const PORTABLE_CAPTURE_MAX_BYTES = LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES;
const ajv = new Ajv({ strict: false, useDefaults: true });
ajv.addSchema(schema, 'semantic');
const structure = ajv.compile<PortableCapture>({ $ref: 'semantic#/definitions/PortableCapture' });
const captureStructure = ajv.compile<CaptureSnapshot>({ $ref: 'semantic#/definitions/CaptureSnapshot' });
export class PortableExportError extends Error {}

export async function preparePortableCapture(capture: CaptureSnapshot, options: {
  origin: string; token: string; signal: AbortSignal; fetchImpl?: typeof fetch;
}): Promise<PortableCapture> {
  const body = JSON.stringify(capture);
  if (Buffer.byteLength(body) > 16 * 1024 * 1024) throw new PortableExportError('Retained capture exceeds the portable preparation input limit. Nothing was truncated.');
  const controller = new AbortController();
  const abort = () => controller.abort();
  options.signal.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(abort, 10_000);
  try {
    if (options.signal.aborted) controller.abort();
    const response = await (options.fetchImpl ?? fetch)(`${options.origin}/api/ide/capture/portable`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': options.token },
      body, signal: controller.signal, redirect: 'error',
    });
    if (!response.ok) {
      await response.body?.cancel();
      if (response.status === 413) throw new PortableExportError('Retained capture exceeds the live-export limit. Nothing was truncated.');
      if (response.status === 400) throw new PortableExportError('Lossless export refused: capture data is invalid or contains a private/path-like literal.');
      throw new PortableExportError('The local backend could not prepare this capture. Reconnect and try again.');
    }
    const reader = response.body?.getReader();
    if (!reader) throw new PortableExportError('The local backend returned no portable capture.');
    const chunks: Uint8Array[] = []; let size = 0;
    try {
      for (;;) {
        const next = await reader.read();
        if (next.done) break;
        size += next.value.byteLength;
        if (size > PORTABLE_CAPTURE_MAX_BYTES) throw new PortableExportError('Retained capture exceeds the 16 MiB live-export limit. Nothing was truncated.');
        chunks.push(next.value);
      }
    } finally { await reader.cancel(); }
    const portable: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks)));
    if (!structure(portable)) throw new PortableExportError('The local backend returned invalid portable evidence.');
    // Every computational field must survive exactly. Only detached source and
    // its bindings may differ; the browser additionally validates their aliases.
    // Pydantic emits schema defaults omitted by older captures (for example a
    // null optional metric). Normalize a copy, never the retained ledger value.
    const normalized: unknown = structuredClone(capture);
    if (!captureStructure(normalized)) throw new PortableExportError('The retained capture has invalid evidence.');
    const { source: _source, sourceBindings: _bindings, ...original } = normalized;
    const { source, sourceBindings: _portableBindings, ...prepared } = portable.capture;
    if (!isDeepStrictEqual(original, prepared) || source.workspaceSessionId !== 'portable'
      || source.workspaceId !== 'portable' || source.modelId !== 'portable' || source.generation !== 0) {
      throw new PortableExportError('Portable preparation changed the captured computation. Lossless export refused.');
    }
    const aliases = new Map(capture.source.files.map((file, i) => [file.path, file.role === 'config' ? 'tensorviz.toml' : `file-${i}.py`]));
    const expectedSource = { ...capture.source, id: source.id, workspaceSessionId: 'portable', workspaceId: 'portable', modelId: 'portable',
      generation: 0, modelFile: aliases.get(capture.source.modelFile),
      files: capture.source.files.map(file => ({ ...file, path: aliases.get(file.path), documentVersion: null })) };
    const expectedBindings = capture.sourceBindings.map(binding => ({ ...binding, snapshotId: source.id,
      anchors: binding.anchors.map(anchor => ({ ...anchor, file: aliases.get(anchor.file) })) }));
    if (!/^portable:[0-9a-f]{64}$/.test(source.id) || !isDeepStrictEqual(source, expectedSource)
      || !isDeepStrictEqual(portable.capture.sourceBindings, expectedBindings)) {
      throw new PortableExportError('Portable preparation did not detach source identity. Export refused.');
    }
    if (controller.signal.aborted) throw new PortableExportError('Portable export was cancelled or timed out.');
    return portable;
  } catch (error) {
    if (controller.signal.aborted) throw new PortableExportError('Portable export was cancelled or timed out.');
    if (error instanceof PortableExportError) throw error;
    throw new PortableExportError('The local backend could not prepare valid portable evidence. Reconnect and try again.');
  } finally { clearTimeout(timer); options.signal.removeEventListener('abort', abort); }
}
