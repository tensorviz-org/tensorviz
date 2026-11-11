import { createHash, randomUUID } from 'node:crypto';
import { MODEL_CATALOG_LIMIT, isModelCatalogEntry, type ModelCatalog, type ModelCatalogEntry } from '@tensorviz/contracts/modelCatalog';
import type { LocalBrowserModelCatalogRequest, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { sameTargetIdentity } from '@tensorviz/graph-host/graphIdentity';

export interface IndexedModelClass { cls: string; file: string }
export interface CanvasModelCatalogOptions {
  target(): LocalBrowserTargetIdentity | undefined;
  connected(viewId: string): boolean;
  reviewing(): boolean;
  captured(target: LocalBrowserTargetIdentity): boolean;
  classes(folder: string, refresh: boolean): Promise<readonly IndexedModelClass[]>;
  select(model: IndexedModelClass, folder: string, native: boolean, current: () => boolean): Promise<void>;
  publish(message: LocalBrowserHostPublish): unknown;
  timeoutMs?: number;
}

/** Source-index discovery and opaque selection handles. No model imports or runs. */
export class CanvasModelCatalog {
  private readonly salt = randomUUID();
  private readonly openedIds = new Set<string>();
  private readonly listed = new Map<string, Map<string, IndexedModelClass>>();
  private readonly requests = new Map<string, symbol>();
  private disposed = false;
  constructor(private readonly options: CanvasModelCatalogOptions) {}

  private id(folder: string, model: IndexedModelClass): string {
    return `model-${createHash('sha256').update(JSON.stringify([this.salt, folder, model.file, model.cls])).digest('hex')}`;
  }

  opened(target: LocalBrowserTargetIdentity): void {
    this.openedIds.add(this.id(target.workspaceFolder, target));
  }

  private async classes(folder: string, refresh: boolean): Promise<readonly IndexedModelClass[]> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([this.options.classes(folder, refresh), new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('Project model discovery is taking longer than expected. Try again.')), this.options.timeoutMs ?? 10_000);
      })]);
    } finally { if (timer) clearTimeout(timer); }
  }

  private catalog(target: LocalBrowserTargetIdentity, classes: readonly IndexedModelClass[]): ModelCatalog {
    const unique = new Map<string, ModelCatalogEntry>();
    for (const candidate of classes) {
      const model = { cls: candidate.cls, file: candidate.file.replace(/\\/g, '/') };
      const id = this.id(target.workspaceFolder, model);
      const active = model.file === target.file && model.cls === target.cls;
      const entry: ModelCatalogEntry = { ...model, id, evidence: active && this.options.captured(target) ? 'captured' : 'source',
        opened: active || this.openedIds.has(id) };
      if (isModelCatalogEntry(entry)) unique.set(id, entry);
    }
    const activeId = this.id(target.workspaceFolder, target);
    const all = [...unique.values()].sort((a, b) => a.cls.localeCompare(b.cls) || a.file.localeCompare(b.file));
    // Keep the displayed model in the bounded list even in very large projects.
    const active = unique.get(activeId);
    const models = all.length > MODEL_CATALOG_LIMIT && active
      ? [active, ...all.filter(model => model.id !== activeId)].slice(0, MODEL_CATALOG_LIMIT) : all.slice(0, MODEL_CATALOG_LIMIT);
    this.listed.set(target.workspaceFolder, new Map(models.map(model => [model.id, { cls: model.cls, file: model.file }])));
    return { status: 'ready', models, ...(models.some(model => model.id === activeId) ? { activeId } : {}),
      ...(all.length > MODEL_CATALOG_LIMIT ? { truncated: true } : {}) };
  }

  async request(request: LocalBrowserModelCatalogRequest, native: boolean): Promise<void> {
    const token = Symbol(request.requestId);
    const current = () => !this.disposed && this.requests.get(request.viewId) === token
      && this.options.connected(request.viewId) && !!this.options.target()
      && sameTargetIdentity(this.options.target()!, request.target);
    if (this.disposed || !this.options.connected(request.viewId) || !this.options.target()
      || !sameTargetIdentity(this.options.target()!, request.target)) return;
    this.requests.set(request.viewId, token);
    const publish = (catalog: ModelCatalog) => {
      if (current()) this.options.publish({ type: 'modelCatalog', viewId: request.viewId, target: request.target, requestId: request.requestId, catalog });
    };
    try {
      if (request.action === 'select') {
        const selected = this.listed.get(request.target.workspaceFolder)?.get(request.modelId);
        if (!selected) throw new Error('This model is no longer listed. Refresh the model list and try again.');
        if (this.options.reviewing()) throw new Error('Apply or discard the current source proposal before switching models.');
        publish({ status: 'loading', models: [] });
        const classes = await this.classes(request.target.workspaceFolder, true);
        if (!current()) return;
        if (this.options.reviewing()) throw new Error('Apply or discard the current source proposal before switching models.');
        if (!classes.some(model => model.cls === selected.cls && model.file.replace(/\\/g, '/') === selected.file)) {
          throw new Error('This model class is no longer in the project. Refresh the model list.');
        }
        await this.options.select(selected, request.target.workspaceFolder, native, () => current() && !this.options.reviewing());
        if (current() && (selected.file !== request.target.file || selected.cls !== request.target.cls)) {
          throw new Error('This model could not be opened. Check the tensorViz output and try again.');
        }
        if (current()) publish(this.catalog(request.target, classes));
      } else {
        publish({ status: 'loading', models: [] });
        const classes = await this.classes(request.target.workspaceFolder, false);
        if (current()) publish(this.catalog(request.target, classes));
      }
    } catch (error) {
      const detail = error instanceof Error && error.message.trim() === error.message && error.message.length > 0 && error.message.length <= 320
        && Array.from(error.message).every(character => character.charCodeAt(0) >= 32 && character.charCodeAt(0) !== 127)
        ? error.message : 'Project model discovery is unavailable. Refresh the model list and try again.';
      publish({ status: 'unavailable', models: [], detail });
    } finally { if (this.requests.get(request.viewId) === token) this.requests.delete(request.viewId); }
  }

  dispose(): void { this.disposed = true; this.requests.clear(); this.listed.clear(); this.openedIds.clear(); }
}
