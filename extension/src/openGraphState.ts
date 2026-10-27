/** The last class graph this window showed, so Reload Window can open it again. */

export const OPEN_GRAPH_KEY = 'tensorviz.openGraph';

export interface OpenGraphRecord {
  workspaceFolder: string;
  file: string;
  cls: string;
}

function part(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 4096 && !value.includes('\0');
}

export function storedOpenGraph(value: unknown): OpenGraphRecord | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const record = value as OpenGraphRecord;
  if (!part(record.workspaceFolder) || !part(record.file) || !part(record.cls)) return undefined;
  if (record.file.split(/[/\\]/).some(segment => !segment || segment === '.' || segment === '..')) return undefined;
  return { workspaceFolder: record.workspaceFolder, file: record.file, cls: record.cls };
}

/** A stored class replaces an empty canvas. It never replaces a graph the user already has. */
export function graphToRestore(stored: OpenGraphRecord | undefined, hasSelection: boolean): OpenGraphRecord | undefined {
  return hasSelection ? undefined : stored;
}

/** Retired editor panels are not the live sidebar. Close the panel and reopen the stored graph there. */
export function retiredCanvasSerializer(restore: () => void): {
  deserializeWebviewPanel(panel: { dispose(): void }): Promise<void>;
} {
  return {
    async deserializeWebviewPanel(panel): Promise<void> {
      panel.dispose();
      restore();
    },
  };
}
