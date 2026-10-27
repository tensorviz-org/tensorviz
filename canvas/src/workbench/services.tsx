import { createContext, useContext, type ComponentType } from 'react';
import type { PersistGraph } from '../services/revisionedGraphSaveQueue';
import type { VerificationRecord } from '../services/verificationRecord';
import type { ModelGraph } from '../types';

/** Optional hosted capabilities. Local Canvas entries have no account or project store. */
export interface WorkbenchServices {
  pathname: string;
  navigate: (path: string) => void;
  AccountMenu: ComponentType<{ placement?: 'rail' }>;
  Environment: ComponentType;
  HistoryPanel: ComponentType<{ open: boolean; onClose: () => void }>;
  persistGraph: PersistGraph;
  saveUserTemplate: (name: string, graph: ModelGraph) => Promise<unknown>;
  recordVerification: (graphId: string, record: VerificationRecord) => void;
}

export const WorkbenchServicesContext = createContext<WorkbenchServices | null>(null);
export const useWorkbenchServices = () => useContext(WorkbenchServicesContext);
