import type { CapturePort, GraphIssue, ModelNodeData } from '../types';

export function nodeFailureLines(data: Pick<ModelNodeData, 'captureDiagnostic' | 'error'>): string[] {
  const diagnostic = data.captureDiagnostic;
  const shape = (dims: (number | null)[]) => `[${dims.map(dim => dim ?? '?').join(', ')}]`;
  if (diagnostic?.code === 'invalid_shape') return ['Incompatible shape',
    ...(diagnostic.expectedShape ? [`Expected ${shape(diagnostic.expectedShape)}`] : []),
    ...(diagnostic.actualShape ? [`Received ${shape(diagnostic.actualShape)}`] : [])];
  if (diagnostic?.code === 'invalid_dtype') return ['Incompatible dtype',
    ...(diagnostic.expectedDtype ? [`Expected ${diagnostic.expectedDtype}`] : []),
    ...(diagnostic.actualDtype ? [`Received ${diagnostic.actualDtype}`] : [])];
  return data.error ? [diagnostic?.code === 'assertion_failed' ? 'Forward assertion failed' : data.error] : [];
}

/** The diagnostic field addresses operation operands, not an ancestor's inputs. */
export function isFailingPort(data: ModelNodeData, port: CapturePort): boolean {
  const field = data.captureDiagnostic?.field;
  return data.debugEvidence !== 'previous' && data.captureRef?.kind === 'operation' && port.direction === 'in'
    && !!field?.length && port.paths.some(path => JSON.stringify(path) === JSON.stringify(['arguments', ...field]));
}

export function ownGraphIssues(nodeId: string, data: ModelNodeData, includeStatus = true): GraphIssue[] {
  if (data.debugEvidence === 'previous') return [];
  const label = data.captureLabel || data.layerType?.displayName || nodeId;
  const issue = (id: string, severity: GraphIssue['severity'], message: string): GraphIssue => ({ id, nodeId, label, severity, message });
  const result: GraphIssue[] = [];
  const diagnostic = data.captureDiagnostic;
  if (diagnostic && data.captureRef) {
    const description: Record<string, string> = { invalid_dtype: 'Incompatible tensor dtype', invalid_shape: 'Incompatible tensor shape',
      assertion_failed: 'Model assertion failed', execution_failed: 'Execution failed' };
    const shape = (dims: (number | null)[]) => `[${dims.map(dim => dim ?? '?').join(', ')}]`;
    const facts = [diagnostic.field.length ? `Input: ${diagnostic.field.join('.')}` : '',
      diagnostic.actualDtype ? `Actual dtype: ${diagnostic.actualDtype}` : '',
      diagnostic.expectedDtype ? `Expected dtype: ${diagnostic.expectedDtype}` : '',
      diagnostic.actualShape ? `Actual shape: ${shape(diagnostic.actualShape)}` : '',
      diagnostic.expectedShape ? `Expected shape: ${shape(diagnostic.expectedShape)}` : ''].filter(Boolean);
    result.push(issue(`capture:${data.captureRef.captureId}:${diagnostic.id}`, 'error',
      [description[diagnostic.code] || diagnostic.code.replace(/_/g, ' '), ...facts].join('\n')));
  } else if (data.error) result.push(issue(`node:${nodeId}:error`, 'error', data.error));
  if (data.resolutionSeverity === 'error' || data.resolutionSeverity === 'warning') {
    result.push(issue(`node:${nodeId}:resolution`, data.resolutionSeverity, data.resolutionDetail || 'Source resolution is incomplete.'));
  }
  if (includeStatus && !result.length && (data.moduleStatus === 'error' || data.moduleStatus === 'warning')) {
    result.push(issue(`node:${nodeId}:status`, data.moduleStatus, 'This module has a recorded issue. No detailed diagnostic is available.'));
  }
  return result;
}

/** Preserve distinct failures; deduplicate only the same recorded diagnostic. */
export function mergeGraphIssues(descendants: GraphIssue[], own: GraphIssue[]): GraphIssue[] {
  const issues = new Map<string, GraphIssue>();
  for (const issue of [...descendants, ...own]) if (!issues.has(issue.id)) issues.set(issue.id, issue);
  return Array.from(issues.values());
}

export function graphIssueLabel(issues: GraphIssue[]): string {
  const errors = issues.filter(issue => issue.severity === 'error').length;
  const warnings = issues.length - errors;
  return [errors ? `${errors} error${errors === 1 ? '' : 's'}` : '',
    warnings ? `${warnings} warning${warnings === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ');
}
