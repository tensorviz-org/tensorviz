import type { ModelNodeData } from '../types';
import { portLabel } from './capturePorts';

type OperationData = Pick<ModelNodeData, 'captureRef' | 'capturePorts' | 'operationGroup'>;

/** One row per captured tensor use, including repeated operands and nested/tuple values. */
export function operationTensorRows(data: OperationData) {
  if (data.operationGroup?.kind === 'slice') {
    const group = data.operationGroup;
    return [group.inputShape, group.outputShape].map((shape, index) => ({
      id: index === 0 ? `${group.members[0]}:in` : `${group.members[group.members.length - 1]}:out`,
      direction: index === 0 ? 'in' : 'out', label: index === 0 ? 'IN' : 'OUT',
      value: shape === undefined ? 'Not recorded' : `(${shape.join(', ')})`,
      title: index === 0 ? 'Input to the source expression' : 'Output of the source expression',
    }));
  }
  if (data.captureRef?.kind !== 'operation' || !data.capturePorts || data.operationGroup) return undefined;
  return ['in', 'out'].flatMap(direction => data.capturePorts!
    .filter(port => port.direction === direction && port.argumentKind === 'value')
    .map((port, index, side) => {
      // Operands are numbered from 0, as positional arguments are; the full
      // argument path stays in the row title and in captured details.
      const suffix = side.length > 1 ? ` ${index}` : '';
      return { id: port.id, direction, label: `${direction === 'in' ? 'IN' : 'OUT'}${suffix}`,
        value: port.shape === null ? 'Not recorded' : `(${port.shape.join(', ')})`,
        title: `${portLabel(port, index)} · ${port.dtype?.replace(/^torch\./, '') ?? 'dtype unknown'}` };
    }));
}
