import { shapeIdToNodeId } from './canvasShapeId';

/**
 * Pick the node identity whose span represents a module definition.
 *
 * A module wrapper's own span is its instantiation in the parent file. The
 * imported-definition affordance therefore targets its first resolved member,
 * whose span belongs to the module's definition file. Empty/opaque modules have
 * no resolved member and correctly fall back to their wrapper declaration.
 */
export function moduleDefinitionNodeId(
  moduleMembers: readonly string[] | undefined,
  moduleInstanceId: string,
  shapeId: string,
): string {
  const resolvedMember = moduleMembers?.find((member) => typeof member === 'string' && member.length > 0);
  return resolvedMember || moduleInstanceId || shapeIdToNodeId(shapeId);
}
