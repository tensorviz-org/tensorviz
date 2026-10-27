import React from 'react';
import { ProjectStatus } from '../../services/verificationRecord';

/**
 * A model's verification state, as a pill (over a thumbnail) or inline (in a
 * table cell).
 *
 * The label carries the meaning and the dot repeats it in colour, so the state
 * is never colour-only. 'never-run' and 'stale' are intentionally uncoloured:
 * only a real outcome earns a hue, and tinting "nobody has checked this yet" as
 * a warning would make every new model look broken.
 */

type StatusPillProps = {
  status: ProjectStatus;
  /** Failure count, so the label can say "3 errors" rather than just "Error". */
  errorCount?: number;
  variant?: 'pill' | 'inline';
  className?: string;
};

function label(status: ProjectStatus, errorCount?: number): string {
  switch (status) {
    case 'verified':
      return 'Verified';
    case 'error': {
      const n = errorCount && errorCount > 0 ? errorCount : 1;
      return `${n} error${n === 1 ? '' : 's'}`;
    }
    case 'stale':
      // Not "Verified": the pass predates the current revision, so nothing has
      // checked the model as it stands now.
      return 'Edited since verify';
    case 'never-run':
    default:
      return 'Never run';
  }
}

const TONE: Record<ProjectStatus, string> = {
  verified: 'verified',
  error: 'error',
  stale: 'neutral',
  'never-run': 'neutral',
};

const INLINE_COLOR: Record<ProjectStatus, string> = {
  verified: 'var(--ui2-green)',
  error: 'var(--ui2-red)',
  stale: 'var(--ui2-text-dim)',
  'never-run': 'var(--ui2-text-dim)',
};

export default function StatusPill({
  status,
  errorCount,
  variant = 'pill',
  className,
}: StatusPillProps) {
  const text = label(status, errorCount);
  if (variant === 'inline') {
    return (
      <span className={`ui2-status-inline ${className ?? ''}`} style={{ color: INLINE_COLOR[status] }}>
        <span aria-hidden="true" />
        {text}
      </span>
    );
  }
  return (
    <span className={`ui2-status-pill ui2-status-pill-${TONE[status]} ${className ?? ''}`}>
      <span aria-hidden="true" />
      {text}
    </span>
  );
}
