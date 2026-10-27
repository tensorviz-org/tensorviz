/**
 * Edited-at formatting for project lists.
 *
 * The design shows "14 minutes ago" for something touched this hour, then
 * "Yesterday, 6:02 PM", then "8 Aug 2026, 11:02 PM" — recency where recency is
 * the useful fact, and an absolute date once it stops being. A single
 * toLocaleString() everywhere (what the app did) makes a model edited two
 * minutes ago and one edited last March cost the same amount of reading.
 */

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

function timeOfDay(date: Date): string {
  return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}

function isSameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

/** `now` is injectable so the tests can assert boundaries instead of sleeping. */
export function formatEditedAt(value: string | number | Date, now: Date = new Date()): string {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown';

  const delta = now.getTime() - date.getTime();
  // A future timestamp means clock skew between this machine and the backend.
  // Saying "in 3 minutes" would read as a bug; the date itself still informs.
  if (delta < -MINUTE) return date.toLocaleString();

  if (delta < MINUTE) return 'Just now';
  if (delta < HOUR) {
    const mins = Math.max(1, Math.round(delta / MINUTE));
    return `${mins} minute${mins === 1 ? '' : 's'} ago`;
  }

  if (isSameDay(date, now)) return `Today, ${timeOfDay(date)}`;

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (isSameDay(date, yesterday)) return `Yesterday, ${timeOfDay(date)}`;

  const sameYear = date.getFullYear() === now.getFullYear();
  const day = date.toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'short',
    ...(sameYear ? {} : { year: 'numeric' }),
  });
  return `${day}, ${timeOfDay(date)}`;
}
