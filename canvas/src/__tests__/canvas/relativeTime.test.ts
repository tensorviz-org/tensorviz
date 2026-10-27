import { formatEditedAt } from '../../utils/relativeTime';

/**
 * The project lists show when a model was last edited. A single
 * toLocaleString() everywhere — what the app did — makes a model touched two
 * minutes ago cost exactly as much reading as one touched last March, which is
 * the opposite of what a "Continue" list is for.
 *
 * `now` is injected so these assert boundaries instead of sleeping.
 */

const NOW = new Date('2026-08-11T14:30:00');

function ago(ms: number): Date {
  return new Date(NOW.getTime() - ms);
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

it('says "Just now" under a minute', () => {
  expect(formatEditedAt(ago(20_000), NOW)).toBe('Just now');
});

it('counts minutes up to the hour boundary', () => {
  expect(formatEditedAt(ago(14 * MINUTE), NOW)).toBe('14 minutes ago');
  expect(formatEditedAt(ago(MINUTE), NOW)).toBe('1 minute ago');
  expect(formatEditedAt(ago(59 * MINUTE), NOW)).toBe('59 minutes ago');
});

it('switches to a clock time once past an hour but still today', () => {
  // Past an hour, "127 minutes ago" is arithmetic the reader has to do; the
  // clock time is not.
  expect(formatEditedAt(ago(2 * HOUR), NOW)).toMatch(/^Today, /);
});

it('names yesterday rather than counting hours across midnight', () => {
  expect(formatEditedAt(new Date('2026-08-10T18:02:00'), NOW)).toMatch(/^Yesterday, /);
});

it('gives an absolute date beyond yesterday, without the year when it is this year', () => {
  // Day/month ORDER is the locale's business, so this asserts the parts are
  // present and that the redundant current year is not.
  const out = formatEditedAt(new Date('2026-08-08T23:02:00'), NOW);
  expect(out).toMatch(/Aug/);
  expect(out).toMatch(/\b8\b/);
  expect(out).not.toMatch(/2026/);
});

it('includes the year once the date is in a different one', () => {
  expect(formatEditedAt(new Date('2025-07-28T15:40:00'), NOW)).toMatch(/2025/);
});

it('falls back to an absolute date for a future timestamp instead of counting backwards', () => {
  // Clock skew between this machine and the backend is real; "in 3 minutes"
  // would read as a bug in the app.
  const out = formatEditedAt(new Date(NOW.getTime() + 10 * MINUTE), NOW);
  expect(out).not.toMatch(/ago/);
  expect(out).not.toBe('Just now');
});

it('says Unknown for an unparseable value rather than "Invalid Date"', () => {
  expect(formatEditedAt('not a date', NOW)).toBe('Unknown');
});
