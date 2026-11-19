/** Send feedback: a short note from the member to the tensorViz team, posted to the
 * team's feedback endpoint (FEEDBACK_URL). It goes only when the member asks: from the
 * Canvas's feedback button, or from the agent's send_feedback tool after the member
 * approves the exact text. Nothing is collected in the background.
 *
 * A report carries the member's words, its kind, where it was sent from, the tensorViz
 * version and the operating system, and, only when the member includes it, the open
 * model's class name and block count. Local paths are replaced before it leaves. A member
 * signed in on tensorviz.dev sends their ID token with it, of which only the account id is
 * kept, so the team can reply; sending never starts a sign-in.
 */
import os from 'node:os';
import { EMULATOR_VARIABLE, SITE_VARIABLE, websiteRequest, WebsiteError, type WebsiteAccount, type WebsiteFetch } from './websiteAccount';

export const FEEDBACK_KINDS = ['bug', 'idea', 'other'] as const;
export type FeedbackKind = typeof FEEDBACK_KINDS[number];
export const FEEDBACK_LIMITS = { message: 4000, model: 128 };
const TIMEOUT_MS = 20_000;
const FEEDBACK_ID = /^[A-Za-z0-9_-]{1,64}$/;

/** The tensorViz team's feedback endpoint. */
export const FEEDBACK_URL = 'https://feedback.tensorviz.dev/api/feedback';
/** Development: a feedback endpoint on this computer. */
export const FEEDBACK_URL_VARIABLE = 'TENSORVIZ_FEEDBACK_URL';

/** Where this host posts feedback: the team's endpoint or, for development, a loopback
 *  address. With a website development variable set, the loopback address is required, so
 *  neither development reports nor emulator sign-ins reach the team. */
export function feedbackUrl(env: Record<string, string | undefined> = process.env): string {
  const local = env[FEEDBACK_URL_VARIABLE];
  if (!local) {
    const development = [EMULATOR_VARIABLE, SITE_VARIABLE].find(name => env[name]);
    if (development) throw new WebsiteError('setup', `Set ${FEEDBACK_URL_VARIABLE} to a feedback endpoint on this computer while ${development} is set.`);
    return FEEDBACK_URL;
  }
  let url: URL | undefined;
  try { url = new URL(local); } catch { url = undefined; }
  if (!url || url.protocol !== 'http:' || !['localhost', '127.0.0.1'].includes(url.hostname) || url.search || url.hash || url.username || url.password) {
    throw new WebsiteError('setup', `${FEEDBACK_URL_VARIABLE} must be a feedback endpoint on this computer, such as http://127.0.0.1:5001/api/feedback.`);
  }
  return url.href;
}

export interface FeedbackGraph { model: string; nodes: number }

export interface FeedbackReport {
  kind: FeedbackKind;
  message: string;
  source: 'canvas' | 'agent';
  client: { product: 'extension' | 'plugin'; version: string; os: string; editor?: string };
  graph?: FeedbackGraph;
}

export interface FeedbackInput {
  kind: FeedbackKind;
  message: string;
  graph?: FeedbackGraph | null;
}

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Replaces the given folders, then any home folder, so a report never names local paths. */
export function redactPaths(text: string, roots: ReadonlyArray<string | undefined> = []): string {
  let result = text;
  const known = [...new Set(roots.filter((root): root is string => !!root && root.length > 1))].sort((a, b) => b.length - a.length);
  for (const root of known) result = result.replace(new RegExp(escape(root.replace(/[\\/]+$/, '')), 'g'), root === os.homedir() ? '~' : '<workspace>');
  return result
    .replace(/(?:\/Users|\/home)\/[^/\s'"`]+/g, '~')
    .replace(/[A-Za-z]:\\Users\\[^\\\s'"`]+/g, '~');
}

/** The report the member approves, checked and with paths replaced. */
export function feedbackReport(input: FeedbackInput, source: FeedbackReport['source'], client: FeedbackReport['client'],
  roots: ReadonlyArray<string | undefined> = []): FeedbackReport {
  if (!FEEDBACK_KINDS.includes(input.kind)) throw new WebsiteError('invalid', 'Choose bug, idea or other.');
  const message = redactPaths(input.message, [os.homedir(), ...roots]).trim();
  if (!message) throw new WebsiteError('invalid', 'Write a few words of feedback first.');
  if (message.length > FEEDBACK_LIMITS.message) throw new WebsiteError('invalid', `Keep feedback under ${FEEDBACK_LIMITS.message} characters.`);
  const graph = input.graph && /^[A-Za-z_]\w*$/.test(input.graph.model) && input.graph.model.length <= FEEDBACK_LIMITS.model
    && Number.isSafeInteger(input.graph.nodes) && input.graph.nodes >= 0 ? { model: input.graph.model, nodes: input.graph.nodes } : undefined;
  return { kind: input.kind, message, source, client, ...(graph ? { graph } : {}) };
}

/** What this computer is, in the words a report uses: the platform and its release. */
export function operatingSystem(): string {
  const name = ({ darwin: 'macOS', win32: 'Windows', linux: 'Linux' } as Record<string, string>)[process.platform] ?? process.platform;
  return `${name} ${os.release()} (${process.arch})`;
}

/** Posts the report, as the signed-in member when there is one. A rejected ID token
 *  sends it once more without one. Returns the id the team's endpoint gave the report. */
export async function sendFeedback(account: WebsiteAccount, report: FeedbackReport, fetcher: WebsiteFetch = fetch,
  env: Record<string, string | undefined> = process.env): Promise<string> {
  const url = feedbackUrl(env);
  const body = JSON.stringify(report);
  let session = await account.session().catch(() => undefined);
  for (let attempt = 0; ; attempt++) {
    const { status, body: answer } = await websiteRequest(fetcher, url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(session ? { Authorization: `Bearer ${session.idToken}` } : {}) },
      body,
    }, TIMEOUT_MS);
    if ((status === 200 || status === 201) && typeof answer?.id === 'string' && FEEDBACK_ID.test(answer.id)) return answer.id;
    if (status === 401 && session && attempt === 0) { account.forget(); session = undefined; continue; }
    const message = typeof answer?.error?.message === 'string' ? answer.error.message.slice(0, 300) : undefined;
    if (status === 429) throw new WebsiteError('too-soon', message ?? 'You sent feedback a moment ago. Try again shortly.');
    if (status === 400 || status === 413) throw new WebsiteError('invalid', message ?? 'tensorViz could not read this feedback.');
    if (status === 403) throw new WebsiteError('denied', message ?? 'tensorViz refused this feedback.');
    throw new WebsiteError('unavailable', 'tensorViz is not answering. Try again shortly.');
  }
}
