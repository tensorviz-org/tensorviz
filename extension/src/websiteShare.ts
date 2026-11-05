/** Add to tensorViz: sends the Canvas's graph file to the member's tensorViz account
 * through the website's upload API, which keeps it as a private graph. The website then
 * opens its Share dialog, where the member decides who can see it and copies the link.
 *
 * The API (POST /api/graphs, the website's functions/index.js) is the one way graphs
 * reach tensorViz, from the website, this extension and the plugin alike: it checks the
 * file again, stores it, draws its card preview and keeps the 15-second posting clock.
 * Nothing here imports `vscode`.
 */
import { gzipSync } from 'node:zlib';
import { websiteRequest, WebsiteError, type WebsiteAccount, type WebsiteConfig, type WebsiteFetch } from './websiteAccount';

/** The API's limits, which are the website viewer's, checked here first for a quick answer. */
export const SHARE_LIMITS = { bytes: 10 * 1024 * 1024, nodes: 5000, title: 100, name: 120 };
/** Long enough for a 10 MB graph's upload on a slow connection. */
const UPLOAD_TIMEOUT_MS = 180_000;
const GRAPH_ID = /^[A-Za-z0-9]{20}$/;

export interface ShareableGraph {
  /** The file as the website stores it: the product's own graph file, unchanged. */
  text: string;
  /** The model's name, which titles the graph; its owner can rename it on the website. */
  title: string;
  name: string;
  nodes: number;
  bytes: number;
}

const oneLine = (value: string, max: number): string => value.replace(/\s+/g, ' ').trim().slice(0, max).trim();

/** What the website keeps about a graph file, or why the website cannot take it. */
export function shareableGraph(bytes: Uint8Array): ShareableGraph {
  if (bytes.length > SHARE_LIMITS.bytes) throw new WebsiteError('invalid', 'tensorViz takes graph files up to 10 MB.');
  const text = Buffer.from(bytes).toString('utf8');
  let file: any;
  try { file = JSON.parse(text); } catch { file = undefined; }
  const semantic = file?.document?.semantic;
  if (file?.format !== 'tensorviz.graph' || !Array.isArray(semantic?.nodes)) throw new WebsiteError('invalid', 'This is not a tensorViz graph file.');
  const nodes = semantic.nodes.length;
  if (nodes < 1 || nodes > SHARE_LIMITS.nodes) throw new WebsiteError('invalid', 'tensorViz takes graphs with 1 to 5,000 blocks.');
  const name = oneLine(typeof semantic.name === 'string' ? semantic.name : '', SHARE_LIMITS.name) || 'Graph';
  return { text, title: oneLine(name, SHARE_LIMITS.title), name, nodes, bytes: Buffer.byteLength(text, 'utf8') };
}

/** What the API takes: the file gzipped in base64, titled with the model's name, private. */
export function uploadBody(graph: ShareableGraph): string {
  return JSON.stringify({ file: gzipSync(Buffer.from(graph.text, 'utf8')).toString('base64'), title: graph.title, visibility: 'private' });
}

/** The graph's page on the website; `share` opens its Share dialog. */
export function graphUrl(config: WebsiteConfig, id: string, share = false): string {
  return `${config.site}/explore/graph/?id=${id}${share ? '&share=1' : ''}`;
}

/** Adds the graph as the signed-in member and returns its id. An ID token the API
 *  rejects is renewed once; callers sign the member in first. */
export async function shareGraph(account: WebsiteAccount, graph: ShareableGraph, fetcher: WebsiteFetch = fetch): Promise<string> {
  const config = account.config;
  const body = uploadBody(graph);
  for (let attempt = 0; ; attempt++) {
    const session = await account.session();
    if (!session) throw new WebsiteError('signed-out', 'Sign in to tensorViz to add graphs.');
    const { status, body: answer } = await websiteRequest(fetcher, `${config.site}/api/graphs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${session.idToken}` }, body,
    }, UPLOAD_TIMEOUT_MS);
    if (status === 201 && typeof answer?.id === 'string' && GRAPH_ID.test(answer.id)) return answer.id;
    if (status === 401 && attempt === 0) { account.forget(); continue; }
    // The API's own words, such as how many seconds the posting clock still needs.
    const message = typeof answer?.error?.message === 'string' ? answer.error.message : undefined;
    if (status === 429) throw new WebsiteError('too-soon', message ?? 'You posted to tensorViz a moment ago. Try again shortly.');
    if (status === 401) {
      await account.signOut();
      throw new WebsiteError('signed-out', 'Your tensorViz sign-in has expired. Sign in again.');
    }
    if (status === 400 || status === 413) throw new WebsiteError('invalid', message ?? 'tensorViz could not read this graph.');
    if (status === 403) throw new WebsiteError('denied', message ?? 'tensorViz refused this graph.');
    throw new WebsiteError('unavailable', 'tensorViz is not answering. Try again shortly.');
  }
}
