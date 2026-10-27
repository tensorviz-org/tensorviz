import * as assert from 'node:assert/strict';
import { gunzipSync } from 'node:zlib';
import { WEBSITE, WebsiteError } from '../../websiteAccount';
import { graphUrl, shareGraph, shareableGraph, SHARE_LIMITS, uploadBody } from '../../websiteShare';
import { graphFile, setup, signIn } from '../websiteFixture';

const GRAPH_ID = 'AbCdEfGhIjKlMnOpQrSt';
const isUpload = (url: string) => url.endsWith('/api/graphs');

suite('Add to tensorViz', () => {
  test('reads what the website keeps about a graph file, within its limits', () => {
    const graph = shareableGraph(graphFile(3, '  Tiny\n Net '));
    assert.deepEqual([graph.title, graph.name, graph.nodes], ['Tiny Net', 'Tiny Net', 3]);
    assert.equal(graph.bytes, Buffer.byteLength(graph.text));
    assert.equal(shareableGraph(graphFile(1, 'x'.repeat(300))).title.length, 100);
    assert.equal(shareableGraph(graphFile(1, '')).title, 'Graph');
    for (const bytes of [Buffer.from('{}'), Buffer.from('not json'), graphFile(0), graphFile(SHARE_LIMITS.nodes + 1),
      Buffer.alloc(SHARE_LIMITS.bytes + 1, 32)]) {
      assert.throws(() => shareableGraph(bytes), (error: WebsiteError) => error.code === 'invalid');
    }
  });

  test('sends the upload API the file gzipped, titled with the model, private', () => {
    const graph = shareableGraph(graphFile(2, 'TinyNet'));
    const body = JSON.parse(uploadBody(graph));
    assert.deepEqual(Object.keys(body).sort(), ['file', 'title', 'visibility']);
    assert.deepEqual([body.title, body.visibility], ['TinyNet', 'private']);
    assert.equal(gunzipSync(Buffer.from(body.file, 'base64')).toString('utf8'), graph.text);
  });

  test('posts once with the member\'s ID token and links the Share dialog', async () => {
    const world = setup(url => isUpload(url) ? Response.json({ id: GRAPH_ID, thread: null }, { status: 201 }) : undefined);
    await signIn(world);
    const id = await shareGraph(world.account, shareableGraph(graphFile(2)), world.fetcher);
    assert.equal(id, GRAPH_ID);
    const posts = world.seen.filter(request => isUpload(request.url));
    assert.deepEqual(posts.map(request => request.url), ['https://tensorviz.dev/api/graphs']);
    assert.match((posts[0].init?.headers as Record<string, string>).Authorization, /^Bearer e30\./);
    assert.equal(graphUrl(WEBSITE, id, true), `https://tensorviz.dev/explore/graph/?id=${id}&share=1`);
  });

  test('passes on the API\'s refusals in its own words, renewing a rejected token once', async () => {
    let posts = 0;
    const world = setup(url => {
      if (!isUpload(url)) return undefined;
      posts++;
      return posts === 1 ? new Response('{}', { status: 401 })
        : Response.json({ error: { code: 'too-soon', message: 'You can post again in 9 seconds.', retryAfter: 9 } }, { status: 429 });
    });
    await signIn(world);
    await assert.rejects(shareGraph(world.account, shareableGraph(graphFile(2)), world.fetcher),
      (error: WebsiteError) => error.code === 'too-soon' && error.message === 'You can post again in 9 seconds.');
    assert.equal(posts, 2);
    assert.equal(world.seen.filter(request => request.url.includes('securetoken')).length, 1, 'renewed after the 401');

    const refused = setup(url => isUpload(url)
      ? Response.json({ error: { code: 'invalid', message: 'tensorViz takes graphs with 1 to 5,000 blocks.' } }, { status: 400 }) : undefined);
    await signIn(refused);
    await assert.rejects(shareGraph(refused.account, shareableGraph(graphFile(2)), refused.fetcher),
      (error: WebsiteError) => error.code === 'invalid' && /5,000 blocks/.test(error.message));
  });
});
