// Check the deployable site, including source-edit and Markdown links.
import assert from 'node:assert/strict'
import { readFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.join(here, 'dist')
const generated = JSON.parse(await readFile(path.join(here, 'cache/generated.json'), 'utf8'))
const { base, sources } = generated
const siteUrl = `https://docs.tensorviz.dev${base}`
const pages = new Map()
async function collect(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name)
    if (entry.isDirectory()) await collect(file)
    else if (entry.name.endsWith('.html')) pages.set(file, await readFile(file, 'utf8'))
  }
}
await collect(root)
let links = 0
for (const [file, html] of pages) {
  const pageUrl = new URL(base + path.relative(root, file).split(path.sep).join('/'), 'https://tensorviz-docs.invalid')
  for (const match of html.matchAll(/\b(?:href|src)="([^"\s]+)"/g)) {
    const url = new URL(match[1].replaceAll('&amp;', '&'), pageUrl)
    if (url.origin !== pageUrl.origin) continue
    assert(url.pathname.startsWith(base), `Link escapes ${base}: ${match[1]}`)
    let target = path.resolve(root, decodeURIComponent(url.pathname.slice(base.length)))
    assert(target === root || target.startsWith(root + path.sep), `Unsafe path: ${match[1]}`)
    if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html')
    assert((await stat(target)).isFile(), `Missing ${target}`)
    if (url.hash && pages.has(target)) {
      const anchor = decodeURIComponent(url.hash.slice(1))
      assert(pages.get(target).includes(`id="${anchor}"`), `Missing #${anchor} in ${target}`)
    }
    links++
  }
}
let editLinks = 0
for (const [output, source] of Object.entries(sources)) {
  const html = pages.get(path.join(root, output.replace(/\.md$/, '.html')))
  assert(html, `Missing page for ${source}`)
  const head = html.slice(0, html.indexOf('</head>'))
  const tags = [...head.matchAll(/<(?:link|meta)\b[^>]*>/g)].map(match =>
    Object.fromEntries([...match[0].matchAll(/([\w:-]+)="([^"]*)"/g)].map(([, key, value]) => [key, value])))
  const tag = (key, value) => {
    const matches = tags.filter(attrs => attrs[key] === value)
    assert.equal(matches.length, 1, `Expected one ${value} tag for ${source}`)
    return matches[0]
  }
  const url = siteUrl + output.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html')
  assert.equal(tag('rel', 'canonical').href, url, `Incorrect canonical URL for ${source}`)
  assert.equal(tag('property', 'og:url').content, url, `Incorrect Open Graph URL for ${source}`)
  assert(tag('property', 'og:title').content, `Missing Open Graph title for ${source}`)
  assert.equal(tag('property', 'og:description').content, tag('name', 'description').content, `Incorrect Open Graph description for ${source}`)
  assert.equal(tag('property', 'og:site_name').content, 'TensorViz')
  assert.equal(tag('name', 'twitter:card').content, 'summary')
  if (output === '404.md') assert.equal(tag('name', 'robots').content, 'noindex')
  // VitePress renders its dedicated not-found layout without an edit footer.
  if (output !== '404.md') {
    assert(html.includes(`https://github.com/tensorviz-org/tensorviz/edit/main/${source}`), `Missing edit link for ${source}`)
    editLinks++
  }
  assert((await stat(path.join(root, 'markdown', output))).isFile(), `Missing Markdown for ${source}`)
}
for (const name of ['index.html', 'quickstart.html', 'guides.html', 'install.html', 'agent-plugin.html', 'jax-support.html', 'graph-diff-review/index.html', 'llms.txt', 'robots.txt', 'sitemap.xml', 'licenses/geist-OFL.txt', 'licenses/geist-mono-OFL.txt']) {
  assert((await stat(path.join(root, name))).isFile(), `Missing docs resource: ${name}`)
}
const sitemap = await readFile(path.join(root, 'sitemap.xml'), 'utf8')
const sitemapUrls = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => match[1])
const expectedUrls = Object.keys(sources).filter(output => output !== '404.md').map(output =>
  siteUrl + output.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html'))
assert.deepEqual(sitemapUrls.toSorted(), expectedUrls.toSorted(), 'Sitemap must list each documentation page once, excluding 404 and static resources')
assert(sitemapUrls.includes(`${siteUrl}quickstart.html`), 'Missing quickstart URL in sitemap')
const robots = await readFile(path.join(root, 'robots.txt'), 'utf8')
const [allowGroup, sitemapDirective] = robots.trim().split(/\n\s*\n/)
const agents = [...allowGroup.matchAll(/^User-agent: (.+)$/gm)].map(match => match[1])
assert.deepEqual(agents, ['*', 'GPTBot', 'ChatGPT-User', 'OAI-SearchBot', 'ClaudeBot', 'Claude-User', 'Claude-SearchBot', 'anthropic-ai', 'PerplexityBot', 'Perplexity-User', 'Google-Extended', 'Applebot-Extended', 'Bingbot', 'CCBot'])
assert(allowGroup.endsWith('Allow: /'), 'All crawlers must share the allow group')
assert.equal(sitemapDirective, 'Sitemap: https://docs.tensorviz.dev/sitemap.xml')
const info = JSON.parse(await readFile(path.join(root, 'build-info.json'), 'utf8'))
assert.equal(info.source, 'https://github.com/tensorviz-org/tensorviz')
assert.equal(info.revision, generated.sha)
assert.equal(info.base, base)
console.log(`Verified ${pages.size} pages, ${links} local links/anchors, ${editLinks} source-edit links, ${Object.keys(sources).length} Markdown downloads, page metadata, robots, sitemap and build provenance at ${base}.`)
