// Check the deployable site, including source-edit and Markdown links.
import assert from 'node:assert/strict'
import { readFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.join(here, 'dist')
const generated = JSON.parse(await readFile(path.join(here, 'cache/generated.json'), 'utf8'))
const { base, sources } = generated
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
  // VitePress renders its dedicated not-found layout without an edit footer.
  if (output !== '404.md') {
    assert(html.includes(`https://github.com/tensorviz-org/tensorviz/edit/main/${source}`), `Missing edit link for ${source}`)
    editLinks++
  }
  assert((await stat(path.join(root, 'markdown', output))).isFile(), `Missing Markdown for ${source}`)
}
for (const name of ['index.html', 'quickstart.html', 'guides.html', 'install.html', 'agent-plugin.html', 'jax-support.html', 'graph-diff-review/index.html', 'llms.txt', 'licenses/geist-OFL.txt', 'licenses/geist-mono-OFL.txt']) {
  assert((await stat(path.join(root, name))).isFile(), `Missing docs resource: ${name}`)
}
const info = JSON.parse(await readFile(path.join(root, 'build-info.json'), 'utf8'))
assert.equal(info.source, 'https://github.com/tensorviz-org/tensorviz')
assert.equal(info.revision, generated.sha)
assert.equal(info.base, base)
console.log(`Verified ${pages.size} pages, ${links} local links/anchors, ${editLinks} source-edit links, ${Object.keys(sources).length} Markdown downloads and build provenance at ${base}.`)
