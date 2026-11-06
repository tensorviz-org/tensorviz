// Shared route helpers (source path in the repo -> site route).
export const REPO = 'tensorviz-org/tensorviz'

export const outPath = (p) => {
  if (p === 'docs/README.md') return 'guides.md'
  const rel = p.startsWith('docs/') ? p.slice(5) : p
  return /(^|\/)README\.md$/.test(rel) ? rel.replace(/README\.md$/, 'index.md') : rel
}

// Keep explicit article filenames. A copied README becomes index.md and is
// served at its directory URL.
export const route = (p) => {
  const o = outPath(p)
  if (o === 'index.md') return '/'
  if (o.endsWith('/index.md')) return '/' + o.slice(0, -'index.md'.length)
  if (o.endsWith('.md')) return '/' + o.slice(0, -'.md'.length) + '.html'
  return '/' + o
}
