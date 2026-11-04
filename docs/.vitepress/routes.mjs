// Shared route helpers (source path in the repo -> site route).
export const REPO = 'tensorviz-org/tensorviz'

export const outPath = (p) => (/(^|\/)README\.md$/.test(p) && p !== 'README.md' ? p.replace(/README\.md$/, 'index.md') : p)

// GitHub Pages serves the file it was given. Extensionless routes 404 there,
// so pages are `/docs/page.html`. A copied README is `index.md`, which Pages
// serves at the directory URL.
export const route = (p) => {
  const o = outPath(p)
  if (o === 'index.md') return '/'
  if (o.endsWith('/index.md')) return '/' + o.slice(0, -'index.md'.length)
  if (o.endsWith('.md')) return '/' + o.slice(0, -'.md'.length) + '.html'
  return '/' + o
}
