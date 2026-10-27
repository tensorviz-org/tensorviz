// Presentation-only content layer: copies the repo docs (unchanged text) into the
// VitePress root, rewrites links to site routes, and generates the landing page.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { REPO, outPath, route } from './routes.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(here, '..')
const OUT = path.join(here, 'site')
const GH = `https://github.com/${REPO}`
const SKIP_DIRS = new Set(['node_modules', '.git', 'docs-site', 'dist'])

function git(args) {
  return execFileSync('git', args, { cwd: SRC, encoding: 'utf8' }).trim()
}

const commit = git(['rev-parse', 'HEAD'])

function resolveRef() {
  const explicit = process.env.DOCS_SHA?.trim()
  if (explicit) return explicit
  let branch = ''
  try {
    branch = git(['rev-parse', '--abbrev-ref', 'HEAD'])
  } catch {
    branch = ''
  }
  // Actions checks out a detached commit. GITHUB_REF is still refs/heads/main
  // on a main build, and linking the branch keeps Edit on GitHub current.
  if (branch === 'main' || process.env.GITHUB_REF === 'refs/heads/main') return 'main'
  return commit
}

const REF = resolveRef()
if (!/^[\w./+-]+$/.test(REF)) {
  throw new Error(`Refused to embed ref ${JSON.stringify(REF)} in a GitHub URL`)
}

function normalizeBase(value) {
  const raw = (value ?? '').trim()
  if (!raw || raw === '/') return '/'
  let base = raw.startsWith('/') ? raw : `/${raw}`
  if (!base.endsWith('/')) base += '/'
  return base
}

// DOCS_BASE is the site prefix (`/` locally, `/tensorviz/` on GitHub Pages).
const base = normalizeBase(process.env.DOCS_BASE)

// Absolute site paths. VitePress prefixes `base` on Markdown links itself;
// call withBase only for HTML that VitePress does not rewrite.
function withBase(url) {
  if (!url || !url.startsWith('/')) return url
  if (base === '/') return url
  if (url === base || url.startsWith(base)) return url
  return `${base}${url.slice(1)}`
}

// 1. wipe generated content (keep the theme, logos, and the 404 page)
for (const e of fs.readdirSync(OUT)) {
  if (['.vitepress', 'public', '404.md'].includes(e)) continue
  fs.rmSync(path.join(OUT, e), { recursive: true, force: true })
}
fs.rmSync(path.join(OUT, '.vitepress/dist'), { recursive: true, force: true })
fs.rmSync(path.join(OUT, '.vitepress/cache'), { recursive: true, force: true })
fs.rmSync(path.join(OUT, 'public/raw'), { recursive: true, force: true })
fs.mkdirSync(path.join(OUT, 'public/raw'), { recursive: true })

function walk(dir, pred, acc = []) {
  if (!fs.existsSync(dir)) return acc
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP_DIRS.has(e.name)) continue
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p, pred, acc)
    else if (pred(e.name)) acc.push(path.relative(SRC, p).split(path.sep).join('/'))
  }
  return acc
}

function readmes(parent) {
  const dir = path.join(SRC, parent)
  if (!fs.existsSync(dir)) return []
  const out = []
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!e.isDirectory() || SKIP_DIRS.has(e.name)) continue
    const rel = `${parent}/${e.name}/README.md`
    if (fs.existsSync(path.join(SRC, rel))) out.push(rel)
  }
  return out
}

const rootDocs = ['README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CODE_OF_CONDUCT.md', 'AGENTS.md']
const files = [
  ...rootDocs.filter((f) => fs.existsSync(path.join(SRC, f))),
  ...walk(path.join(SRC, 'docs'), (name) => name.endsWith('.md') || name === 'getting-started.html'),
  ...walk(path.join(SRC, 'extension'), (name) => name.endsWith('.md')),
  ...readmes('packages'),
  ...readmes('plugins'),
  ...readmes('examples'),
]
const fileSet = new Set(files)
const sources = {}

const rewriteTarget = (fromPath, target) => {
  if (!target || /^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('#') || target.startsWith('/')) return target
  const m = target.match(/^([^#?]*)(.*)$/)
  const rel = m[1], tail = m[2]
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(fromPath), rel))
  if (fileSet.has(resolved)) return route(resolved) + tail
  // not a doc in the site: point at the repository
  if (resolved.startsWith('..')) return target
  const kind = rel.endsWith('/') ? 'tree' : 'blob'
  return `${GH}/${kind}/${REF}/${resolved.replace(/\/$/, '')}${tail}`
}

const CALLOUT = { note: 'NOTE', tip: 'TIP', important: 'IMPORTANT', warning: 'WARNING', caution: 'CAUTION' }
const transformMd = (p, text) => {
  let inFence = false
  const lines = text.split('\n').map((line) => {
    if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; return line }
    if (inFence) return line
    // bold "Note:" blockquotes -> GitHub-style alerts (none today; future-proof)
    const c = line.match(/^>\s*\*\*(Note|Tip|Important|Warning|Caution):?\*\*:?\s*(.*)$/i)
    if (c) return `> [!${CALLOUT[c[1].toLowerCase()]}]\n> ${c[2]}`
    return line.replace(/\]\(([^)\s]+)((?:\s+"[^"]*")?)\)/g, (m0, t, title) => `](${rewriteTarget(p, t)}${title})`)
  })
  return lines.join('\n')
}

for (const f of files) {
  if (f === 'docs/getting-started.html') continue
  const o = outPath(f)
  let md = transformMd(f, fs.readFileSync(path.join(SRC, f), 'utf8'))
  if (f === 'README.md') md = md.replace(/^# TensorViz\s*\n/, '# Repository README\n')
  const dest = path.join(OUT, o)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, md)
  sources[o] = f
}

// 2. raw HTML example: static asset + in-site wrapper page.
// The raw file is copied into public/ and is not passed through VitePress,
// so root-absolute URLs in it have to carry `base` themselves.
{
  let html = fs.readFileSync(path.join(SRC, 'docs/getting-started.html'), 'utf8')
  html = html.replace(/(href=")([^"]*)(")/g, (m0, a, u, b) => a + withBase(rewriteTarget('docs/getting-started.html', u)) + b)
  const embedCss = withBase('/raw/embed.css')
  const embedJs = withBase('/raw/embed.js')
  const icon = withBase('/logo.svg')
  html = html.replace('</head>', `<base target="_top">\n<link rel="icon" href="${icon}" type="image/svg+xml">\n<link rel="stylesheet" href="${embedCss}">\n<script src="${embedJs}"></script>\n</head>`)
  fs.writeFileSync(path.join(OUT, 'public/raw/getting-started.html'), html)
  fs.copyFileSync(path.join(here, 'embed/embed.css'), path.join(OUT, 'public/raw/embed.css'))
  fs.copyFileSync(path.join(here, 'embed/embed.js'), path.join(OUT, 'public/raw/embed.js'))
  const rawPage = withBase('/raw/getting-started.html')
  fs.writeFileSync(path.join(OUT, 'docs/getting-started.md'),
`---
title: Small example (HTML)
outline: false
---

<span id="example"></span>
<EmbedFrame src="${rawPage}" />

<p class="embed-note">This page embeds <code>docs/getting-started.html</code> from the repository, unchanged. <a href="${rawPage}" target="_blank" rel="noopener">Open the raw HTML</a>.</p>
`)
  sources['docs/getting-started.md'] = 'docs/getting-started.html'
}

// 3. landing page generated from README.md + docs/README.md (plain doc page, no hero)
const readme = fs.readFileSync(path.join(SRC, 'README.md'), 'utf8')
const para = readme.split(/\n\s*\n/).find((b) => /^TensorViz shows/.test(b.trim())).trim()
const sentences = para.match(/[^.]+\.(?=\s|$)/g).map((s) => s.trim())
const strip = (s) => s.replace(/`/g, '')
const tagline = strip(sentences[0])
const detail = strip(sentences[1])
const tryBlock = readme.split(/^## Try it\s*$/m)[1]
const tryCode = tryBlock.match(/```sh\n([\s\S]*?)```/)[1].replace(/\n$/, '')
const tryNoteMd = tryBlock.split('```')[2].trim().split(/\n\s*\n/)[0].trim()
const mdLinks = (t, from) => t.replace(/\]\(([^)\s]+)\)/g, (m0, u) => `](${rewriteTarget(from, u)})`)
const idx = fs.readFileSync(path.join(SRC, 'docs/README.md'), 'utf8')
const order = ['Quickstart', 'Concepts', 'Guides', 'Agents', 'Reference', 'Contribute']
const rows = []
for (const sec of idx.split(/^## /m).slice(1)) {
  const title = sec.split('\n')[0].trim()
  if (!order.includes(title)) continue
  const links = [...sec.matchAll(/^- \[([^\]]+)\]\(([^)]+)\)/gm)].map((m) => ({ text: m[1], href: rewriteTarget('docs/README.md', m[2]) }))
  rows.push(`| [${title}](${links[0].href}) | ${links.map((l) => `[${l.text}](${l.href})`).join(', ')} |`)
}
const home = `---
title: TensorViz documentation
titleTemplate: false
outline: false
editLink: false
---

# TensorViz

${mdLinks(para, 'README.md')}

## Quickstart

From the repository root:

\`\`\`sh
${tryCode}
\`\`\`

${mdLinks(tryNoteMd, 'README.md')}

## Sections

<div class="sections">

| Section | Pages |
| --- | --- |
${rows.join('\n')}

</div>
`
fs.writeFileSync(path.join(OUT, 'index.md'), home)
fs.writeFileSync(path.join(OUT, '.vitepress/generated.json'), JSON.stringify({ sha: commit, ref: REF, base, sources, tryCode, detail, tagline }, null, 2))
console.log(`prepared ${Object.keys(sources).length} pages @ ${REF} (${commit.slice(0, 7)})`)
