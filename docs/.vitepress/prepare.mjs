// Copy repository docs into VitePress, adapt links, and export agent-readable Markdown.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { REPO, outPath, route } from './routes.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(here, '../..')
const OUT = path.join(here, 'cache/content')
const GH = `https://github.com/${REPO}`
const SKIP_DIRS = new Set(['node_modules', '.git', '.vitepress', 'public'])

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

// Generated pages and downloads stay in the ignored build cache.
fs.rmSync(OUT, { recursive: true, force: true })
fs.mkdirSync(OUT, { recursive: true })
fs.cpSync(path.join(SRC, 'docs/public'), path.join(OUT, 'public'), { recursive: true })
fs.cpSync(path.join(SRC, 'docs/images'), path.join(OUT, 'public/docs/images'), { recursive: true })

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

const rootDocs = ['CONTRIBUTING.md', 'SECURITY.md', 'CODE_OF_CONDUCT.md', 'extension/CHANGELOG.md']
const files = [
  ...rootDocs,
  ...walk(path.join(SRC, 'docs'), name => name.endsWith('.md')).filter(file => file !== 'docs/index.md'),
]
const fileSet = new Set(files)
const sources = {}

const rewriteTarget = (fromPath, target) => {
  if (!target || /^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('#') || target.startsWith('/')) return target
  const m = target.match(/^([^#?]*)(.*)$/)
  const rel = m[1], tail = m[2]
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(fromPath), rel))
  if (fileSet.has(resolved)) return route(resolved) + tail
  if (resolved.startsWith('docs/images/') && fs.existsSync(path.join(SRC, resolved))) return `/${resolved}${tail}`
  // not a doc in the site: point at the repository
  if (resolved.startsWith('..')) return target
  const kind = rel.endsWith('/') ? 'tree' : 'blob'
  return `${GH}/${kind}/${REF}/${resolved.replace(/\/$/, '')}${tail}`
}

const CALLOUT = { note: 'NOTE', tip: 'TIP', important: 'IMPORTANT', warning: 'WARNING', caution: 'CAUTION' }
const transformMd = (p, text, interactive = true) => {
  let inFence = false
  const lines = text.split('\n').map((line) => {
    if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; return line }
    if (inFence) return line
    // bold "Note:" blockquotes -> GitHub-style alerts (none today; future-proof)
    const c = line.match(/^>\s*\*\*(Note|Tip|Important|Warning|Caution):?\*\*:?\s*(.*)$/i)
    if (c) return `> [!${CALLOUT[c[1].toLowerCase()]}]\n> ${c[2]}`
    return line.replace(/\]\(([^)\s]+)((?:\s+"[^"]*")?)\)/g, (m0, t, title) => `](${rewriteTarget(p, t)}${title})`)
  })
  const markdown = lines.join('\n')
  return interactive ? markdown.replace(/<!-- install-options:start -->[\s\S]*?<!-- install-options:end -->/g, '<InstallOptions />') : markdown
}

function exportMarkdown(source, output, content) {
  const markdown = transformMd(source, content, false)
    .replace(/<InstallOptions\b[^>]*\/>/g, `[Install TensorViz](${withBase('/docs/install.html')})`)
    .replace(/<\/?div\b[^>]*>/g, '')
    .replace(/<p\b[^>]*>(.*?)<\/p>/gs, '$1\n')
    .replace(/\]\((\/[^)]*)\)/g, (_, url) => `](${withBase(url)})`)
  const dest = path.join(OUT, 'public/markdown', output)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  // Raw Markdown needs an encoding marker on hosts that omit a charset.
  fs.writeFileSync(dest, "\ufeff" + markdown)
}

for (const f of files) {
  const o = outPath(f)
  let md = transformMd(f, fs.readFileSync(path.join(SRC, f), 'utf8'))
  const dest = path.join(OUT, o)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, md)
  exportMarkdown(f, o, fs.readFileSync(path.join(SRC, f), 'utf8'))
  sources[o] = f
}

const home = fs.readFileSync(path.join(SRC, 'docs/index.md'), 'utf8')
const tagline = 'Install TensorViz, open a model, and explore its graph.'
sources['index.md'] = 'docs/index.md'
fs.writeFileSync(path.join(OUT, 'index.md'), transformMd('docs/index.md', home))
exportMarkdown('docs/index.md', 'index.md', home)
const agentPages = [
  ['Quickstart', 'docs/quickstart.md', 'Create a small PyTorch model and open its first graph.'],
  ['Installation', 'docs/install.md', 'Published clients, compatibility, and source builds.'],
  ['Explore the graph', 'docs/canvas.md', 'Layers, connections, recorded shapes, and navigation.'],
  ['Follow the code', 'docs/code-focus.md', 'Find a selected layer in Python.'],
  ['Review changes', 'docs/graph-diff-review/README.md', 'Compare source revisions without executing them.'],
  ['Execution cache', 'docs/execution-cache.md', 'Reuse a compatible result or run fresh.'],
  ['Python environments', 'docs/environment-recovery.md', 'Choose an environment and resolve missing libraries.'],
  ['Model inputs', 'docs/graph-setup.md', 'Configure constructor arguments and example inputs.'],
]
fs.writeFileSync(path.join(OUT, 'public/llms.txt'), `# TensorViz\n\n> Local model visualization for editors and coding agents.\n\nCapture executes model code. Source previews and graph reviews do not establish runtime shapes or behavioral equivalence. See Installation for client-specific platform support.\n\n## Guides\n\n${agentPages.map(([title, file, description]) => `- [${title}](${withBase(`/markdown/${outPath(file)}`)}): ${description}`).join('\n')}\n`)
fs.writeFileSync(path.join(here, 'cache/generated.json'), JSON.stringify({ sha: commit, ref: REF, base, sources, tagline }, null, 2))
console.log(`prepared ${Object.keys(sources).length} pages @ ${REF} (${commit.slice(0, 7)})`)
