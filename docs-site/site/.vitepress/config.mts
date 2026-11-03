import { defineConfig } from 'vitepress'
import fs from 'node:fs'
import path from 'node:path'
import { monoLight, monoDark } from './mono-theme'
import { REPO, route } from '../../routes.mjs'

const gen = JSON.parse(fs.readFileSync(path.join(__dirname, 'generated.json'), 'utf8'))
const REF: string = gen.ref
const base: string = gen.base || '/'
// `head` is written into the HTML as given. The nav logo goes through
// VPImage's withBase, which prefixes `base` itself.
const asset = (p: string) => (base === '/' || p.startsWith(base) ? p : `${base}${p.replace(/^\//, '')}`)
const r = (p: string) => route(p)
const item = (text: string, p: string) => ({ text, link: r(p) })

const sidebar = [
  { text: 'Start here', items: [
    { text: 'Overview', link: '/' }, item('Quickstart', 'docs/quickstart.md'),
    item('Installation options', 'docs/install.md')] },
  { text: 'Use TensorViz', items: [
    item('Open a model', 'docs/graph-opening.md'), item('Explore the graph', 'docs/canvas.md'),
    item('Follow the code', 'docs/code-focus.md'), item('Review changes', 'docs/graph-diff-review/README.md'),
    item('Reuse a capture', 'docs/execution-cache.md'), item('Edit your model', 'docs/source-editing.md'),
    item('Export and share', 'docs/graph-files.md')] },
  { text: 'Setup & help', collapsed: true, items: [
    item('Python environments', 'docs/environment-recovery.md'), item('Model inputs', 'docs/graph-setup.md'),
    item('Coding-agent plugin', 'docs/agent-plugin.md'), item('Connect your editor to an agent', 'docs/mcp.md'),
    item('JAX and Flax', 'docs/jax-support.md'), item('Hugging Face', 'docs/hugging-face-models.md')] },
  { text: 'Reference', collapsed: true, items: [
    item('All guides', 'docs/README.md'), item('Settings and commands', 'docs/reference.md'),
    item('How capture works', 'docs/concepts.md'), item('Saved model profiles', 'docs/model-validation-profiles.md'),
    item('Model libraries', 'docs/graph-libraries.md'), item('MCP tools', 'docs/mcp-tools.md'),
    item('MCP security', 'docs/mcp-security.md'), item('Canvas reference', 'docs/reference/canvas.md'),
    item('Graph review reference', 'docs/reference/graph-review.md')] },
  { text: 'Contribute', collapsed: true, items: [
    item('Build from source', 'docs/local-setup.md'), item('Build the plugin', 'docs/plugin-build.md'),
    item('Contributing', 'CONTRIBUTING.md'), item('Licensing', 'docs/licensing.md'),
    item('Security', 'SECURITY.md'), item('Extension changelog', 'extension/CHANGELOG.md')] },
]

// VitePress serialises this function into the client bundle, so constants are inlined via eval.
const editPattern = eval(`(({ filePath }) => 'https://github.com/${REPO}/blob/${REF}/' + (${JSON.stringify(gen.sources)}[filePath] ?? filePath))`)

export default async () => {
  return defineConfig({
    title: 'TensorViz',
    titleTemplate: ':title · TensorViz docs',
    description: gen.tagline,
    lang: 'en-US',
    base,
    appearance: true,
    // Pages does not map /page to /page.html. Links keep the .html suffix.
    cleanUrls: false,
    srcExclude: ['**/node_modules/**', 'public/**'],
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: asset('/logo.svg') }],
      ['meta', { name: 'theme-color', content: '#ffffff', media: '(prefers-color-scheme: light)' }],
      ['meta', { name: 'theme-color', content: '#0f0f0f', media: '(prefers-color-scheme: dark)' }],
    ],
    markdown: {
      theme: { light: monoLight as any, dark: monoDark as any },
      lineNumbers: false,
      // GitHub-compatible heading ids so existing #anchors in the docs keep working
      // Collapse whitespace the way the repo link checker and GitHub do, so an
      // em dash in a heading does not become a double hyphen the docs do not use.
      anchor: { slugify: (s: string) => s.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s+/g, '-') },
      config(md) {
        const fence = md.renderer.rules.fence!
        md.renderer.rules.fence = (tokens, idx, options, env, self) => {
          const t = tokens[idx]
          if (t.info.trim().split(/\s/)[0] === 'mermaid') return `<Mermaid code="${encodeURIComponent(t.content)}" />\n`
          return fence(tokens, idx, options, env, self)
        }
        // docs contain literal {{PLACEHOLDER}} text: keep Vue from interpolating it
        md.core.ruler.push('escape_mustache', (state) => {
          for (const blk of state.tokens) {
            if (blk.type !== 'inline' || !blk.children) continue
            for (const t of blk.children) {
              if (t.type === 'text' && t.content.includes('{{')) {
                t.type = 'html_inline'
                t.content = md.utils.escapeHtml(t.content).replace(/\{\{/g, '&#123;&#123;')
              }
            }
          }
        })
        md.renderer.rules.table_open = () => '<div class="table-wrap"><table>\n'
        md.renderer.rules.table_close = () => '</table></div>\n'
      },
    },
    themeConfig: {
      logo: { light: '/logo-light.svg', dark: '/logo-dark.svg', alt: 'TensorViz' },
      siteTitle: 'TensorViz',
      nav: [
        { text: 'Quickstart', link: r('docs/quickstart.md') },
        { text: 'Guides', link: r('docs/README.md') },
        { text: 'Reference', link: r('docs/reference.md') },
        { text: 'Contribute', link: r('docs/local-setup.md') },
      ],
      sidebar,
      socialLinks: [{ icon: 'github', link: `https://github.com/${REPO}` }],
      search: { provider: 'local', options: { detailedView: true } },
      outline: { level: [2, 3], label: 'On this page' },
      editLink: {
        pattern: editPattern,
        text: `Edit on GitHub (${/^[0-9a-f]{7,40}$/i.test(REF) ? REF.slice(0, 7) : REF})`,
      },
      docFooter: { prev: 'Previous', next: 'Next' },
    } as any,
  })
}
