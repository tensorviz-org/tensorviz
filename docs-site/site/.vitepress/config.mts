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
  { text: 'Overview', items: [item('Documentation index', 'docs/README.md'), item('Repository README', 'README.md')] },
  { text: 'Quickstart', items: [item('Open a model', 'docs/quickstart.md'), item('Small example (HTML)', 'docs/getting-started.html')] },
  { text: 'Concepts', items: [
    item('Capture, files, and hosts', 'docs/concepts.md'), item('Graph files', 'docs/graph-files.md'),
    item('Saved profiles (tensorviz.toml)', 'docs/model-validation-profiles.md')] },
  { text: 'Guides', items: [
    item('Open a graph', 'docs/graph-opening.md'), item('Setup and inputs', 'docs/graph-setup.md'), item('Model libraries', 'docs/graph-libraries.md'),
    item('Canvas', 'docs/canvas.md'), item('Code focus', 'docs/code-focus.md'), item('Source proposals', 'docs/source-editing.md'),
    item('Graph diff', 'docs/graph-diff-review/README.md'), item('Review a pull request', 'docs/graph-diff-review/pr-workflow.md'),
    item('Python environments', 'docs/environment-recovery.md'), item('JAX and Flax', 'docs/jax-support.md'),
    item('Hugging Face models', 'docs/hugging-face-models.md'), item('Share to tensorviz.dev', 'docs/share-to-website.md')] },
  { text: 'Agents', items: [
    item('Connect MCP', 'docs/mcp.md'), item('MCP tools', 'docs/mcp-tools.md'), item('MCP security', 'docs/mcp-security.md'),
    item('Coding-agent plugin', 'docs/agent-plugin.md')] },
  { text: 'Reference', items: [
    item('Settings, commands, tools', 'docs/reference.md'), item('Local browser broker', 'docs/local-browser-broker.md'),
    item('Extension README', 'extension/README.md'), item('Extension changelog', 'extension/CHANGELOG.md')] },
  { text: 'Extension walkthrough', collapsed: true, items: [
    item('Repository', 'extension/walkthrough/repository.md'), item('Environment', 'extension/walkthrough/environment.md'),
    item('Graph', 'extension/walkthrough/graph.md'), item('Validate', 'extension/walkthrough/validate.md'),
    item('Source edit', 'extension/walkthrough/source-edit.md'), item('Agent', 'extension/walkthrough/agent.md'),
    item('Export', 'extension/walkthrough/export.md')] },
  { text: 'Contribute', items: [
    item('Local setup', 'docs/local-setup.md'), item('Plugin build', 'docs/plugin-build.md'), item('Licensing', 'docs/licensing.md'),
    item('Contributing', 'CONTRIBUTING.md'), item('Security', 'SECURITY.md'), item('Code of conduct', 'CODE_OF_CONDUCT.md'),
    item('Agent instructions', 'AGENTS.md')] },
  { text: 'Packages and examples', collapsed: true, items: [
    item('contracts', 'packages/contracts/README.md'), item('graph-host', 'packages/graph-host/README.md'),
    item('Plugin (plugins/tensorviz)', 'plugins/tensorviz/README.md'),
    item('tensorviz-research example', 'examples/tensorviz-research/README.md'), item('tensorviz-test example', 'examples/tensorviz-test/README.md')] },
  { text: 'Packaged templates', collapsed: true, items: [
    item('public-plugin README', 'docs/public-plugin/README.md'), item('Installation', 'docs/public-plugin/docs/installation.md'),
    item('BUILD', 'docs/public-plugin/BUILD.md'), item('PRIVACY', 'docs/public-plugin/PRIVACY.md'),
    item('SUPPORT', 'docs/public-plugin/SUPPORT.md'), item('TERMS', 'docs/public-plugin/TERMS.md'),
    { text: 'Stubs', collapsed: true, items: [
      item('standalone-install', 'docs/standalone-install.md'), item('standalone-permissions', 'docs/standalone-permissions.md'),
      item('standalone-cursor', 'docs/standalone-cursor.md'), item('standalone-panel', 'docs/standalone-panel.md'),
      item('standalone-t3', 'docs/standalone-t3.md'), item('package-names', 'docs/package-names.md')] }] },
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
    srcExclude: ['**/node_modules/**'],
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
        { text: 'Quickstart', link: r('docs/quickstart.md'), activeMatch: '^/docs/(quickstart|getting-started)' },
        { text: 'Concepts', link: r('docs/concepts.md'), activeMatch: '^/docs/(concepts|graph-files|model-validation)' },
        { text: 'Guides', link: r('docs/graph-opening.md'), activeMatch: '^/docs/(graph-(?!files)|canvas|code-focus|source-editing|environment|jax|hugging|share)' },
        { text: 'Agents', link: r('docs/mcp.md'), activeMatch: '^/docs/(mcp|agent-plugin)' },
        { text: 'Reference', link: r('docs/reference.md'), activeMatch: '^/(docs/(reference|local-browser)|extension/)' },
        { text: 'Contribute', link: r('docs/local-setup.md'), activeMatch: '^/(docs/(local-setup|plugin-build|licensing)|CONTRIBUTING|SECURITY|CODE_OF)' },
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
