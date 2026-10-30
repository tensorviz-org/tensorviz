<script setup lang="ts">
import { computed, ref, useId, watch } from 'vue'
import { withBase } from 'vitepress'

const id = useId()
const routes = [
  { id: 'plugin', label: 'Coding-agent plugin' },
  { id: 'extension', label: 'Editor extension' },
  { id: 'source', label: 'Build from source' },
] as const
const active = ref('plugin')
const client = ref('Codex')
const editor = ref('VS Code')
const copyStatus = ref('')
const pluginCommands: Record<string, string> = {
  Codex: 'codex plugin marketplace add tensorviz-org/tensorviz-plugins\ncodex plugin add tensorviz@tensorviz-standalone',
  'Claude Code': 'claude plugin marketplace add tensorviz-org/tensorviz-plugins\nclaude plugin install tensorviz@tensorviz-standalone --scope user',
}
const editorCommands: Record<string, string> = {
  'VS Code': 'code --install-extension tensorviz.tensorviz-vscode',
  Cursor: 'cursor --install-extension tensorviz.tensorviz-vscode',
  VSCodium: 'codium --install-extension tensorviz.tensorviz-vscode',
}
const command = computed(() => active.value === 'plugin'
  ? pluginCommands[client.value] ?? ''
  : active.value === 'extension' ? editorCommands[editor.value]
    : 'git clone https://github.com/tensorviz-org/tensorviz.git\ncd tensorviz')
watch([active, client, editor], () => { copyStatus.value = '' })

async function copy() {
  try {
    await navigator.clipboard.writeText(command.value)
    copyStatus.value = 'Copied'
  } catch {
    copyStatus.value = 'Copy failed. Select and copy the command text.'
  }
}

function moveTab(event: KeyboardEvent, index: number) {
  const next = event.key === 'ArrowRight' ? (index + 1) % routes.length
    : event.key === 'ArrowLeft' ? (index + routes.length - 1) % routes.length
      : event.key === 'Home' ? 0 : event.key === 'End' ? routes.length - 1 : -1
  if (next < 0) return
  event.preventDefault()
  active.value = routes[next].id
  document.getElementById(`${id}-tab-${active.value}`)?.focus()
}
</script>

<template>
  <section class="install-options" aria-label="Install TensorViz">
    <div class="install-tabs" role="tablist" aria-label="Choose how to install">
      <button v-for="(option, index) in routes" :key="option.id"
        :id="`${id}-tab-${option.id}`" role="tab" type="button"
        :aria-selected="active === option.id" :aria-controls="active === option.id ? `${id}-panel-${option.id}` : undefined"
        :tabindex="active === option.id ? 0 : -1" @click="active = option.id"
        @keydown="moveTab($event, index)">{{ option.label }}</button>
    </div>
    <div :id="`${id}-panel-${active}`" class="install-panel" role="tabpanel"
      :aria-labelledby="`${id}-tab-${active}`" tabindex="0">
      <template v-if="active === 'plugin'">
        <h2>Bring the graph into your agent.</h2>
        <p>Install once, then ask your agent to open a model.</p>
        <div class="install-choices" role="group" aria-label="Coding client">
          <button v-for="name in ['Codex', 'Claude Code', 'Cursor']" :key="name" type="button"
            :aria-pressed="client === name" @click="client = name">{{ name }}</button>
        </div>
        <p class="install-requirements">Plugin preview · Apple Silicon Mac · CPython 3.12 · CPU capture</p>
      </template>
      <template v-else-if="active === 'extension'">
        <h2>Keep the graph beside your code.</h2>
        <p>Install from your editor’s marketplace, or run its terminal command.</p>
        <div class="install-choices" role="group" aria-label="Editor">
          <button v-for="name in ['VS Code', 'Cursor', 'VSCodium']" :key="name" type="button"
            :aria-pressed="editor === name" @click="editor = name">{{ name }}</button>
        </div>
        <p class="install-requirements">VS Code 1.106+ compatible editor · Python and your model’s dependencies</p>
      </template>
      <template v-else>
        <h2>Build this repository locally.</h2>
        <p>For contributors and anyone who wants the source-built version. The repository is public.</p>
      </template>

      <div v-if="command" class="install-command">
        <div class="install-command-bar"><span>Terminal</span><button type="button" @click="copy">{{ copyStatus === 'Copied' ? 'Copied' : copyStatus ? 'Copy failed — select text' : 'Copy commands' }}</button></div>
        <pre><code>{{ command }}</code></pre>
      </div>
      <span class="install-copy-status" role="status">{{ copyStatus }}</span>

      <template v-if="active === 'plugin'">
        <template v-if="client === 'Cursor'">
          <a class="install-primary" href="https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.3.60/tensorviz-cursor-0.3.60.zip">Download Cursor plugin</a>
          <p>Import the Cursor ZIP using the <a href="https://github.com/tensorviz-org/tensorviz-plugins#install-in-cursor">Cursor setup steps</a>.</p>
        </template>
        <p v-else class="install-next">Run both commands in a terminal with {{ client }} installed. Open your model project and start a <strong>new chat</strong>.</p>
        <a :href="withBase('/docs/agent-plugin.html')">Plugin setup, permissions &amp; other clients →</a>
      </template>
      <template v-else-if="active === 'extension'">
        <div class="install-actions">
          <a class="install-primary" href="https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode">VS Code Marketplace</a>
          <a href="https://open-vsx.org/extension/tensorviz/tensorviz-vscode">Open VSX</a>
          <a href="https://open-vsx.org/api/tensorviz/tensorviz-vscode/0.1.35/file/tensorviz.tensorviz-vscode-0.1.35.vsix">Download VSIX</a>
        </div>
        <p class="install-next">For a downloaded file, run <strong>Extensions: Install from VSIX…</strong> in your editor.</p>
        <a :href="withBase('/docs/install.html#editor-extension')">More installation options →</a>
      </template>
      <template v-else>
        <p>Continue with Node, Python, and packaging in the build guide.</p>
        <div class="install-actions">
          <a class="install-primary" :href="withBase('/docs/local-setup.html')">Build the extension</a>
          <a :href="withBase('/docs/plugin-build.html')">Build the plugin</a>
        </div>
        <p class="install-next">The published packages and this source checkout have separate versions.</p>
      </template>
    </div>
  </section>
</template>
