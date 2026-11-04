<script setup lang="ts">
import { ref, watch } from 'vue'
import { useData, withBase } from 'vitepress'

const { page, frontmatter } = useData()
const status = ref('')
const copying = ref(false)
const markdownUrl = () => withBase(`/markdown/${page.value.relativePath}`)
watch(() => page.value.relativePath, () => { status.value = '' })

async function copyPage() {
  copying.value = true
  status.value = ''
  const path = page.value.relativePath
  try {
    const response = await fetch(markdownUrl())
    if (!response.ok) throw new Error('Markdown unavailable')
    await navigator.clipboard.writeText(await response.text())
    if (page.value.relativePath === path) status.value = 'Copied'
  } catch {
    if (page.value.relativePath === path) status.value = 'Could not copy. Open Markdown to select the text.'
  } finally {
    copying.value = false
  }
}
</script>

<template>
  <div v-if="frontmatter.docActions !== false && !page.isNotFound" class="page-actions">
    <button type="button" :disabled="copying" @click="copyPage">{{ copying ? 'Copying…' : 'Copy page' }}</button>
    <a :href="markdownUrl()" target="_blank" rel="noopener">View Markdown ↗</a>
    <span role="status">{{ status }}</span>
  </div>
</template>
