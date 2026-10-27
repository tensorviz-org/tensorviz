<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
const props = defineProps<{ code: string }>()
const el = ref<HTMLElement | null>(null)
const failed = ref(false)
let mo: MutationObserver | null = null
let n = 0

// strictly grayscale
const vars = (dark: boolean) => dark
  ? { background: 'transparent', primaryColor: '#1c1c1c', primaryBorderColor: '#6b6b6b', primaryTextColor: '#ededed', secondaryColor: '#161616', tertiaryColor: '#111111',
      lineColor: '#8f8f8f', textColor: '#ededed', mainBkg: '#1c1c1c', nodeBorder: '#6b6b6b', clusterBkg: '#141414', clusterBorder: '#333333', edgeLabelBackground: '#111111',
      fontFamily: 'Geist Variable, Geist, system-ui, sans-serif', fontSize: '14px' }
  : { background: 'transparent', primaryColor: '#f5f5f5', primaryBorderColor: '#a3a3a3', primaryTextColor: '#171717', secondaryColor: '#fafafa', tertiaryColor: '#ffffff',
      lineColor: '#737373', textColor: '#171717', mainBkg: '#f5f5f5', nodeBorder: '#a3a3a3', clusterBkg: '#fafafa', clusterBorder: '#d4d4d4', edgeLabelBackground: '#ffffff',
      fontFamily: 'Geist Variable, Geist, system-ui, sans-serif', fontSize: '14px' }

async function render() {
  if (!el.value) return
  try {
    const mermaid = (await import('mermaid')).default
    const dark = document.documentElement.classList.contains('dark')
    mermaid.initialize({ startOnLoad: false, theme: 'base', themeVariables: vars(dark), securityLevel: 'strict', flowchart: { curve: 'linear', padding: 12, htmlLabels: true } })
    const { svg } = await mermaid.render(`mmd-${Math.random().toString(36).slice(2)}-${n++}`, decodeURIComponent(props.code))
    el.value.innerHTML = svg
    failed.value = false
  } catch (e) { failed.value = true; console.error(e) }
}
onMounted(() => {
  render()
  mo = new MutationObserver(render)
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
})
onBeforeUnmount(() => mo?.disconnect())
</script>

<template>
  <figure class="mermaid-figure">
    <div ref="el" class="mermaid-render" role="img" aria-label="Diagram"></div>
    <pre v-if="failed" class="mermaid-src">{{ decodeURIComponent(code) }}</pre>
  </figure>
</template>
