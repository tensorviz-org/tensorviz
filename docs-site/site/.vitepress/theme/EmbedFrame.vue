<script setup lang="ts">
import { withBase, useData } from 'vitepress'
import { computed, ref } from 'vue'
const props = defineProps<{ src: string }>()
const { site } = useData()
const frame = ref<HTMLIFrameElement | null>(null)
const h = ref(520)
// `src` may already include `base` (prepare.mjs writes it into the page).
const src = computed(() => {
  const base = site.value.base || '/'
  if (base !== '/' && (props.src === base || props.src.startsWith(base))) return props.src
  return withBase(props.src)
})
function fit() {
  const d = frame.value?.contentDocument
  if (d) h.value = Math.max(d.documentElement.scrollHeight, d.body.scrollHeight) + 4
}
</script>
<template>
  <iframe ref="frame" class="embed-frame" :src="src" :style="{ height: h + 'px' }" title="Small example" @load="fit"></iframe>
</template>
<style scoped>
.embed-frame { width: 100%; border: 0; display: block; background: transparent; color-scheme: normal }
</style>
