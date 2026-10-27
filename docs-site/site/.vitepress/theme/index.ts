import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import '@fontsource-variable/geist'
import '@fontsource-variable/geist-mono'
import './style.css'
import Layout from './Layout.vue'
import Mermaid from './Mermaid.vue'
import EmbedFrame from './EmbedFrame.vue'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('Mermaid', Mermaid)
    app.component('EmbedFrame', EmbedFrame)
  },
} satisfies Theme
