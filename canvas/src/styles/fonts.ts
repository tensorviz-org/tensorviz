import inter from '../assets/fonts/Inter-latin-variable.woff2';
import mono from '../assets/fonts/JetBrainsMono-latin-variable.woff2';
import './fonts.css';
import './canvas.tokens.css';

// Start both faces before the lazy Canvas mounts; panel URLs are inline data.
for (const href of [inter, mono]) {
  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'font';
  link.type = 'font/woff2';
  link.crossOrigin = 'anonymous';
  link.href = href;
  document.head.append(link);
}
