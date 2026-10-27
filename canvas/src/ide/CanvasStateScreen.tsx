import React from 'react';
import { Workflow } from 'lucide-react';
import './CanvasStateScreen.css';

/** Canvas-owned empty and recovery states, without mounting the graph editor. */
export default function CanvasStateScreen({ title, children, actions, icon, illustration = false, embedded = false }: {
  title: string; children: React.ReactNode; actions?: React.ReactNode;
  icon?: React.ReactNode; illustration?: boolean; embedded?: boolean;
}) {
  const [reduced, setReduced] = React.useState(true);
  const [visible, setVisible] = React.useState(false);
  React.useEffect(() => {
    if (!illustration) return;
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => { setReduced(media.matches); setVisible(!document.hidden); };
    update();
    media.addEventListener('change', update);
    document.addEventListener('visibilitychange', update);
    return () => { media.removeEventListener('change', update); document.removeEventListener('visibilitychange', update); };
  }, [illustration]);
  const id = React.useId();
  return <section className={`canvas-state${embedded ? ' canvas-state-embedded' : ''}`} aria-labelledby={id}
    data-motion={reduced || !visible ? 'paused' : 'playing'}>
    {!embedded && <div className="canvas-state-brand"><Workflow size={22} aria-hidden="true" />tensorViz</div>}
    <div className="canvas-state-content">
      <div className="canvas-state-art" aria-hidden="true">
        {illustration ? <svg viewBox="0 0 320 190" focusable="false">
          <g className="canvas-state-wires" fill="none">
            <path d="M40 95C68 95 66 35 96 35M40 95H96M40 95C68 95 66 155 96 155" />
            <path d="M128 35C160 35 160 65 192 65M128 35C164 35 156 125 192 125M128 95C160 95 160 65 192 65M128 95C160 95 160 125 192 125M128 155C164 155 156 65 192 65M128 155C160 155 160 125 192 125" />
            <path d="M224 65C256 65 250 95 280 95M224 125C256 125 250 95 280 95" />
          </g>
          <path className="canvas-state-signal" pathLength="100" fill="none" d="M40 95C68 95 66 35 96 35M128 35C160 35 160 65 192 65M224 65C256 65 250 95 280 95" />
          <path className="canvas-state-signal second" pathLength="100" fill="none" d="M40 95H96M128 95C160 95 160 125 192 125M224 125C256 125 250 95 280 95" />
          <g className="canvas-state-nodes">
            <rect x="8" y="79" width="32" height="32" rx="8" /><rect x="280" y="79" width="32" height="32" rx="8" />
            <rect x="96" y="19" width="32" height="32" rx="6" /><rect x="96" y="79" width="32" height="32" rx="6" />
            <rect x="96" y="139" width="32" height="32" rx="6" /><rect x="192" y="49" width="32" height="32" rx="6" /><rect x="192" y="109" width="32" height="32" rx="6" />
          </g>
          <g className="canvas-state-detail" fill="none">
            <path d="M18 90h12M18 95h12M18 100h12M290 90h12M290 95h12M290 100h12M106 35h12m-6-6v12M106 95h12m-6-6v12M106 155h12m-6-6v12M202 69l4-8 4 8h4M202 129l4-8 4 8h4" />
          </g>
        </svg> : icon}
      </div>
      <h1 id={id}>{title}</h1>
      <div className="canvas-state-copy">{children}</div>
      {actions && <div className="canvas-state-actions">{actions}</div>}
    </div>
  </section>;
}
