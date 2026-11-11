import React from 'react';
import type { cardHeaderLayout } from './cardIdentity';

/** Exact identity, wrapped without replacing any characters with ellipses. */
export function CardIdentity({ layout, title, subtitle, module = false }: {
  layout: ReturnType<typeof cardHeaderLayout>; title: string; subtitle?: string; module?: boolean;
}) {
  return <div className={module ? 'node-title' : 'ui2-node-title-wrap'}>
    <div className={module ? 'layer-name' : 'ui2-node-title'} title={title}>
      {layout.titleLines.map((line, index) => <span className="node-identity-line" key={index}>{line}</span>)}
    </div>
    {subtitle && <div className={module ? 'layer-id' : 'ui2-node-subtitle'} title={subtitle}>
      {layout.subtitleLines.map((line, index) => <span className="node-identity-line" key={index}>{line}</span>)}
    </div>}
  </div>;
}
