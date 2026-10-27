import React from 'react';
import type { GraphOpeningState, GraphSetupAction } from './protocol.generated';

interface Props {
  state: GraphOpeningState;
  connected: boolean;
  standalone?: boolean;
  onAction(action: GraphSetupAction): void;
  onClose?(): void;
  children: React.ReactNode;
}

/** The host forwards any bounded JSON draft, including an agent's attempt the backend
 * later rejects. A draft the form cannot edit must not unmount the whole Canvas: the
 * failure stays in this panel, and the next opening state retries the form. */
export class GraphSetupBoundary extends React.Component<Props, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    // eslint-disable-next-line no-console
    console.error('tensorViz: the setup form could not show this draft', error);
  }

  componentDidUpdate(previous: Props): void {
    if (this.state.failed && previous.state !== this.props.state) this.setState({ failed: false });
  }

  render(): React.ReactNode {
    if (!this.state.failed) return this.props.children;
    const { state, connected, standalone, onAction, onClose } = this.props;
    return <section className="graph-opening-panel graph-model-setup-panel" aria-label="Open captured graph">
      {onClose && <button type="button" className="graph-opening-close" aria-label="Close inputs" title="Close inputs (Escape)" onClick={onClose}>×</button>}
      <header className="graph-setup-header">
        <div className="graph-opening-heading"><span className="graph-opening-indicator" />
          <div><div className="graph-opening-model"><strong>{state.target.cls}</strong><code>{state.target.file}</code></div>
            <h1>This setup can’t be shown in the form</h1>
            <p role="status">{state.detail}</p></div></div>
      </header>
      <div className="graph-setup-body">
        <p>The current setup uses a shape the form cannot edit, such as an agent’s draft that is still being checked.
          The graph stays available. Edit the profile directly, or wait for the next setup update.</p>
        {state.requirements?.length ? <ul role="alert">{state.requirements.map((item, index) => <li key={index}>{item}</li>)}</ul> : null}
        <div className="graph-setup-actions"><button type="button" disabled={!connected} onClick={() => onAction({ kind: 'edit_config' })}>
          {standalone ? 'Copy config path' : 'Edit tensorviz.toml'}</button></div>
      </div>
    </section>;
  }
}
