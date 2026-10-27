import React from 'react';
import { TriangleAlert } from 'lucide-react';
import CanvasStateScreen from './CanvasStateScreen';

interface State { error?: Error; componentStack?: string; attempt: number }

/** An uncaught render or effect error would otherwise unmount the whole Canvas and leave
 * a blank view. Show what failed, and let Reload remount the Canvas; a fresh mount
 * reconnects to VS Code. The page is not reloaded, so its launch token is kept. */
export class CanvasErrorBoundary extends React.Component<{ children: React.ReactNode }, State> {
  state: State = { attempt: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: React.ErrorInfo): void {
    this.setState({ componentStack: info.componentStack ?? undefined });
    // eslint-disable-next-line no-console
    console.error('tensorViz: the Canvas stopped after an error', error);
  }

  private reload = () => this.setState(state => ({ error: undefined, componentStack: undefined, attempt: state.attempt + 1 }));

  render(): React.ReactNode {
    const { error, componentStack, attempt } = this.state;
    if (!error) return <React.Fragment key={attempt}>{this.props.children}</React.Fragment>;
    const details = [error.stack || `${error.name}: ${error.message}`, componentStack?.trim()].filter(Boolean).join('\n\n').slice(0, 8000);
    return <CanvasStateScreen title="The Canvas stopped after an error" icon={<TriangleAlert aria-hidden="true" />}
      actions={<button type="button" className="primary" onClick={this.reload}>Reload Canvas</button>}>
      <p role="alert">{error.message || error.name}</p>
      <p>Reload reconnects to VS Code; source files and saved setup are unchanged. If it happens again, include the details below when reporting it.</p>
      <details className="canvas-state-details"><summary>Error details</summary><pre>{details}</pre></details>
    </CanvasStateScreen>;
  }
}
