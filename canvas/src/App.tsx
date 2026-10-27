import React from 'react';

/** Non-IDE CRA fallback. The extension loads src/ide/main when REACT_APP_IDE is set. */
export default function App(): React.ReactElement {
  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: 24, maxWidth: 640 }}>
      <h1>tensorViz Canvas</h1>
      <p>The hosted website is not part of this repo. Open the Canvas from the tensorViz extension.</p>
    </main>
  );
}
