import '../styles/fonts';
import React from 'react';
import ReactDOM from 'react-dom/client';
import '../index.css';
import '../styles/ui2.tokens.css';
import GraphFileViewer from './GraphFileViewer';
import { configureCanvasTheme } from '../styles/canvasTheme';

// An independent entry: no auth provider, VS Code bridge or backend.
configureCanvasTheme('browser');
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><GraphFileViewer /></React.StrictMode>,
);
