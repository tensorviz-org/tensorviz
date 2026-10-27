import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { findMissingRequiredParameters, describeMissingParameters } from '../utils/requiredParameters';
import * as LucideIcons from 'lucide-react';
import { ModelGraph } from '../types';
import { UnsupportedModule, ApiService } from '../services/apiService';
import { parseInputShape } from '../utils/inputShape';
import * as hostServices from '../ide/hostServices';
import { useBackendEnvironment } from '../services/useBackendEnvironment';

interface CodeExportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  graph: ModelGraph;
  graphName?: string;
  /**
   * Add an Input node wired to the head of the chain. When this is provided, a
   * graph with no usable Input node gets an offer instead of a dead end — the
   * refusal used to name the problem and leave the user to discover that the fix
   * was a palette layer the tutorial never mentioned.
   */
  onAddInputNode?: () => void;
}

/**
 * One entry of the zip the backend builds (backend/services/verification_service.py,
 * `export_package`). `text` is present only for the entry whose contents an API
 * actually returns — model.py, from /api/models/generate-code. The other four are
 * written server-side while the zip is assembled and no endpoint exposes them, so
 * they are listed and left unpreviewable rather than reconstructed here from a
 * second copy of the backend's templates that would silently drift out of date.
 */
type PackageEntry = { name: string; text?: string };

/**
 * Thrown for the one export problem the dialog can offer to fix for you, so the
 * message and the offer cannot drift apart.
 */
class MissingInputError extends Error {}

/** Sizes in this dialog are measured off the exact string being shown, so a row
 *  can never advertise a size the file does not have. */
function byteSize(text: string): number {
  return new Blob([text]).size;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const MONO = 'var(--font-mono)';

/**
 * The dialog's body lives in its own component so it mounts only while open:
 * `useBackendEnvironment()` probes /health, and an always-mounted export dialog
 * would fire that probe on every builder mount for a dialog nobody opened.
 */
const ExportPackageBody: React.FC<Omit<CodeExportDialogProps, 'isOpen'>> = ({
  onClose,
  onAddInputNode,
  graph,
  graphName = 'GeneratedModel',
}) => {
  const env = useBackendEnvironment();
  const [code, setCode] = useState('');
  const [className, setClassName] = useState('');
  const [inputShape, setInputShape] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  // Only set for the problem the dialog can offer to fix, so the offer cannot
  // appear beside a message it does not answer.
  const [canAddInput, setCanAddInput] = useState(false);
  const [copied, setCopied] = useState(false);
  const [unsupported, setUnsupported] = useState<UnsupportedModule[]>([]);
  const [selectedName, setSelectedName] = useState('model.py');
  const [zip, setZip] = useState<Blob | null>(null);
  const [zipError, setZipError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setCode('');
    setError('');
    setCopied(false);
    setZip(null);
    setZipError('');
    setUnsupported([]);

    // Generate a clean class name from graph name
    const modelClassName = graphName
      .replace(/[^a-zA-Z0-9]/g, '')
      .replace(/^[0-9]/, 'Model$&') // Prefix with 'Model' if starts with number
      || 'GeneratedModel';
    setClassName(modelClassName);

    let shape: number[];
    try {
      // The Input node has to FEED something, not merely exist. This used to be a
      // bare `.find`, so an Input dropped on the canvas and left unconnected
      // satisfied the gate and the export went ahead against a shape that reached
      // none of the layers — the gate checked presence and called it connectivity.
      const inputNode = graph.nodes.find((n) => n.data.layerType.id === 'input');
      if (!inputNode || !inputNode.data.parameters.shape) {
        throw new MissingInputError(
          'This model has no Input node, so there is nothing to say what shape goes '
          + 'into it. Validate falls back to 1,3,224,224; Export will not guess.'
        );
      }
      const feedsSomething = graph.edges.some((edge) => edge.source === inputNode.id);
      if (!feedsSomething && graph.nodes.length > 1) {
        throw new MissingInputError(
          'The Input node is not connected to anything, so its shape reaches none of '
          + 'your layers. Wire it to the first layer in the chain.'
        );
      }

      // Refuse rather than hand over code that cannot run. Exporting a graph whose
      // layers are missing required parameters produced `nn.Linear()`, offered with
      // Copy and Download buttons, which raises TypeError on the first line that
      // uses it.
      const incomplete = findMissingRequiredParameters(graph.nodes as any);
      if (incomplete.length) {
        throw new Error(
          `Can't export yet — ${incomplete.length} layer${incomplete.length > 1 ? 's are' : ' is'} `
          + `missing required parameters: ${describeMissingParameters(incomplete)}. `
          + 'Fill them in on the canvas, then export.'
        );
      }

      // Tolerant of both stored forms. Calling .split on the raw value crashed export
      // outright when an AI-created graph held a list: "shape.split is not a function".
      shape = parseInputShape(inputNode.data.parameters.shape);
    } catch (err) {
      setInputShape(null);
      setCanAddInput(err instanceof MissingInputError);
      setError(err instanceof Error ? err.message : 'Failed to generate code');
      return;
    }
    setCanAddInput(false);
    setInputShape(shape);

    setIsLoading(true);
    ApiService.generateCode({ graph, input_shape: shape, class_name: modelClassName })
      .then((response) => {
        if (cancelled) return;
        setCode(response.code);
        setUnsupported(response.unsupported || []);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('Code generation failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to generate code');
      })
      .then(() => {
        if (!cancelled) setIsLoading(false);
      });

    // The zip is fetched up front so the foot can state the size of the very bytes
    // the download hands over, instead of a figure derived from the preview. It is
    // also the blob Download reuses, so the number and the file cannot disagree.
    ApiService.exportPackage(graph, shape, modelClassName)
      .then((blob) => {
        if (!cancelled) setZip(blob);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('Export package failed:', err);
        setZipError(err instanceof Error ? err.message : 'Package request failed');
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphName, attempt]);

  const entries: PackageEntry[] = useMemo(
    () => [
      { name: 'model.py', text: code || undefined },
      { name: '__init__.py' },
      { name: 'sample_run.py' },
      { name: 'README.md' },
      { name: 'requirements.txt' },
    ],
    [code]
  );

  const selected = entries.find((e) => e.name === selectedName) || entries[0];

  /**
   * /api/models/export-package takes graph, input_shape and class_name only — it
   * has no flags — so the three options the design lists are not settings this
   * dialog can send. They are reported as observed properties of the file that was
   * actually generated, which is a checkable claim, and left disabled.
   */
  const options = useMemo(() => {
    const shapeAssertions = /^\s*assert\b/m.test(code);
    const typeHints = /->\s*[A-Za-z_]|\bdef\s+\w+\([^)]*:\s*[A-Za-z_]/.test(code);
    // The modularizer factors repeated blocks into their own nn.Module classes,
    // when there are repeats to factor; more than one class means it happened.
    const subModuleClasses = (code.match(/^class\s+\w+\(nn\.Module\)/gm) || []).length > 1;
    return [
      {
        label: 'Shape assertions',
        on: shapeAssertions,
        note: shapeAssertions ? 'in file' : 'not emitted',
      },
      { label: 'Type hints', on: typeHints, note: typeHints ? 'in file' : 'not emitted' },
      {
        label: 'Sub-modules as classes',
        on: subModuleClasses,
        note: subModuleClasses ? 'factored' : 'no repeats',
      },
    ];
  }, [code]);

  const handleCopyFile = async () => {
    if (!selected.text) return;
    try {
      await hostServices.copyText(selected.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const saveZip = (blob: Blob) => {
    hostServices.saveFile(`${className}.zip`, blob);
  };

  const handleDownloadZip = async () => {
    if (zip) {
      saveZip(zip);
      return;
    }
    if (!inputShape) return;
    setZipError('');
    try {
      const blob = await ApiService.exportPackage(graph, inputShape, className);
      setZip(blob);
      saveZip(blob);
    } catch (err) {
      console.error('Export package failed:', err);
      setZipError(err instanceof Error ? err.message : 'Package request failed');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Prevent delete/backspace from bubbling up
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.stopPropagation();
    }

    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      onClose();
    }
  };

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  // The torch version is the backend's own report of itself. With no report there
  // is no clause: a fixed "torch 2.4.1" in the lede would be decoration shaped
  // like a fact about the machine.
  const lede = env.torch
    ? `A self-contained zip. Runs against torch ${env.torch} with no canvas dependency.`
    : 'A self-contained zip with no canvas dependency.';

  const zipLabel = zip
    ? `${className}.zip · ${formatBytes(zip.size)}`
    : zipError
      ? `${className}.zip · size unavailable`
      : `${className}.zip · measuring…`;

  return (
    <div
      className="ui2-dialog-backdrop"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="ui2-dialog"
        style={{ width: 860, maxWidth: '100%', maxHeight: '84vh' }}
        role="dialog"
        aria-modal="true"
        aria-label="Export model package"
        onKeyDown={handleKeyDown}
      >
        <div className="ui2-dialog-head">
          <div>
            <h2 className="ui2-dialog-title">Export model package</h2>
            <p className="ui2-dialog-lede">{lede}</p>
          </div>
          <button
            className="ui2-icon-button"
            style={{ marginLeft: 'auto', flex: '0 0 auto' }}
            onClick={onClose}
            aria-label="Close"
          >
            <LucideIcons.X size={16} />
          </button>
        </div>

        {/* A definite body height is what lets the two panes scroll independently:
            with an auto row the code pane grows to the length of the file and the
            whole dialog scrolls as one. It still shrinks under the 84vh cap. */}
        <div
          className="ui2-dialog-body"
          style={{
            display: 'grid',
            gridTemplateColumns: '232px minmax(0, 1fr)',
            gridTemplateRows: 'minmax(0, 1fr)',
            height: 470,
            overflow: 'hidden',
          }}
        >
          <div
            className="ui2-scroll-surface"
            style={{
              borderRight: '1px solid var(--ui2-rule)',
              padding: 14,
              minHeight: 0,
              overflow: 'auto',
            }}
          >
            <div className="ui2-eyebrow" style={{ marginBottom: 10 }}>Contents</div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {entries.map((entry) => {
                const isSelected = entry.name === selected.name;
                const previewable = Boolean(entry.text);
                return (
                  <button
                    key={entry.name}
                    onClick={() => setSelectedName(entry.name)}
                    disabled={!previewable}
                    title={previewable ? undefined : 'Written when the zip is built — not previewable here'}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 9,
                      padding: 8,
                      width: '100%',
                      textAlign: 'left',
                      borderRadius: 7,
                      cursor: previewable ? 'pointer' : 'default',
                      background: isSelected ? 'rgba(255,255,255,0.07)' : 'transparent',
                      boxShadow: isSelected ? 'inset 2px 0 0 var(--ui2-text)' : 'none',
                    }}
                  >
                    <span
                      style={{
                        fontFamily: MONO,
                        fontSize: 12.5,
                        color: isSelected
                          ? 'var(--ui2-text)'
                          : previewable
                            ? 'var(--ui2-text-dim)'
                            : 'var(--ui2-text-faint)',
                      }}
                    >
                      {entry.name}
                    </span>
                    <span
                      style={{
                        marginLeft: 'auto',
                        fontFamily: MONO,
                        fontSize: 10.5,
                        color: 'var(--ui2-text-dim)',
                      }}
                    >
                      {entry.text ? formatBytes(byteSize(entry.text)) : ''}
                    </span>
                  </button>
                );
              })}
            </div>
            <p className="ui2-muted" style={{ fontSize: 11, lineHeight: 1.5, margin: '8px 0 0' }}>
              Only model.py is returned by the API; the rest are written server-side.
            </p>

            <div className="ui2-eyebrow" style={{ margin: '20px 0 10px' }}>Options</div>
            {options.map((option) => (
              <div key={option.label}>
                <label className="ui2-check">
                  <input type="checkbox" checked={option.on} disabled readOnly />
                  <span>{option.label}</span>
                </label>
                <div
                  className="ui2-muted"
                  style={{ fontFamily: MONO, fontSize: 10.5, margin: '-4px 0 0 25px' }}
                >
                  {option.note}
                </div>
              </div>
            ))}
            <p className="ui2-muted" style={{ fontSize: 11, lineHeight: 1.5, margin: '10px 0 0' }}>
              Not switches — the endpoint takes none. These report the file.
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 16px',
                borderBottom: '1px solid var(--ui2-rule)',
              }}
            >
              <span style={{ fontFamily: MONO, fontSize: 11.5, color: 'var(--ui2-text-dim)' }}>
                {selected.name}
              </span>
              {/* Not "imports clean" — nothing here runs python. This is the one claim
                  the response supports: every module in the graph was reproduced, so
                  the file is the whole model rather than a partial reconstruction. */}
              {!isLoading && !error && code && unsupported.length === 0 && (
                <span
                  style={{
                    marginLeft: 'auto',
                    fontFamily: MONO,
                    fontSize: 10.5,
                    color: 'var(--ui2-green)',
                  }}
                >
                  ✓ all modules reproduced
                </span>
              )}
            </div>

            <div style={{ padding: error || unsupported.length ? '12px 16px 0' : 0 }}>
              {error && (
                <div
                  className="ui2-panel ui2-notice-error"
                  role="alert"
                  style={{ padding: 12, display: 'flex', gap: 10, alignItems: 'flex-start', fontSize: 12.5, lineHeight: 1.55 }}
                >
                  <LucideIcons.AlertCircle size={15} style={{ flexShrink: 0, marginTop: 2 }} />
                  <span style={{ minWidth: 0 }}>{error}</span>
                  {canAddInput && onAddInputNode && (
                    // An offer, not a dead end. The refusal used to name the
                    // problem and stop there, leaving the user to work out that
                    // the fix was a palette layer no tutorial mentioned.
                    <button
                      className="ui2-chip"
                      style={{ marginLeft: 'auto', flex: '0 0 auto' }}
                      onClick={() => {
                        onAddInputNode();
                        onClose();
                      }}
                    >
                      Add an Input node
                    </button>
                  )}
                  <button
                    className="ui2-chip"
                    style={{ marginLeft: canAddInput && onAddInputNode ? 0 : 'auto', flex: '0 0 auto' }}
                    onClick={retry}
                  >
                    Try again
                  </button>
                </div>
              )}

              {/* An imported model's blocks are library classes whose bodies are not in the
                  graph, so the export is a partial reconstruction. Saying so here is the
                  point: previously this dialog presented such output as the model. */}
              {!isLoading && !error && unsupported.length > 0 && (
                <div
                  className="ui2-panel ui2-notice-error"
                  role="alert"
                  style={{
                    display: 'flex', gap: 10, alignItems: 'flex-start',
                    padding: 12, fontSize: 12.5, lineHeight: 1.55,
                  }}
                >
                  <LucideIcons.AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 2 }} />
                  <span>
                    <strong>Incomplete export.</strong>{' '}
                    {unsupported.length} module{unsupported.length === 1 ? '' : 's'} could not be
                    reproduced because {unsupported.length === 1 ? 'its' : 'their'} implementation
                    is not part of the graph
                    {unsupported.some((u) => u.imported)
                      ? ' — imported models keep theirs in the source library.'
                      : '.'}{' '}
                    {unsupported.some((u) => u.imported)
                      ? 'Those classes raise instead of returning their input, so this model cannot silently run with the wrong architecture.'
                      : 'They are emitted as passthroughs for you to fill in.'}
                    <br />
                    <span style={{ opacity: 0.75, fontFamily: MONO, fontSize: 11.5 }}>
                      {/* Deduplicated with counts: a UNet lists ResnetBlock2D a dozen times
                          otherwise, which reads as noise and hides the other classes. */}
                      {Object.entries(
                        unsupported.reduce<Record<string, number>>((acc, u) => {
                          const key = u.source_class || u.op;
                          acc[key] = (acc[key] || 0) + 1;
                          return acc;
                        }, {})
                      )
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 6)
                        .map(([name, count]) => (count > 1 ? `${name} ×${count}` : name))
                        .join(', ')}
                    </span>
                  </span>
                </div>
              )}
            </div>

            <pre
              className="ui2-code-block ui2-scroll-surface"
              style={{
                margin: 0,
                padding: 16,
                flex: '1 1 auto',
                minHeight: 0,
                overflow: 'auto',
                background: 'var(--ui2-surface-code)',
                color: 'var(--ui2-text-dim)',
              }}
            >
              {isLoading ? 'Generating model.py…' : selected.text || ''}
            </pre>
          </div>
        </div>

        <div className="ui2-dialog-foot">
          <span
            style={{ fontFamily: MONO, fontSize: 11, color: 'var(--ui2-text-dim)', flex: '0 0 auto' }}
          >
            {zipLabel}
          </span>
          {zipError && (
            <span
              role="alert"
              title={zipError}
              style={{
                fontSize: 11,
                color: 'var(--ui2-red)',
                minWidth: 0,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {zipError}
            </span>
          )}
          <button
            className="ui2-chip"
            style={{
              marginLeft: 'auto',
              flex: '0 0 auto',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 7,
            }}
            onClick={handleCopyFile}
            disabled={!selected.text}
          >
            {copied ? <LucideIcons.Check size={14} /> : <LucideIcons.Copy size={14} />}
            {copied ? 'Copied' : 'Copy file'}
          </button>
          <button
            className="ui2-primary-action"
            onClick={handleDownloadZip}
            disabled={!inputShape}
          >
            <LucideIcons.Download size={14} />
            Download zip
          </button>
        </div>
      </div>
    </div>
  );
};

const CodeExportDialog: React.FC<CodeExportDialogProps> = ({ isOpen, ...rest }) => {
  if (!isOpen) return null;
  return <ExportPackageBody {...rest} />;
};

export default CodeExportDialog;
