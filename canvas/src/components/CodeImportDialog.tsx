import React, { useState, useRef } from 'react';
import { findMissingRequiredParameters, describeMissingParameters } from '../utils/requiredParameters';
import * as LucideIcons from 'lucide-react';
import { ModelGraph, ModelNode, ModelEdge } from '../types';
import { ApiService } from '../services/apiService';
import { getLayerByType } from '../data/layerTypes';

interface CodeImportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (graph: ModelGraph, className: string) => void;
}

/**
 * The importer used to surface the transport line verbatim — "API Error (400): Code
 * parsing error: invalid syntax (<unknown>, line 1)" — to someone who had just pasted
 * a sentence into a box labelled "Import Model". Say what is wrong and what to do.
 */
const importErrorMessage = (err: unknown): string => {
  const raw = err instanceof Error ? err.message : String(err);
  if (/parsing error|invalid syntax|SyntaxError/i.test(raw)) {
    return "That doesn't look like valid Python. Paste the contents of a .py file that "
      + 'defines a class inheriting from nn.Module — a plain description of a model '
      + "won't work here.";
  }
  if (/No PyTorch model classes found/i.test(raw)) {
    return 'No nn.Module class found in that code. The importer needs a class that '
      + 'inherits from nn.Module, with its layers assigned in __init__.';
  }
  return raw.replace(/^API Error \(\d+\):\s*/, '');
};

const CodeImportDialog: React.FC<CodeImportDialogProps> = ({
  isOpen,
  onClose,
  onImport,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [code, setCode] = useState('');
  const [inputShape, setInputShape] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset state when dialog opens
  React.useEffect(() => {
    if (isOpen) {
      setSelectedFile(null);
      setCode('');
      setInputShape('');
      setError('');
      setSuccess(false);
    }
  }, [isOpen]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.name.endsWith('.py')) {
        setSelectedFile(file);
        setError('');
        
        // Read file content
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          setCode(content);
        };
        reader.readAsText(file);
      } else {
        setError('Please select a Python (.py) file');
      }
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.name.endsWith('.py')) {
        setSelectedFile(file);
        setError('');
        
        // Read file content
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          setCode(content);
        };
        reader.readAsText(file);
      } else {
        setError('Please select a Python (.py) file');
      }
    }
  };

  const handleImport = async () => {
    if (!code.trim()) {
      setError('Please select a Python file or paste code');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const rawInputShape = inputShape.trim();
      const parsedInputShape = rawInputShape
        ? rawInputShape.split(',').map((value) => Number(value.trim()))
        : [];

      if (parsedInputShape.some((value) => !Number.isInteger(value) || value <= 0)) {
        setError('Input shape must be a comma-separated list of positive integers');
        setIsLoading(false);
        return;
      }

      const request = {
        code: code.trim(),
        ...(parsedInputShape.length > 0 ? { input_shape: parsedInputShape } : {}),
      };

      const response = await ApiService.parseCode(request);

      // Convert response to ModelGraph format. The backend's parser returns
      // a stub layerType ({id, name, parameters: []}) and positional args as
      // arg_0/arg_1/... — hydrate from the canonical layerTypes schema so the
      // ParameterEditor has real fields and the imported values land on the
      // right keys.
      const nodes: ModelNode[] = response.nodes.map((node: any) => {
        const rawLayerType = node.data.layerType;
        const rawParams: Record<string, any> = node.data.parameters || {};
        const fullLayerType = rawLayerType?.id ? getLayerByType(rawLayerType.id) : undefined;

        let parameters: Record<string, any> = rawParams;
        if (fullLayerType?.parameters?.length) {
          const remapped: Record<string, any> = {};
          Object.entries(rawParams).forEach(([key, val]) => {
            const m = key.match(/^arg_(\d+)$/);
            if (m) {
              const idx = parseInt(m[1], 10);
              const paramDef = fullLayerType.parameters[idx];
              if (paramDef) {
                remapped[paramDef.name] = val;
                return;
              }
            }
            remapped[key] = val;
          });
          parameters = remapped;
        }

        // Input shape comes as a number[] from the backend; the input
        // parameter schema and InputNode renderer expect a comma-separated
        // string.
        if (rawLayerType?.id === 'input' && Array.isArray(parameters.shape)) {
          parameters = { ...parameters, shape: parameters.shape.join(',') };
        }

        return {
          id: node.id,
          type: node.type,
          position: node.position,
          data: {
            layerType: fullLayerType || rawLayerType,
            parameters,
          },
        };
      });

      const edges: ModelEdge[] = response.edges.map((edge: any) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        data: {},
      }));

      const graph: ModelGraph = { nodes, edges };

      // Constructor arguments the file never gives a value to (`nn.Linear(784,
      // hidden_dim)` with no default) arrive empty. Say so now: the import reported
      // success and the card simply read "Linear #fc2" with no parameters.
      const incomplete = findMissingRequiredParameters(nodes as any);
      if (incomplete.length) {
        window.dispatchEvent(new CustomEvent('builder-toast', {
          detail: {
            label: 'Imported with gaps',
            detail: `${describeMissingParameters(incomplete)} — the code did not say what `
              + 'these are. Fill them in before exporting or validating.',
            status: 'warn',
          },
        }));
      }

      onImport(graph, response.class_name);
      setSuccess(true);

      // Say so when the file defined more than one nn.Module. Only one is imported,
      // and staying silent about the others read as "all of it came across". This
      // goes out as a builder toast rather than dialog-local text because onImport
      // closes the dialog immediately, so nothing rendered here would be seen.
      const others = (response.available_classes || []).filter((name) => name !== response.class_name);
      if (others.length) {
        window.dispatchEvent(new CustomEvent('builder-toast', {
          detail: {
            label: `Imported ${response.class_name}`,
            detail: `This file also defines ${others.join(', ')}; `
              + `${others.length > 1 ? 'those were' : 'that was'} not imported.`,
            status: 'warn',
          },
        }));
      }

      // Close dialog after a short delay
      setTimeout(() => {
        onClose();
      }, 1000);

    } catch (err) {
      console.error('Code import failed:', err);
      setError(importErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.stopPropagation();
    }
    
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      onClose();
    }
  };

  const handleCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCode(e.target.value);
    setError('');
  };

  if (!isOpen) return null;

  return (
    <div className="code-export-dialog-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="code-export-dialog" onKeyDown={handleKeyDown}>
        <div className="dialog-header">
          <div className="dialog-title">
            <LucideIcons.Upload size={20} />
            <h3>Import PyTorch Model</h3>
          </div>
          <button className="close-button" onClick={onClose}>
            <LucideIcons.X size={20} />
          </button>
        </div>

        <div className="dialog-content">
          <div className="import-scope-note">
            Best-effort import supports straightforward <code>nn.Module</code> and{' '}
            <code>nn.Sequential</code> models. Custom modules, dynamic control flow, and
            advanced <code>forward</code> logic may need manual cleanup after import.
          </div>

          {/* File Upload Section */}
          <div className="form-group">
            <label className="form-label">Upload Python File</label>
            <div 
              className="file-upload-area"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                accept=".py"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              {selectedFile ? (
                <div className="file-selected">
                  <LucideIcons.File size={24} />
                  <span>{selectedFile.name}</span>
                </div>
              ) : (
                <div className="file-upload-placeholder">
                  <LucideIcons.Upload size={24} />
                  <p>Click to select a .py file or drag and drop</p>
                </div>
              )}
            </div>
          </div>

          {/* Code Preview/Editor */}
          <div className="form-group">
            <label className="form-label">PyTorch Model Code</label>
            <textarea
              value={code}
              onChange={handleCodeChange}
              onKeyDown={handleKeyDown}
              className="code-textarea"
              rows={10}
              placeholder="Paste your PyTorch model code here..."
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="code-import-input-shape">
              Input Shape <span className="form-label-optional">optional</span>
            </label>
            <input
              id="code-import-input-shape"
              className="form-input"
              value={inputShape}
              onChange={(event) => {
                setInputShape(event.target.value);
                setError('');
              }}
              placeholder="1, 3, 224, 224"
            />
            <p className="form-help">
              Enter the shape your model expects—for example, <code>1, 784</code> for an MLP or{' '}
              <code>1, 3, 32, 32</code> for a CNN. Leave blank to use best-effort detection.
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <LucideIcons.AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="success-message">
              <LucideIcons.CheckCircle size={16} />
              <span>Model imported successfully!</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="dialog-actions">
            <button 
              className="cancel-button"
              onClick={onClose}
              disabled={isLoading}
            >
              Cancel
            </button>
            <button 
              className="import-button"
              onClick={handleImport}
              disabled={!code.trim() || isLoading}
            >
              {isLoading ? (
                <>
                  <LucideIcons.Loader2 size={16} className="spinning" />
                  Importing...
                </>
              ) : (
                <>
                  <LucideIcons.Download size={16} />
                  Import Model
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodeImportDialog;
