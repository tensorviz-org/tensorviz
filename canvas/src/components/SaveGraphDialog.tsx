import React, { useState, useEffect } from 'react';
import * as LucideIcons from 'lucide-react';
import { SavedGraph } from '../types';

interface SaveGraphDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description?: string) => void;
  existingGraph?: SavedGraph | null;
  suggestedName?: string;
}

const SaveGraphDialog: React.FC<SaveGraphDialogProps> = ({
  isOpen,
  onClose,
  onSave,
  existingGraph,
  suggestedName,
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      if (existingGraph) {
        setName(existingGraph.name);
        setDescription(existingGraph.description || '');
      } else {
        setName(suggestedName || '');
        setDescription('');
      }
      setError('');
    }
  }, [isOpen, existingGraph, suggestedName]);

  const handleSave = () => {
    if (!name.trim()) {
      setError('Graph name is required');
      return;
    }

    if (name.trim().length < 3) {
      setError('Graph name must be at least 3 characters');
      return;
    }

    onSave(name.trim(), description.trim() || undefined);
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Prevent delete/backspace from bubbling up
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.stopPropagation();
    }
    
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      e.stopPropagation();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="save-dialog-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="save-dialog" onKeyDown={handleKeyDown}>
        <div className="dialog-header">
          <div className="dialog-title">
            <LucideIcons.Save size={20} />
            <h3>{existingGraph ? 'Update Graph' : 'Save Graph'}</h3>
          </div>
          <button className="close-button" onClick={onClose}>
            <LucideIcons.X size={20} />
          </button>
        </div>

        <div className="dialog-content">
          <div className="form-group">
            <label className="form-label">
              Graph Name <span className="required">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setError('');
              }}
              onKeyDown={handleKeyDown}
              className={`form-input ${error ? 'error' : ''}`}
              placeholder="Enter a name for your graph..."
              autoFocus
            />
            {error && <span className="form-error">{error}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Description (Optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onKeyDown={handleKeyDown}
              className="form-textarea"
              placeholder="Describe your model architecture..."
              rows={3}
            />
          </div>

          <div className="save-tips">
            <div className="tip-item">
              <LucideIcons.Lightbulb size={16} />
              <span>Tip: Use descriptive names like "ResNet-18 Classifier" or "Simple CNN"</span>
            </div>
            <div className="tip-item">
              <LucideIcons.Keyboard size={16} />
              <span>Press Ctrl+Enter to save quickly</span>
            </div>
          </div>
        </div>

        <div className="dialog-footer">
          <button className="cancel-button" onClick={onClose}>
            Cancel
          </button>
          <button className="save-button" onClick={handleSave}>
            <LucideIcons.Save size={16} />
            {existingGraph ? 'Update' : 'Save'} Graph
          </button>
        </div>
      </div>
    </div>
  );
};

export default SaveGraphDialog; 