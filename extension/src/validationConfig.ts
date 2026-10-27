/** Navigation only: the Python resolver remains the TOML/schema authority. */
export interface FieldLocation { start: number; end: number; exact: boolean }
interface Token { text: string; start: number; end: number }

/** Index keys and array elements without mistaking strings/comments for TOML keys.
 * Invalid/incomplete documents fall back to the closest containing field. */
export function locateConfigField(source: string, field: (string | number)[]): FieldLocation {
  if (source.length > 256 * 1024) return { start: 0, end: 0, exact: false };
  const tokens: Token[] = [];
  const pattern = /#[^\r\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\[\s\S]|[^"\\])*"|'[^']*'|\r?\n|[^\S\r\n]+|[\[\]{}=,.]|[^\s\[\]{}=,.#]+/g;
  for (const match of source.matchAll(pattern)) {
    if (match[0].startsWith('#') || /^[^\S\r\n]+$/.test(match[0])) continue;
    tokens.push({ text: match[0], start: match.index!, end: match.index! + match[0].length });
  }
  const locations = new Map<string, Token>();
  const key = (path: (string | number)[]) => JSON.stringify(path);
  let i = 0;
  const at = () => tokens[i]?.text;
  function keys(): string[] {
    const parts: string[] = [];
    while (i < tokens.length) {
      const text = at();
      if (!text || ['\n', '\r\n', '=', ']', '}', ','].includes(text)) break;
      if (text === '.') { i++; continue; }
      if (text.startsWith('"')) {
        try { parts.push(JSON.parse(text)); } catch { parts.push(text); }
      } else parts.push(text.startsWith("'") ? text.slice(1, -1) : text);
      i++;
    }
    return parts;
  }
  function value(path: (string | number)[], depth = 0): void {
    if (i >= tokens.length || depth > 32) return;
    if (at() === '{') {
      i++;
      while (i < tokens.length && at() !== '}') {
        if ([',', '\n', '\r\n'].includes(at())) { i++; continue; }
        const start = tokens[i], parts = keys();
        if (at() !== '=') break;
        i++;
        const child = [...path, ...parts];
        locations.set(key(child), start);
        value(child, depth + 1);
      }
      if (at() === '}') i++;
    } else if (at() === '[') {
      i++;
      let index = 0;
      while (i < tokens.length && at() !== ']') {
        if ([',', '\n', '\r\n'].includes(at())) { i++; continue; }
        const child = [...path, index++];
        locations.set(key(child), tokens[i]);
        const before = i;
        value(child, depth + 1);
        if (i === before) break;
      }
      if (at() === ']') i++;
    } else {
      // A scalar may contain punctuation (floats, dates). Strings are one token.
      while (i < tokens.length && ![',', '}', ']', '\n', '\r\n'].includes(at())) i++;
    }
  }
  let table: (string | number)[] = [];
  const arrayTables = new Map<string, number>();
  while (i < tokens.length) {
    if (['\n', '\r\n'].includes(at())) { i++; continue; }
    const start = tokens[i];
    if (at() === '[') {
      i++;
      const array = at() === '[';
      if (array) i++;
      const parts = keys();
      table = [];
      for (const part of parts) {
        table.push(part);
        const name = key(table);
        if (array && table.filter(value => typeof value === 'string').length === parts.length) {
          arrayTables.set(name, (arrayTables.get(name) ?? -1) + 1);
        }
        const index = arrayTables.get(name);
        if (index !== undefined) table.push(index);
      }
      if (at() === ']') i++;
      if (array && at() === ']') i++;
      locations.set(key(table), start);
    } else {
      const parts = keys();
      if (at() === '=') {
        i++;
        const path = [...table, ...parts];
        locations.set(key(path), start);
        value(path);
      } else i++;
    }
  }
  for (let length = field.length; length > 0; length--) {
    const token = locations.get(key(field.slice(0, length)));
    if (token) return { start: token.start, end: token.end, exact: length === field.length };
  }
  return { start: 0, end: 0, exact: false };
}

/** Deliberately incomplete: no model, inputs, mode or capture choice is guessed. */
export const CONFIG_STARTER = `# Fill the required fields for your selected script or model profile.
# Run is explicit; editing and saving this file never execute project code.
version = 1

[profiles.run]
# Script: kind, script, args, capture.model, capture.call
# Model: kind, model, mode, device, grad_enabled, init, inputs
# Complex setup belongs in an ordinary Python script.
`;

export function modelConfigStarter(model: string): string {
  return `# Complete this example call for the selected model; no coding agent is required.
# Saving a complete profile during graph setup continues opening the graph.
version = 1

[profiles.example]
kind = "model"
model = ${JSON.stringify(model)}
# Explicit example execution settings; change these to suit your model.
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 0

# Required: add constructor arguments and forward inputs for this model.
# init = { args = [], kwargs = {} }
# inputs = { args = [], kwargs = {} }
# For a tensor, specify kind, shape, dtype and generator inside args or kwargs.
# Use the Canvas form if you need help with these fields.
# Configuration objects or checkpoint loading can use a Python factory instead of init:
# factory = { file = "model_setup.py", callable = "make_model", kwargs = {} }
`;
}
