/**
 * The Inspector's Properties fields.
 *
 * Until now this was a generic `key -> text input` loop over whatever values the
 * node happened to carry, so it never consulted the catalogue's Parameter
 * metadata: no types, no option lists, no bounds, and no way to express that two
 * arguments exclude each other. That last one is not cosmetic — F.interpolate
 * takes `size` XOR `scale_factor` and raises when given both, so rendering every
 * field unconditionally made a broken graph the DEFAULT outcome.
 *
 * Two things live here:
 *
 *  - metadata-driven fields, honouring `Parameter.type`, `options` and `showIf`;
 *  - a range control for Slice, so a slice is chosen on the axis rather than
 *    typed as three numbers the user has to check against a remembered shape.
 *
 * Anything without catalogue metadata still falls back to the plain text input,
 * so no existing op loses its editor.
 */
import React, { useMemo, useState } from 'react';
import { getLayerByType } from '../../data/layerTypes';
import { Parameter } from '../../types';

export interface ParameterFieldsProps {
  /** The catalogue id, e.g. 'narrow' — the selected node's layer type. */
  layerTypeId: string;
  /** Known only after shape inference has run; the range control needs it. */
  inputShape?: number[];
  /** Draft values, as strings — the Inspector edits a draft and applies on demand. */
  values: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}

/* ── showIf ──────────────────────────────────────────────────────────────── */

/** Presentational only. The backend still enforces the real constraint. */
export function isParameterVisible(param: Parameter, values: Record<string, string>): boolean {
  const rule = param.showIf;
  if (!rule) return true;
  const actual = values[rule.param];
  if (rule.oneOf) return rule.oneOf.some((candidate) => String(candidate) === String(actual));
  if ('equals' in rule) return String(rule.equals) === String(actual);
  return true;
}

/**
 * Blank out any value whose field is no longer shown. Codegen already skips
 * empty params, so clearing is what keeps the predicate from having to be
 * reimplemented in Python — see Parameter.showIf.
 */
export function clearHiddenParameters(
  params: Parameter[],
  values: Record<string, string>,
): Record<string, string> {
  let next = values;
  params.forEach((param) => {
    if (!isParameterVisible(param, values) && values[param.name] !== '' && values[param.name] !== undefined) {
      if (next === values) next = { ...values };
      next[param.name] = '';
    }
  });
  return next;
}

/* ── the slice range control ─────────────────────────────────────────────── */

/** What an <input type="number"> can actually display: one float, or nothing. */
const NUMBER_INPUT = /^$|^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

const AXIS_LABELS = ['N', 'C', 'H', 'W', 'D'];
const int = (value: string | undefined, fallback: number): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
};

interface SliceRangeProps {
  shape: number[];
  values: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
  /** Drop to plain number fields — for matching a paper exactly. */
  onUseExactNumbers: () => void;
}

const SliceRange: React.FC<SliceRangeProps> = ({ shape, values, onChange, onUseExactNumbers }) => {
  const rank = shape.length;
  // A negative dim is legal on the node but has no axis to highlight, so resolve
  // it before picking a chip.
  const rawDim = int(values.dim, 1);
  const dim = rawDim < 0 ? Math.max(0, rank + rawDim) : Math.min(rawDim, rank - 1);
  const size = shape[dim] ?? 1;
  const start = Math.min(Math.max(int(values.start, 0), 0), Math.max(0, size - 1));
  const length = Math.min(Math.max(int(values.length, 1), 1), size - start);
  const end = start + length;

  const commit = (nextDim: number, nextStart: number, nextLength: number) => {
    const clampedSize = shape[nextDim] ?? 1;
    const s = Math.min(Math.max(nextStart, 0), Math.max(0, clampedSize - 1));
    const l = Math.min(Math.max(nextLength, 1), clampedSize - s);
    onChange({ ...values, dim: String(nextDim), start: String(s), length: String(l) });
  };

  const pct = (value: number) => `${(value / size) * 100}%`;

  const dragFrom = (handle: 'start' | 'end') => (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const track = event.currentTarget.parentElement;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const valueAt = (clientX: number) =>
      Math.round(Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)) * size);

    const move = (moveEvent: PointerEvent) => {
      const v = valueAt(moveEvent.clientX);
      if (handle === 'start') commit(dim, Math.min(v, end - 1), end - Math.min(v, end - 1));
      else commit(dim, start, Math.max(1, v - start));
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const keyOn = (handle: 'start' | 'end') => (event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 8 : 1;
    const delta = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
    if (delta === 0) return;
    event.preventDefault();
    if (handle === 'start') {
      const s = Math.min(Math.max(0, start + delta), end - 1);
      commit(dim, s, end - s);
    } else {
      commit(dim, start, length + delta);
    }
  };

  const presets: Array<[string, () => void]> = [
    ['First half', () => commit(dim, 0, Math.max(1, Math.floor(size / 2)))],
    ['Second half', () => {
      const half = Math.floor(size / 2);
      commit(dim, half, size - half);
    }],
    ['Drop first', () => commit(dim, 1, Math.max(1, size - 1))],
    ['All', () => commit(dim, 0, size)],
  ];

  return (
    <div className="ui2-slice">
      <span className="ui2-slice-label">Axis — from the input shape</span>
      <div className="ui2-slice-axes">
        {shape.map((extent, index) => (
          <button
            key={index}
            type="button"
            className={`ui2-slice-axis${index === dim ? ' on' : ''}`}
            // The batch axis is not a meaningful thing to slice on this canvas.
            disabled={index === 0}
            title={index === 0 ? 'The batch axis is not sliceable here' : `Slice on ${AXIS_LABELS[index] || index}`}
            onClick={() => commit(index, 0, Math.min(length, extent))}
          >
            {extent}
            <small>{AXIS_LABELS[index] || index}</small>
          </button>
        ))}
      </div>

      <span className="ui2-slice-label">{`Range on ${AXIS_LABELS[dim] || dim} · 0…${size}`}</span>
      {/* The track IS the bound, so start + length > size cannot be expressed. */}
      <div className="ui2-slice-track">
        <div className="ui2-slice-band" style={{ left: pct(start), width: pct(length) }} />
        <div
          className="ui2-slice-grab"
          style={{ left: pct(start) }}
          role="slider"
          tabIndex={0}
          aria-label="Range start"
          aria-valuemin={0}
          aria-valuemax={size}
          aria-valuenow={start}
          onPointerDown={dragFrom('start')}
          onKeyDown={keyOn('start')}
        />
        <div
          className="ui2-slice-grab"
          style={{ left: pct(end) }}
          role="slider"
          tabIndex={0}
          aria-label="Range end"
          aria-valuemin={0}
          aria-valuemax={size}
          aria-valuenow={end}
          onPointerDown={dragFrom('end')}
          onKeyDown={keyOn('end')}
        />
      </div>
      <div className="ui2-slice-read">
        <span><b>{start}</b> → <b>{end}</b></span>
        <span>{length} of {size}</span>
      </div>

      <span className="ui2-slice-label">Common intents</span>
      <div className="ui2-slice-presets">
        {presets.map(([label, apply]) => (
          <button key={label} type="button" className="ui2-slice-preset" onClick={apply}>
            {label}
          </button>
        ))}
      </div>

      <div className="ui2-slice-emits">{`torch.narrow(x, ${dim}, ${start}, ${length})`}</div>

      <button type="button" className="ui2-slice-exact" onClick={onUseExactNumbers}>
        Type exact numbers instead
      </button>
    </div>
  );
};

/* ── the fields ──────────────────────────────────────────────────────────── */

export const ParameterFields: React.FC<ParameterFieldsProps> = ({
  layerTypeId,
  inputShape,
  values,
  onChange,
}) => {
  const definition = useMemo(() => getLayerByType(layerTypeId), [layerTypeId]);
  const params = definition?.parameters || [];
  const [exact, setExact] = useState(false);

  const update = (name: string, value: string) => {
    const next = { ...values, [name]: value };
    onChange(clearHiddenParameters(params, next));
  };

  // Slice, with a known shape: the whole point is not to type three numbers.
  // `exact` is the escape hatch out of it, for matching a paper verbatim.
  if (layerTypeId === 'narrow' && inputShape && inputShape.length > 1 && !exact) {
    return (
      <SliceRange
        shape={inputShape}
        values={values}
        onChange={onChange}
        onUseExactNumbers={() => setExact(true)}
      />
    );
  }

  const byName = new Map(params.map((param) => [param.name, param]));
  // Iterate the VALUES, so a parameter the node carries but the catalogue does
  // not describe still gets an editor — no op loses its properties panel.
  const rows = Object.keys(values).filter((key) => {
    const param = byName.get(key);
    return !param || isParameterVisible(param, values);
  });

  return (
    <>
      {layerTypeId === 'narrow' && (
        <p className="ui2-muted ui2-slice-hint">
          Run <strong>Refresh shapes</strong> to pick the range on the axis visually.
        </p>
      )}
      {rows.map((key) => {
        const param = byName.get(key);
        const value = values[key] ?? '';
        return (
          <label key={key} className="flex items-center justify-between gap-2">
            <span className="ui2-muted" title={param?.description}>{key}</span>
            {param?.type === 'select' && param.options ? (
              <select
                value={value}
                onChange={(event) => update(key, event.target.value)}
                className="ui2-inspector-input"
                aria-label={`parameter-${key}`}
              >
                {param.options.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            ) : param?.type === 'boolean' ? (
              <select
                // Case-INSENSITIVE. The draft is seeded through formatParamValue,
                // which spells booleans the way Python does ('True' — the same form
                // the card and the code snippet use), so comparing against the
                // lowercase form rendered every ENABLED flag as False. And it could
                // not be corrected by hand: a <select> fires onChange only when the
                // DOM selection actually moves, so picking "False" on a control
                // already showing False did nothing at all.
                value={/^true$/i.test(String(value)) ? 'true' : 'false'}
                onChange={(event) => update(key, event.target.value)}
                className="ui2-inspector-input"
                aria-label={`parameter-${key}`}
              >
                <option value="true">True</option>
                <option value="false">False</option>
              </select>
            ) : (
              <input
                value={value}
                // A number input BLANKS anything that is not a single float — so a
                // kernel_size the node holds as [7, 7], which every imported conv
                // layer and the shipped ResNet-50/VGG-16 templates carry, rendered
                // as an empty required-looking field. Worse, correcting it by typing
                // 3 dropped the asymmetry of a [3, 5]. Fall back to a text input
                // when the value is not something a number input can show.
                type={param?.type === 'number' && NUMBER_INPUT.test(value) ? 'number' : 'text'}
                min={param?.min}
                max={param?.max}
                onChange={(event) => update(key, event.target.value)}
                className="ui2-inspector-input"
                aria-label={`parameter-${key}`}
              />
            )}
          </label>
        );
      })}
    </>
  );
};

export default ParameterFields;
