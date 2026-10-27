import { object, type JsonObject } from './mcpGraph';

const list = (value: unknown): JsonObject[] => Array.isArray(value) ? value.map(object) : [];
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const metric = (value: number) => Number.isFinite(value) ? value : String(value);
type Value = number | bigint | [number, number];
const widths: Record<string, number> = { float16: 2, bfloat16: 2, float32: 4, float64: 8,
  int8: 1, uint8: 1, int16: 2, int32: 4, int64: 8, bool: 1, complex64: 8, complex128: 16 };
const integer = (dtype: unknown) => ['int8', 'uint8', 'int16', 'int32', 'int64', 'bool'].includes(String(dtype));
function smallFloat(bits: number, fractionBits: number, exponentBits: number, bias: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> fractionBits) & ((1 << exponentBits) - 1);
  const fraction = bits & ((1 << fractionBits) - 1);
  if (exponent === (1 << exponentBits) - 1) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? 2 ** (exponent - bias) * (1 + fraction / 2 ** fractionBits)
    : 2 ** (1 - bias) * fraction / 2 ** fractionBits);
}
function decoder(leaf: JsonObject): ((index: number) => Value) | undefined {
  const dtype = String(leaf.dtype), width = widths[dtype], values = object(leaf.values);
  if (!width || !Number.isSafeInteger(leaf.numel) || Number(leaf.numel) < 0 || values.encoding !== 'base64' ||
      !['little', 'big'].includes(String(values.byteOrder)) || typeof values.data !== 'string' ||
      values.data.length > 5_592_408 || values.data.length % 4 !== 0 || /[^A-Za-z0-9+/=]/.test(values.data)) return;
  const buffer = Buffer.from(values.data, 'base64');
  if (buffer.length !== Number(leaf.numel) * width || buffer.toString('base64') !== values.data) return;
  const little = values.byteOrder === 'little';
  const u16 = (offset: number) => little ? buffer.readUInt16LE(offset) : buffer.readUInt16BE(offset);
  const f32 = (offset: number) => little ? buffer.readFloatLE(offset) : buffer.readFloatBE(offset);
  const f64 = (offset: number) => little ? buffer.readDoubleLE(offset) : buffer.readDoubleBE(offset);
  return index => {
    const offset = index * width;
    switch (dtype) {
      case 'float16': return smallFloat(u16(offset), 10, 5, 15);
      case 'bfloat16': return smallFloat(u16(offset), 7, 8, 127);
      case 'float32': return f32(offset);
      case 'float64': return f64(offset);
      case 'complex64': return [f32(offset), f32(offset + 4)];
      case 'complex128': return [f64(offset), f64(offset + 8)];
      case 'int8': return buffer.readInt8(offset);
      case 'int16': return little ? buffer.readInt16LE(offset) : buffer.readInt16BE(offset);
      case 'int32': return little ? buffer.readInt32LE(offset) : buffer.readInt32BE(offset);
      case 'int64': return little ? buffer.readBigInt64LE(offset) : buffer.readBigInt64BE(offset);
      default: return buffer.readUInt8(offset);
    }
  };
}
const printable = (value: Value): unknown => typeof value === 'bigint' ? value.toString()
  : Array.isArray(value) ? value.map(metric) : metric(value);
const pair = (value: Value): [number, number] | undefined => typeof value === 'bigint'
  ? value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? [Number(value), 0] : undefined
  : Array.isArray(value) ? value : [value, 0];

function compareTensor(before: JsonObject, after: JsonObject, atol: number, rtol: number): JsonObject {
  const base = { path: before.path, baselineShape: before.shape, candidateShape: after.shape,
    baselineDtype: before.dtype, candidateDtype: after.dtype, dtypeChanged: before.dtype !== after.dtype,
    bitwiseEqual: typeof before.digest === 'string' && before.digest === after.digest,
    baselineNonfinite: before.nonfinite ?? null, candidateNonfinite: after.nonfinite ?? null };
  if (!same(before.shape, after.shape) || before.numel !== after.numel) return { ...base, status: 'shape_changed', allClose: false };
  if (Number(before.nonfinite) > 0 || Number(after.nonfinite) > 0) return { ...base, status: 'nonfinite', allClose: false };
  if (base.bitwiseEqual && before.nonfinite === 0 && after.nonfinite === 0) return { ...base, status: 'equal', allClose: true,
    comparedElements: before.numel, mismatchedElements: 0, maxAbsDifference: 0, maxRelativeDifference: 0 };
  const a = decoder(before), b = decoder(after);
  if (!a || !b) return { ...base, status: 'values_unavailable', allClose: null,
    baselineReason: before.unavailable ?? 'missing_or_invalid_values', candidateReason: after.unavailable ?? 'missing_or_invalid_values' };
  const exactIntegers = integer(before.dtype) && integer(after.dtype);
  let mismatched = 0, maxAbs = 0, maxRelative = 0, maxInteger = 0n;
  const examples: JsonObject[] = [];
  for (let index = 0; index < Number(before.numel); index++) {
    const av = a(index), bv = b(index);
    let close: boolean;
    if (exactIntegers) {
      const ai = BigInt(av as number | bigint), bi = BigInt(bv as number | bigint);
      const difference = ai > bi ? ai - bi : bi - ai;
      if (difference > maxInteger) maxInteger = difference;
      close = difference === 0n;
    } else {
      const ap = pair(av), bp = pair(bv);
      if (!ap || !bp) return { ...base, status: 'integer_precision_unavailable', allClose: null };
      const [ar, ai] = ap, [br, bi] = bp;
      if (![ar, ai, br, bi].every(Number.isFinite)) return { ...base, status: 'nonfinite', allClose: false };
      const difference = Math.hypot(ar - br, ai - bi);
      const reference = Math.hypot(ar, ai);
      maxAbs = Math.max(maxAbs, difference);
      maxRelative = Math.max(maxRelative, reference ? difference / reference : difference ? Infinity : 0);
      // Scaled arithmetic avoids overflow turning Infinity <= Infinity into an
      // accidental pass for otherwise finite float64/complex128 tensors.
      const scale = Math.max(Math.abs(ar), Math.abs(ai), Math.abs(br), Math.abs(bi), atol, Number.MIN_VALUE);
      close = Math.hypot(ar / scale - br / scale, ai / scale - bi / scale) <=
        atol / scale + rtol * Math.hypot(ar / scale, ai / scale);
    }
    if (!close) {
      mismatched++;
      if (examples.length < 8) examples.push({ flatIndex: index, baseline: printable(av), candidate: printable(bv) });
    }
  }
  return { ...base, status: mismatched ? 'different' : 'within_tolerance', allClose: mismatched === 0,
    method: exactIntegers ? 'exact_integer' : 'absolute_and_relative_tolerance', comparedElements: before.numel,
    mismatchedElements: mismatched, maxAbsDifference: exactIntegers ? maxInteger.toString() : metric(maxAbs),
    maxRelativeDifference: exactIntegers ? null : metric(maxRelative), examples };
}

/** Compare only retained values. Absence, sampling limits and NaNs cannot prove equivalence. */
export function compareOutputs(baseline: unknown, candidate: unknown, atol = 1e-6, rtol = 1e-5): JsonObject {
  const before = object(baseline), after = object(candidate);
  if (!Object.keys(before).length || !Object.keys(after).length) return { status: 'unavailable', allClose: null,
    detail: 'Both runs need captured final outputs.', items: [] };
  const a = list(before.leaves), b = list(after.leaves);
  const complete = Array.isArray(before.truncated) && Array.isArray(after.truncated) && before.truncated.length === 0 && after.truncated.length === 0;
  const structureEqual = same(before.containers, after.containers) && same(a.map(x => [x.path, x.kind]), b.map(x => [x.path, x.kind]));
  const byPath = new Map(b.map(leaf => [leaf.path, leaf]));
  const items = a.map(leaf => {
    const other = byPath.get(leaf.path);
    if (!other) return { path: leaf.path, status: complete ? 'structure_changed' : 'structure_unavailable', allClose: complete ? false : null };
    if (leaf.kind !== other.kind) return { path: leaf.path, status: 'structure_changed', allClose: false };
    if (leaf.kind === 'tensor') return compareTensor(leaf, other, atol, rtol);
    if (Number(leaf.nonfinite) > 0 || Number(other.nonfinite) > 0) return { path: leaf.path, status: 'nonfinite', allClose: false };
    if (leaf.kind === 'float' && typeof leaf.value === 'number' && typeof other.value === 'number') {
      const difference = Math.abs(leaf.value - other.value);
      const scale = Math.max(Math.abs(leaf.value), Math.abs(other.value), atol, Number.MIN_VALUE);
      return { path: leaf.path, status: 'scalar', allClose: Math.abs(leaf.value / scale - other.value / scale) <=
        atol / scale + rtol * Math.abs(leaf.value / scale), maxAbsDifference: metric(difference) };
    }
    return { path: leaf.path, status: 'literal', allClose: typeof leaf.digest === 'string' ? leaf.digest === other.digest : null };
  });
  for (const leaf of b) if (!a.some(item => item.path === leaf.path)) {
    items.push({ path: leaf.path, status: complete ? 'structure_changed' : 'structure_unavailable', allClose: complete ? false : null });
  }
  const allClose = complete && !structureEqual || items.some(item => item.allClose === false) ? false
    : !complete || items.some(item => item.allClose === null) ? null : true;
  return { status: allClose === true ? 'equivalent' : allClose === false ? 'different_or_nonfinite' : 'incomplete',
    allClose, structureEqual: complete ? structureEqual : null, baselineLeafCount: a.length, candidateLeafCount: b.length,
    baselineTruncation: before.truncated ?? null, candidateTruncation: after.truncated ?? null,
    bitwiseEqual: typeof before.digest === 'string' && before.digest === after.digest, atol, rtol,
    toleranceReference: 'baseline', integerComparison: 'exact', items };
}
