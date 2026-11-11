/** Shared, deterministic text layout for card bounds, rendering and SVG export.
 * Lines are explicit so zoom and browser font measurement cannot change geometry.
 * Conservative advances leave room for the bundled Inter/JetBrains Mono fonts.
 */
const characterAdvance = (character: string, size: number, mono: boolean): number => {
  // Non-Latin identifiers may use a full-width fallback even in a mono family.
  if (character.codePointAt(0)! > 127) return size * 1.05;
  if (mono) return size * 0.64;
  if (/[MWmw@%&#]/.test(character)) return size * 1.02;
  if (/[ilI.,:;'!|\s]/.test(character)) return size * 0.4;
  return size * (/[A-Z0-9]/.test(character) ? 0.8 : 0.68);
};

export function wrapCardText(value: string, width: number, size: number, mono = false): string[] {
  const lines: string[] = [];
  let line = '', used = 0;
  for (const character of value) {
    if (character === '\n') { lines.push(line); line = ''; used = 0; continue; }
    const advance = characterAdvance(character, size, mono);
    if (line && used + advance > width) {
      // Prefer a word, path or CamelCase boundary while preserving every character.
      const boundaries = [...line.matchAll(/[._/\s]+|[a-z0-9](?=[A-Z])/g)];
      const last = boundaries[boundaries.length - 1];
      const boundary = last ? last.index! + last[0].length : line.length;
      const remainder = line.slice(boundary);
      const remainderWidth = [...remainder].reduce((sum, char) => sum + characterAdvance(char, size, mono), 0);
      // A very early separator can leave a remainder that still cannot fit the
      // next glyph. In that case keep the already fitting line intact.
      if (remainderWidth + advance > width) {
        lines.push(line); line = ''; used = 0;
      } else {
        lines.push(line.slice(0, boundary)); line = remainder; used = remainderWidth;
      }
    }
    line += character; used += advance;
  }
  if (line || !lines.length) lines.push(line);
  return lines;
}

export function cardHeaderLayout(title: string, subtitle?: string, width = 200, module = false) {
  // Icon, gaps, padding and a separate 24px status/action column.
  const textWidth = Math.max(40, width - (module ? 148 : 94));
  const titleLines = wrapCardText(title, textWidth, 14);
  const subtitleLines = subtitle ? wrapCardText(subtitle, textWidth, 11, true) : [];
  return { titleLines, subtitleLines, height: Math.max(module ? 56 : 51, 20 + titleLines.length * 18 + subtitleLines.length * 14) };
}
