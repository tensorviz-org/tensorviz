import fs from 'fs';
import path from 'path';

const UI2_DIR = path.resolve(__dirname, '../..', 'components', 'ui2');

type ButtonFinding = {
  file: string;
  tagPreview: string;
};

function getUi2TsxFiles(): string[] {
  return fs
    .readdirSync(UI2_DIR)
    .filter((file) => file.endsWith('.tsx'))
    .map((file) => path.join(UI2_DIR, file));
}

function findUngardedButtons(filePath: string): ButtonFinding[] {
  const source = fs.readFileSync(filePath, 'utf8');
  const findings: ButtonFinding[] = [];
  const buttonOpenTagRegex = /<button\b([\s\S]*?)>/g;
  let match: RegExpExecArray | null;

  while ((match = buttonOpenTagRegex.exec(source)) !== null) {
    const tag = match[0];
    const hasOnClick = /\bonClick\s*=/.test(tag);
    const hasDisabled = /\bdisabled\b/.test(tag);
    const isSubmit = /\btype\s*=\s*["']submit["']/.test(tag);
    // Drag-only buttons (e.g. palette items dragged onto the canvas) are actionable
    // through their onDragStart handler even without an onClick.
    const hasDragStart = /\bonDragStart\s*=/.test(tag);

    if (!hasOnClick && !hasDisabled && !isSubmit && !hasDragStart) {
      findings.push({
        file: filePath,
        tagPreview: tag.replace(/\s+/g, ' ').slice(0, 180),
      });
    }
  }

  return findings;
}

describe('UI v2 interaction audit', () => {
  it('ensures every ui2 button is actionable or explicitly disabled', () => {
    const files = getUi2TsxFiles();
    const violations = files.flatMap((file) => findUngardedButtons(file));

    if (violations.length > 0) {
      const details = violations
        .map((item) => `- ${path.relative(process.cwd(), item.file)} :: ${item.tagPreview}`)
        .join('\n');
      throw new Error(
        `Found ui2 button(s) without onClick/disabled/submit guard:\n${details}`
      );
    }

    expect(violations).toHaveLength(0);
  });
});

