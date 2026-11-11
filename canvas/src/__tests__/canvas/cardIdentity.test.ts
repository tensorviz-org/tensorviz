import { cardHeaderLayout, wrapCardText } from '../../canvas/cardIdentity';

describe('shared card identity wrapping', () => {
  it('does not overflow a line when an early separator leaves a wide remainder', () => {
    const value = '.WWWWWWWWW';
    const lines = wrapCardText(value, 40, 14);
    expect(lines.join('')).toBe(value);
    expect(lines[0]).toBe('.WW');
    expect(lines).not.toContain('WWW');
  });

  it.each(['mmmmmmmmmmmm', 'wwwwwwwwwwww', 'MWmwMWmwMWmw'])('reserves enough room for wide Latin names: %s', value => {
    const lines = wrapCardText(value, 40, 14);
    expect(lines.join('')).toBe(value);
    expect(lines.every(line => [...line].length <= 2)).toBe(true);
  });

  it('keeps full-width callable identifiers safe in the mono subtitle too', () => {
    const value = '模型.时间候选分数';
    const lines = wrapCardText(value, 40, 11, true);
    expect(lines.join('')).toBe(value);
    // A full-width glyph has at least the same reserved advance in either font.
    expect(wrapCardText('模型时间候选', 40, 11, true)).toEqual(wrapCardText('模型时间候选', 40, 11));
    expect(lines.every(line => [...line].length <= 3)).toBe(true);
  });

  it('prefers readable boundaries without losing punctuation or name characters', () => {
    const value = 'models.video.CustomTemporalScoringBlock';
    const lines = wrapCardText(value, 90, 14);
    expect(lines.join('')).toBe(value);
    expect(lines).toContain('models.');
    expect(lines).toContain('Custom');
    expect(lines.some(line => line.includes('…'))).toBe(false);
  });

  it('reserves every wrapped title and technical subtitle line in header height', () => {
    const title = 'WNCausalConvolutionWithAdaptiveTemporalPadding';
    const subtitle = `video.audio.${title}`;
    const layout = cardHeaderLayout(title, subtitle);
    expect(layout.titleLines.join('')).toBe(title);
    expect(layout.subtitleLines.join('')).toBe(subtitle);
    expect(layout.height).toBe(20 + layout.titleLines.length * 18 + layout.subtitleLines.length * 14);
  });
});
