// Near-monochrome Shiki themes: foreground text with gray comments / strings / punctuation. No hue.
const make = (name: string, type: 'light' | 'dark', c: { fg: string; bg: string; comment: string; string: string; punct: string; keyword: string }) => ({
  name, type, colors: { 'editor.foreground': c.fg, 'editor.background': c.bg },
  fg: c.fg, bg: c.bg,
  settings: [
    { settings: { foreground: c.fg, background: c.bg } },
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: c.comment } },
    { scope: ['string', 'string.quoted', 'punctuation.definition.string', 'constant.other.symbol'], settings: { foreground: c.string } },
    { scope: ['punctuation', 'meta.brace', 'keyword.operator', 'punctuation.separator', 'keyword.operator.assignment', 'keyword.operator.logical.shell'], settings: { foreground: c.punct } },
    { scope: ['keyword', 'storage', 'storage.type', 'keyword.control', 'constant.language', 'support.type.property-name', 'entity.name.tag'], settings: { foreground: c.keyword } },
    { scope: ['constant.numeric', 'constant.character', 'variable.other.constant'], settings: { foreground: c.string } },
  ],
})
export const monoLight = make('tv-mono-light', 'light', { fg: '#171717', bg: '#fafafa', comment: '#8a8a8a', string: '#525252', punct: '#737373', keyword: '#171717' })
export const monoDark = make('tv-mono-dark', 'dark', { fg: '#ededed', bg: '#141414', comment: '#7a7a7a', string: '#a1a1a1', punct: '#8f8f8f', keyword: '#ededed' })
