import { createDropoutInsertionReveal, isDropoutLayer, reviewAddsDropout } from '../../ide/dropoutInsertion';

const document = (nodes: any[]) => ({ semantic: { nodes } });
const state = (op = 'torch.nn.Dropout') => ({ before: { document: document([]) },
  after: { file: 'model.py', cls: 'MLP', sourceDigest: 'accepted', document: document([{ node_id: 'drop', op }]) } } as any);
const payload = (overrides = {}) => ({ ...state().after, traced: true, document: document([
  { node_id: 'drop', op: 'torch.nn.modules.dropout.Dropout' },
  { node_id: 'old', op: 'torch.nn.modules.dropout.Dropout' },
  { node_id: 'identity', op: 'torch.nn.modules.linear.Identity' },
]), ...overrides } as any);
const nodes = ['drop', 'old', 'identity'].map(id => ({ id, data: { capturePassThrough: { valueId: 'x' } } } as any));

test.each(['Dropout', 'nn.Dropout2d', 'torch.nn.AlphaDropout', 'FeatureAlphaDropout'])('%s gets insertion guidance', type => {
  expect(isDropoutLayer(type)).toBe(true);
  expect(reviewAddsDropout(state(type))).toBe(true);
});
test.each(['Linear', 'LayerNorm', 'MyDropoutWrapper'])('%s does not arm a Dropout reveal', type => {
  expect(reviewAddsDropout(state(type))).toBe(false);
});
test('accept reveals only new pass-through Dropout calls at the accepted capture, once', () => {
  const reveal = createDropoutInsertionReveal();
  expect(reveal.apply(payload(), nodes)).toBe(nodes);
  reveal.accept(state(), { document: document([{ node_id: 'old' }]) } as any);
  expect(reveal.apply(payload({ traced: false }), nodes)).toBe(nodes);
  expect(reveal.apply(payload({ sourceDigest: 'old' }), nodes)).toBe(nodes);
  expect(reveal.apply(payload({ greyed: true }), nodes)).toBe(nodes);
  const shown = reveal.apply(payload(), nodes);
  expect(shown[0].data.passThroughShown).toBe(true);
  expect(shown[1]).toBe(nodes[1]);
  expect(shown[2]).toBe(nodes[2]);
  expect(reveal.apply(payload(), nodes)).toBe(nodes);
});
test('discard and opening another model clear the pending reveal', () => {
  const reveal = createDropoutInsertionReveal();
  reveal.accept(state()); reveal.clear();
  expect(reveal.apply(payload(), nodes)).toBe(nodes);
  reveal.accept(state());
  reveal.apply(payload({ file: 'other.py' }), nodes);
  expect(reveal.apply(payload(), nodes)).toBe(nodes);
});
