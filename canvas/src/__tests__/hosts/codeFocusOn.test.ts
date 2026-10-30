import { configureCodeFocusOn, isCodeFocusOn, onCodeFocusOnChange, setCodeFocusOn } from '../../ide/codeFocusOn';

beforeEach(() => { localStorage.clear(); configureCodeFocusOn('host'); });

it('starts off so exploring the graph never opens code until the user turns Code focus on', () => {
  expect(isCodeFocusOn()).toBe(false);
});

it('keeps each surface\'s own choice, like its appearance', () => {
  setCodeFocusOn(true);
  configureCodeFocusOn('browser');
  expect(isCodeFocusOn()).toBe(false);
  configureCodeFocusOn('host');
  expect(isCodeFocusOn()).toBe(true);
  expect(localStorage.getItem('tensorviz.canvas.codeFocus.host')).toBe('on');
});

it('ignores an unknown saved value and still switches when storage is unavailable', () => {
  localStorage.setItem('tensorviz.canvas.codeFocus.host', 'edit');
  configureCodeFocusOn('host');
  expect(isCodeFocusOn()).toBe(false);
  const setItem = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  try {
    setCodeFocusOn(true);
    expect(isCodeFocusOn()).toBe(true);
  } finally { setItem.mockRestore(); }
});

it('notifies listeners once per real change', () => {
  const seen: boolean[] = [];
  const stop = onCodeFocusOnChange(on => seen.push(on));
  setCodeFocusOn(true);
  setCodeFocusOn(true);
  setCodeFocusOn(false);
  stop();
  setCodeFocusOn(true);
  expect(seen).toEqual([true, false]);
});
