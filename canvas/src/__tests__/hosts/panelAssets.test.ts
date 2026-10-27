/** @jest-environment node */
import { panelAssetFetch } from '../../plugin/panelAssets';
import { runInThisContext } from 'vm';

// Jest 27 does not copy Node's native Fetch globals into its test environment.
globalThis.Response = runInThisContext('Response');

const translation = 'data:application/json;base64,' + Buffer.from('{"label":"Français"}').toString('base64');
const icon = 'data:image/svg+xml;base64,PHN2Zy8+';

test('bundled translations decode as UTF-8 without making a network request', async () => {
  const fallback = jest.fn().mockRejectedValue(new Error('connect-src is none'));
  const localFetch = panelAssetFetch({ 'translations/fr.json': translation }, fallback);
  const response = await localFetch(translation, { referrerPolicy: 'strict-origin-when-cross-origin' });
  expect(response.ok).toBe(true);
  expect(response.headers.get('Content-Type')).toBe('application/json');
  expect(await response.json()).toEqual({ label: 'Français' });
  expect(fallback).not.toHaveBeenCalled();
});

test('only known bundled GET translations bypass the original fetch implementation', async () => {
  const fallback = jest.fn().mockResolvedValue(new Response('original'));
  const localFetch = panelAssetFetch({ 'translations/fr.json': translation, 'icons/test.svg': icon }, fallback);
  for (const input of ['https://example.invalid/translations/en.json', 'data:application/json;base64,e30=', icon]) {
    await localFetch(input);
    expect(fallback).toHaveBeenLastCalledWith(input, undefined);
  }
  await localFetch(translation, { method: 'POST' });
  expect(fallback).toHaveBeenLastCalledWith(translation, { method: 'POST' });
  const controller = new AbortController();
  controller.abort();
  await expect(localFetch(translation, { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
  expect(fallback).toHaveBeenCalledTimes(4);
});
