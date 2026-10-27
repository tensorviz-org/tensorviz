import portable from '../../../../tests/fixtures/contracts/portable-capture.json';
import type { components } from '../../types/semanticContracts.generated';
import type { CaptureInspectionQuery } from '../../ide/protocol.generated';
import { isCaptureInspectionPage } from '../../ide/protocol.generated';
import { createOfflineCaptureClient } from '../../viewer/offlineCaptureClient';

const capture = portable.capture as components['schemas']['CaptureSnapshot'];
const query = (fields: Partial<CaptureInspectionQuery> = {}): CaptureInspectionQuery => ({
  captureId: capture.id, sourceId: capture.source.id, path: [], offset: 0, limit: 2, ...fields,
});

it('reconstructs the exact Python portable fixture using bounded offline pages', async () => {
  const client = createOfflineCaptureClient(capture);
  const original = JSON.stringify(capture);
  async function read(path: Array<string | number>): Promise<unknown> {
    let page = (await client.inspect(query({ path }))).page!;
    expect(isCaptureInspectionPage(page)).toBe(true);
    if (!['object', 'array'].includes(page.node.kind)) return page.node.value;
    const result: Record<string | number, unknown> | unknown[] = page.node.kind === 'array' ? [] : {};
    for (;;) {
      for (const item of page.items) Object.defineProperty(result, item.key!, { value: await read([...path, item.key!]), enumerable: true });
      if (page.nextOffset === null) return result;
      page = (await client.inspect(query({ path, offset: page.nextOffset }))).page!;
    }
  }
  expect(await read([])).toEqual(capture);
  expect(JSON.stringify(capture)).toBe(original);
  const args = (await client.inspect(query({ entity: { kind: 'operation', id: 'add2' }, path: ['arguments', 'args'] }))).page!;
  expect(args.items.map(item => item.key)).toEqual([0, 1]);
  expect(args.items[0].label).toEqual(args.items[1].label);
});

it('refuses foreign capture identity and prototype/invalid traversal; closed files cannot be read', async () => {
  const client = createOfflineCaptureClient(capture);
  for (const fields of [{ captureId: 'other' }, { sourceId: 'other' }, { path: ['__proto__'] }, { path: ['operations', '0'] }, { limit: 0 }]) {
    expect((await client.inspect(query(fields))).page).toBeNull();
  }
  client.dispose();
  expect((await client.inspect(query())).detail).toMatch(/closed/);
});

it('counts UTF-8 bytes instead of characters while paging large Unicode values', async () => {
  const large = JSON.parse(JSON.stringify(capture));
  large.modules[0].paths = Array.from({ length: 100 }, (_, i) => `${i}` + '界'.repeat(4000));
  const client = createOfflineCaptureClient(large);
  const received: unknown[] = [];
  let offset = 0;
  for (;;) {
    const { page } = await client.inspect(query({ path: ['modules', 0, 'paths'], offset, limit: 200 }));
    expect(page).not.toBeNull();
    expect(new Blob([JSON.stringify(page)]).size).toBeLessThanOrEqual(64 * 1024);
    received.push(...page!.items.map(item => item.value));
    if (page!.nextOffset === null) break;
    expect(page!.nextOffset).toBeGreaterThan(offset);
    offset = page!.nextOffset;
  }
  expect(received).toEqual(large.modules[0].paths);
});
