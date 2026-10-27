import { test, expect, Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { selectedNodeEdit } from './local-browser-node-controls';

const pageFile = process.env.TENSORVIZ_QA_PAGE_FILE;
test.skip(!pageFile, 'Requires the installed complex-model host with qa-local-browser.cjs');
const control = pageFile ? path.dirname(pageFile) : '';
const root = path.dirname(control);
const file = 'complex_models/network.py';

async function command(type: string, output: string, extra = {}) {
  const id = crypto.randomUUID();
  fs.writeFileSync(path.join(control, 'command.json'), JSON.stringify({ type, id, ...extra }));
  const read = () => {
    try { return JSON.parse(fs.readFileSync(path.join(control, output), 'utf8')); } catch { return null; }
  };
  await expect.poll(() => read()?.id, { timeout: 15_000 }).toBe(id);
  return read();
}
const inspect = () => command('inspect', 'source-state.json');
async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(JSON.parse(fs.readFileSync(pageFile!, 'utf8')).pageUrl);
  await page.getByRole('button', { name: 'Expand toolbar' }).click();
  await expect(page.getByTitle('Model summary', { exact: true })).toContainText('102');
}
async function select(page: Page, id: string) {
  await page.getByRole('button', { name: 'More tools', exact: true }).press('ControlOrMeta+p');
  await page.getByRole('textbox', { name: 'Quick jump to node' }).fill(id);
  await page.getByRole('button', { name: new RegExp(`^${id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} —`) }).click();
}
async function editForm(page: Page, id = 'branch_fc') {
  const close = page.getByRole('button', { name: 'Close source editor', exact: true });
  if (await close.isVisible()) await close.click();
  await select(page, id);
  await (await selectedNodeEdit(page)).click();
  await expect(page.getByRole('region', { name: 'Source edit proposal' })).toBeVisible();
}
async function waitReview() {
  let state: any;
  await expect.poll(async () => { state = await inspect(); return state.reviews.length; }, { timeout: 20_000 }).toBe(1);
  return state;
}
async function apply(page: Page) {
  await command('apply', 'decision-result.json');
  await expect(page.getByRole('status').filter({ hasText: 'Applied to the VS Code buffer' })).toBeVisible();
}
async function undo(expected: string, targetFile = file) {
  const state = await command('undo', 'undo-result.json', { file: targetFile });
  expect(state.text).toBe(expected);
}

async function restore(baseline: string, targetFile = file) {
  const state = await inspect();
  if (state.reviews.length) await command('discard', 'decision-result.json');
  if (state.files[targetFile]?.text !== baseline) await undo(baseline, targetFile);
}

test('parameter review writes only after native apply and one native Undo restores it', async ({ page }) => {
  const baseline = fs.readFileSync(path.join(root, 'fixture', file), 'utf8');
  await open(page);
  await editForm(page);
  await page.getByRole('combobox', { name: 'Parameter', exact: true }).selectOption('out_features');
  await page.getByRole('textbox', { name: 'Parameter value' }).fill('12');
  await page.getByRole('button', { name: 'Review in VS Code' }).click();
  const review = await waitReview();
  expect(review.files[file].text).toBe(baseline);
  expect(review.files[file].dirty).toBe(false);
  expect(review.reviews[0].after).toContain('nn.Linear(10, 12)');
  try {
    await apply(page);
    const state = await inspect();
    expect(state.files[file].text).toBe(baseline.replace('nn.Linear(10, 10)', 'nn.Linear(10, 12)'));
    expect(state.files[file].dirty).toBe(true);
    expect(fs.readFileSync(path.join(root, 'fixture', file), 'utf8')).toBe(baseline);
    await page.screenshot({ path: path.join(root, 'evidence/source-parameter-applied.png') });
  } finally { await restore(baseline); }
});

test('add and delete each use native review, reproject, and undo', async ({ page }) => {
  const baseline = fs.readFileSync(path.join(root, 'fixture', file), 'utf8');
  for (const operation of ['add_layer', 'remove_layer']) {
    await open(page);
    await editForm(page);
    await page.getByRole('combobox', { name: 'Source operation' }).selectOption(operation);
    if (operation === 'add_layer') await page.getByRole('textbox', { name: 'New module name' }).fill('extra');
    await page.getByRole('button', { name: 'Review in VS Code' }).click();
    await waitReview();
    try {
      await apply(page);
      const state = await inspect();
      if (operation === 'add_layer') {
        expect(state.files[file].text).toContain('self.extra = nn.ReLU(');
        await expect(page.getByTitle('Model summary', { exact: true })).toContainText('103');
      } else {
        expect(state.files[file].text).not.toContain('self.branch_fc');
        await expect(page.getByTitle('Model summary', { exact: true })).toContainText('101');
      }
      expect(fs.readFileSync(path.join(root, 'fixture', file), 'utf8')).toBe(baseline);
    } finally { await restore(baseline); }
  }
});

test('selected trunks become an expandable source block after native review', async ({ page }) => {
  const baseline = fs.readFileSync(path.join(root, 'fixture', file), 'utf8');
  await open(page);
  await page.getByRole('button', { name: 'Fit graph to view', exact: true }).click();
  // tldraw hit-tests its background; the rendered shape container is pointer-inert.
  await page.locator('[data-shape-id="shape:node:trunk1"] .layer-id').click({ force: true });
  await expect(page.locator('[data-shape-id="shape:node:trunk1"] .module-node')).toHaveClass(/\bselected\b/);
  await page.getByRole('button', { name: 'Fit graph to view', exact: true }).click();
  await page.locator('[data-shape-id="shape:node:trunk2"] .layer-id').click({ force: true, modifiers: ['Shift'] });
  await (await selectedNodeEdit(page)).click();
  await expect(page.getByRole('combobox', { name: 'Source operation' })).toHaveValue('group_layers');
  await page.getByRole('textbox', { name: 'New module name' }).fill('backbone');
  await page.getByRole('button', { name: 'Review in VS Code' }).click();
  const review = await waitReview();
  expect(review.reviews[0].after).toContain('class BackboneBlock(nn.Module)');
  try {
    await apply(page);
    await page.getByRole('button', { name: 'Close source editor', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Expand backbone', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Expand backbone', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Expand backbone.trunk1', exact: true })).toBeVisible();
    await page.screenshot({ path: path.join(root, 'evidence/source-group-applied.png') });
  } finally { await restore(baseline); }
});

test('a source change during review refuses apply and preserves the user edit', async ({ page }) => {
  const baseline = fs.readFileSync(path.join(root, 'fixture', file), 'utf8');
  await open(page);
  await editForm(page);
  await page.getByRole('combobox', { name: 'Parameter', exact: true }).selectOption('out_features');
  await page.getByRole('textbox', { name: 'Parameter value' }).fill('12');
  await page.getByRole('button', { name: 'Review in VS Code' }).click();
  await waitReview();
  await command('edit', 'edit-result.json', { file, before: 'nn.Linear(10, 10)', after: 'nn.Linear(10, 14)' });
  try {
    await command('apply', 'decision-result.json');
    await expect(page.getByRole('status').filter({ hasText: 'source changed during review' })).toBeVisible();
    expect((await inspect()).files[file].text).toBe(baseline.replace('nn.Linear(10, 10)', 'nn.Linear(10, 14)'));
  } finally { await undo(baseline); }
});

test('an imported child edits its owning file and one Undo restores it', async ({ page }) => {
  const targetFile = 'complex_models/blocks.py';
  const baseline = fs.readFileSync(path.join(root, 'fixture', targetFile), 'utf8');
  await open(page);
  await editForm(page, 'trunk1.stage1.block1.first.conv');
  await page.getByRole('combobox', { name: 'Parameter', exact: true }).selectOption('kernel_size');
  await page.getByRole('textbox', { name: 'Parameter value' }).fill('5');
  await page.getByRole('button', { name: 'Review in VS Code' }).click();
  const review = await waitReview();
  expect(review.reviews[0].before).toBe(baseline);
  expect(review.reviews[0].title).toContain('other projected instances');
  expect(review.reviews[0].after).toContain('nn.Conv2d(width, width, 5, padding=1)');
  try {
    await apply(page);
    expect((await inspect()).files[targetFile].text).toBe(baseline.replace('nn.Conv2d(width, width, 3, padding=1)', 'nn.Conv2d(width, width, 5, padding=1)'));
    expect(fs.readFileSync(path.join(root, 'fixture', targetFile), 'utf8')).toBe(baseline);
  } finally { await restore(baseline, targetFile); }
});

for (const [width, height] of [[698, 994], [1024, 768], [1440, 900]]) {
  test(`source controls stay readable and within ${width}x${height}`, async ({ page }) => {
    await open(page);
    await page.setViewportSize({ width, height });
    await editForm(page);
    await page.getByRole('combobox', { name: 'Source operation' }).selectOption('add_layer');
    await page.getByRole('combobox', { name: 'New layer type' }).selectOption('conv2d');
    const rail = page.getByRole('region', { name: 'Source edit proposal' });
    const bounds = await rail.boundingBox();
    expect(bounds!.width).toBeGreaterThanOrEqual(280);
    const submit = page.getByRole('button', { name: 'Review in VS Code' });
    await submit.scrollIntoViewIfNeeded();
    await expect(submit).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight + 1)).toBe(true);
    await page.getByRole('button', { name: 'Close source editor', exact: true }).click();
    await expect(page.getByRole('button', { name: 'More tools', exact: true })).toBeInViewport();
    await page.screenshot({ path: path.join(root, `evidence/source-form-${width}.png`) });
  });
}


test('symbolic parameters are refused without opening a diff or changing source', async ({ page }) => {
  await open(page);
  await editForm(page, 'branch.fc');
  const before = (await inspect()).files['complex_models/heads.py'].text;
  await page.getByRole('combobox', { name: 'Parameter', exact: true }).selectOption('out_features');
  await page.getByRole('textbox', { name: 'Parameter value' }).fill('13');
  await page.getByRole('button', { name: 'Review in VS Code' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'overwriting it would delete the expression' })).toBeVisible();
  const state = await inspect();
  expect(state.reviews).toHaveLength(0);
  expect(state.files['complex_models/heads.py'].text).toBe(before);
});
