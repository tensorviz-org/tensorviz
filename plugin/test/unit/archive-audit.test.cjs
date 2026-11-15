const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const python = process.env.TENSORVIZ_COMPILER_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const fixture = `import io,json,stat,sys,tarfile,zipfile
spec=json.load(sys.stdin)
if spec['kind']=='runtime':
 with tarfile.open(sys.argv[1],'w:gz',format=tarfile.USTAR_FORMAT) as archive:
  for entry in spec['entries']:
   data=entry.get('data','').encode(); info=tarfile.TarInfo(entry['name']); info.size=len(data)
   if entry.get('type')=='symlink': info.type=tarfile.SYMTYPE; info.linkname='target'; info.size=0
   if entry.get('type')=='hardlink': info.type=tarfile.LNKTYPE; info.linkname='target'; info.size=0
   if entry.get('type')=='directory': info.type=tarfile.DIRTYPE; info.size=0
   if entry.get('type')=='fifo': info.type=tarfile.FIFOTYPE; info.size=0
   archive.addfile(info,io.BytesIO(data))
else:
 with zipfile.ZipFile(sys.argv[1],'w',compression=zipfile.ZIP_DEFLATED) as archive:
  for entry in spec['entries']:
   info=zipfile.ZipInfo(entry['name']); info.create_system=3
   info.external_attr=((stat.S_IFLNK if entry.get('type')=='symlink' else stat.S_IFREG)|0o644)<<16
   archive.writestr(info,entry.get('data',''))
`;
async function archive(t, kind, entries) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-archive-audit-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = path.join(root, kind === 'runtime' ? 'runtime.tar.gz' : 'catalog.zip');
  execFileSync(python, ['-c', fixture, file], { input: JSON.stringify({ kind, entries }), stdio: ['pipe', 'pipe', 'pipe'] });
  return file;
}
const manifest = { name: 'release-manifest.json', data: JSON.stringify({ files: {}, version: '1.2.3' }) };

for (const kind of ['runtime', 'zip']) test(`hashes every ${kind} member without extracting it`, async t => {
  const { auditArchive } = await import('../../scripts/archive-audit.mjs');
  const entries = Array.from({ length: 64 }, (_, index) => ({ name: `assets/file-${index}.txt`, data: `content ${index}` }));
  if (kind === 'runtime') entries.push(manifest);
  const result = auditArchive(await archive(t, kind, entries), kind);
  assert.equal(Object.keys(result.files).length, entries.length);
  for (const entry of entries) assert.equal(result.files[entry.name], createHash('sha256').update(entry.data).digest('hex'));
  if (kind === 'runtime') assert.deepEqual(result.manifest, JSON.parse(manifest.data));
});

for (const kind of ['runtime', 'zip']) for (const name of ['../escape', '/absolute', 'a/../escape', 'a//alias', 'a/./alias']) {
  test(`${kind} rejects unsafe/aliased path ${name}`, async t => {
    const { auditArchive } = await import('../../scripts/archive-audit.mjs');
    const file = await archive(t, kind, [{ name, data: 'bad' }, manifest]);
    assert.throws(() => auditArchive(file, kind), /Unsafe archive path/);
  });
}
for (const kind of ['runtime', 'zip']) test(`${kind} rejects duplicate members`, async t => {
  const { auditArchive } = await import('../../scripts/archive-audit.mjs');
  const file = await archive(t, kind, [{ name: 'duplicate', data: 'a' }, { name: 'duplicate', data: 'b' }, manifest]);
  assert.throws(() => auditArchive(file, kind), /Duplicate/);
});
for (const type of ['symlink', 'hardlink', 'directory', 'fifo']) test(`runtime rejects ${type} entries`, async t => {
  const { auditArchive } = await import('../../scripts/archive-audit.mjs');
  const file = await archive(t, 'runtime', [{ name: 'unsafe', type }, manifest]);
  assert.throws(() => auditArchive(file, 'runtime'), /link or special file/);
});
test('ZIP rejects symbolic links', async t => {
  const { auditArchive } = await import('../../scripts/archive-audit.mjs');
  const file = await archive(t, 'zip', [{ name: 'unsafe', type: 'symlink', data: 'target' }]);
  assert.throws(() => auditArchive(file, 'zip'), /link or special file/);
});
test('runtime rejects source files and missing manifests', async t => {
  const { auditArchive } = await import('../../scripts/archive-audit.mjs');
  const source = await archive(t, 'runtime', [{ name: 'private.py', data: 'secret source' }, manifest]);
  assert.throws(() => auditArchive(source, 'runtime'), /Source file/);
  const missing = await archive(t, 'runtime', [{ name: 'host', data: 'binary' }]);
  assert.throws(() => auditArchive(missing, 'runtime'), /no release manifest/);
});
for (const corruption of ['truncate', 'crc']) test(`runtime rejects ${corruption} gzip corruption`, async t => {
  const { auditArchive } = await import('../../scripts/archive-audit.mjs');
  const file = await archive(t, 'runtime', [{ name: 'host', data: 'binary' }, manifest]);
  const bytes = await fs.readFile(file);
  if (corruption === 'crc') bytes[bytes.length - 8] ^= 0xff;
  await fs.writeFile(file, corruption === 'truncate' ? bytes.subarray(0, bytes.length - 4) : bytes);
  assert.throws(() => auditArchive(file, 'runtime'), /Archive audit failed/);
});
