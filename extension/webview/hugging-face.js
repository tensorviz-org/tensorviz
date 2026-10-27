(() => {
  const vscode = acquireVsCodeApi();
  const $ = id => document.getElementById(id);
  let weights, architecture, maxDownload, busy = false, searchLimit = 200;
  const number = value => new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 2 }).format(value);
  const bytes = value => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GiB` : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(2)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(1)} KiB` : `${value} B`;
  function node(tag, text, className) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; }
  function status(text, error = false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; }
  function controls() {
    $('inspect').disabled = busy;
    $('run').disabled = busy || !architecture || weights?.fileBytes > maxDownload;
    $('url').disabled = busy;
  }
  function inspect() {
    $('results').hidden = true; weights = undefined;
    vscode.postMessage({ type: 'inspect', url: $('url').value });
  }
  $('inspect-form').addEventListener('submit', event => { event.preventDefault(); inspect(); });
  function addInput(input = { name: '', shape: [1, 16], dtype: 'float32', fill: 'zeros' }) {
    const row = node('div', undefined, 'input-row');
    const name = node('input'); name.value = input.name; name.placeholder = 'input_ids'; name.setAttribute('aria-label', 'Input argument');
    const shape = node('input'); shape.value = input.shape.join(', '); shape.placeholder = '1, 16'; shape.setAttribute('aria-label', 'Input shape');
    const dtype = node('select'); dtype.setAttribute('aria-label', 'Input dtype');
    for (const type of ['float32', 'int64', 'bool']) { const option = node('option', type); option.value = type; dtype.append(option); }
    dtype.value = input.dtype;
    const fill = node('select'); fill.setAttribute('aria-label', 'Input fill');
    for (const type of ['zeros', 'ones']) { const option = node('option', type); option.value = type; fill.append(option); }
    fill.value = input.fill;
    const remove = node('button', '×', 'secondary'); remove.type = 'button'; remove.setAttribute('aria-label', 'Remove input'); remove.onclick = () => row.remove();
    row.append(name, shape, dtype, fill, remove); $('inputs').append(row);
  }
  $('add-input').onclick = () => addInput();
  $('run').onclick = () => {
    const inputs = [...$('inputs').children].map(row => {
      const [name, shape, dtype, fill] = row.children;
      return { name: name.value.trim(), shape: shape.value.split(',').map(d => /^\d+$/.test(d.trim()) ? Number(d.trim()) : NaN), dtype: dtype.value, fill: fill.value };
    });
    vscode.postMessage({ type: 'run', inputs });
  };
  function tensorRow(tensor, label = tensor.name) {
    const row = node('div', undefined, 'tensor'); row.title = tensor.name;
    row.append(node('span', label), node('span', `[${tensor.shape.join(', ')}]`, 'shape'), node('span', tensor.dtype), node('span', bytes(tensor.bytes), 'muted'));
    return row;
  }
  function treeBranch(tensors, depth) {
    const fragment = document.createDocumentFragment(), groups = new Map();
    for (const tensor of tensors) {
      const parts = tensor.name.split('.');
      if (parts.length <= depth + 1) { fragment.append(tensorRow(tensor, parts[depth])); continue; }
      const part = parts[depth]; if (!groups.has(part)) groups.set(part, []); groups.get(part).push(tensor);
    }
    for (const [name, items] of groups) {
      const detail = node('details'), summary = node('summary', name);
      summary.append(node('span', `${items.length} tensors · ${bytes(items.reduce((n, t) => n + t.bytes, 0))}`, 'muted'));
      detail.append(summary);
      detail.addEventListener('toggle', () => { if (detail.open && detail.childNodes.length === 1) detail.append(treeBranch(items, depth + 1)); });
      fragment.append(detail);
    }
    return fragment;
  }
  function renderTensors() {
    if (!weights) return;
    $('tree').replaceChildren(); $('more').hidden = true;
    const query = $('search').value.trim().toLowerCase();
    if (!query) {
      $('tree').append(treeBranch(weights.tensors, 0));
      $('match-count').textContent = `${weights.tensors.length.toLocaleString()} tensors · expand a group to see shapes`;
    } else {
      const matches = weights.tensors.filter(t => `${t.name} ${t.shape.join(', ')} ${t.dtype}`.toLowerCase().includes(query));
      $('match-count').textContent = `${matches.length.toLocaleString()} matches${matches.length > searchLimit ? ` · showing ${searchLimit}` : ''}`;
      for (const tensor of matches.slice(0, searchLimit)) $('tree').append(tensorRow(tensor));
      $('more').hidden = matches.length <= searchLimit;
    }
  }
  $('search').oninput = () => { searchLimit = 200; renderTensors(); };
  $('more').onclick = () => { searchLimit += 200; renderTensors(); };
  function render(message) {
    weights = message.weights; architecture = message.architecture; maxDownload = message.maxDownload;
    $('results').hidden = false; $('model-name').textContent = weights.repo;
    $('revision').textContent = `${weights.file} · revision ${weights.commit.slice(0, 12)}`;
    $('stats').replaceChildren();
    for (const [label, value] of [['Stored elements', number(weights.tensors.reduce((n, t) => n + t.elements, 0))], ['Tensors', weights.tensors.length.toLocaleString()], ['Weight file', bytes(weights.fileBytes)], ['Header read', bytes(weights.headerBytes)]]) {
      const stat = node('div', undefined, 'stat'); stat.append(node('strong', value), node('span', label)); $('stats').append(stat);
    }
    const groups = new Map();
    for (const tensor of weights.tensors) { const key = tensor.name.split('.')[0]; groups.set(key, (groups.get(key) ?? 0) + tensor.bytes); }
    $('groups').replaceChildren();
    const sorted = [...groups].sort((a, b) => b[1] - a[1]);
    const shown = sorted.slice(0, 8);
    if (sorted.length > 8) shown.push(['Other groups', sorted.slice(8).reduce((n, t) => n + t[1], 0)]);
    for (const [name, size] of shown) {
      const group = node('div', undefined, 'group'), label = node('button', name);
      label.title = name; label.onclick = () => { $('search').value = name === 'Other groups' ? '' : name + '.'; renderTensors(); $('search').focus(); };
      const bar = node('progress'); bar.max = Math.max(1, weights.fileBytes - weights.headerBytes); bar.value = size; bar.setAttribute('aria-label', `${name} stored bytes`);
      group.append(label, bar, node('span', bytes(size), 'muted')); $('groups').append(group);
    }
    $('run-description').textContent = architecture ? `Load ${architecture} from your installed Transformers library and capture one CPU evaluation.`
      : weights.configNote || 'This file needs an explicit model implementation (missing architecture, custom code, quantization or a checkpoint shard). Open its local Python model with a checkpoint recipe.';
    $('input-form').hidden = !architecture;
    $('inputs').replaceChildren(); message.inputs.forEach(addInput);
    if (architecture && !message.inputs.length) addInput();
    $('run-note').textContent = weights.fileBytes > maxDownload ? 'This file exceeds the 2 GiB import limit. Use a local checkpoint recipe.'
      : `Downloads ${bytes(weights.fileBytes)} into a new workspace folder. Saves an editable model and tensorviz.toml, then opens Canvas. Requires the model’s Transformers class; custom model code is not downloaded.`;
    $('search').value = ''; renderTensors(); controls();
    status(`Read ${bytes(weights.headerBytes)} of metadata. No model execution or full weight download yet.`);
  }
  window.addEventListener('message', event => {
    const data = event.data;
    if (data?.type === 'initial') { $('url').value = data.url; if (data.url) inspect(); }
    if (data?.type === 'busy') { busy = data.busy; controls(); }
    if (data?.type === 'status') status(data.text, data.error);
    if (data?.type === 'weights') render(data);
  });
  vscode.postMessage({ type: 'ready' });
})();
