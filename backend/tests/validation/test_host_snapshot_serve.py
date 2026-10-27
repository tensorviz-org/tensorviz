"""``host_snapshot.py --serve``: one answer per line, equal to the one-shot answer,
with nothing carried between requests that could make an answer stale."""
import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / 'services' / 'validation' / 'host_snapshot.py'
MODEL = 'import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(8, 8)\n    def forward(self, x):\n        return self.a(x)\n'


def request(root):
    return {'root': str(root), 'documents': {}, 'executable': sys.executable, 'environmentRevision': 0, 'session': 's',
            'workspaceId': 'w', 'runId': 'r', 'generation': 1, 'modelFile': 'model.py', 'className': 'Model',
            'timeoutMs': 30000, 'execution': {'inputs': {'args': [{'shape': [2, 8], 'dtype': 'float32', 'fill': 'randn'}]}}}


def one_shot(body):
    done = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT)], input=json.dumps(body).encode(),
                          capture_output=True, cwd=BACKEND, timeout=120)
    return json.loads(done.stdout)


def test_each_line_is_answered_like_a_one_shot_call_and_eof_exits_cleanly(tmp_path):
    (tmp_path / 'model.py').write_text(MODEL)
    body = json.dumps(request(tmp_path))
    lines = f'{body}\nnot json\n{body}\n'.encode()
    done = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), '--serve'], input=lines,
                          capture_output=True, cwd=BACKEND, timeout=120)
    assert done.returncode == 0
    answers = done.stdout.decode().splitlines()
    assert len(answers) == 3
    expected = one_shot(request(tmp_path))
    assert json.loads(answers[0]) == expected
    assert answers[1] == '{"error":"snapshot_failed"}'
    assert json.loads(answers[2]) == expected


def test_a_package_installed_between_requests_changes_the_environment_digest(tmp_path):
    """The served process must re-list distributions per request, as a fresh one does."""
    (tmp_path / 'model.py').write_text(MODEL)
    site = tmp_path / 'site'
    site.mkdir()
    driver = (f'import sys; sys.path.append({str(site)!r}); sys.argv = [{str(SCRIPT)!r}, "--serve"]; '
              f'sys.path.insert(0, {str(BACKEND)!r}); '
              'from services.validation import host_snapshot; sys.exit(host_snapshot.main())')
    process = subprocess.Popen([sys.executable, '-B', '-c', driver], cwd=BACKEND,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        def ask():
            process.stdin.write((json.dumps(request(tmp_path)) + '\n').encode()); process.stdin.flush()
            return json.loads(process.stdout.readline())
        before = ask()['plan']['interpreter']
        dist = site / 'tv_installed_later-1.0.dist-info'
        dist.mkdir()
        (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: tv-installed-later\nVersion: 1.0\n')
        after = ask()['plan']['interpreter']
    finally:
        process.stdin.close(); process.wait(timeout=60)
    assert before != after, 'a package installed mid-session must change the snapshot'


def test_an_oversized_line_is_refused_without_ending_the_session(tmp_path, monkeypatch):
    sys.path.insert(0, str(BACKEND))
    from services.validation import host_snapshot
    monkeypatch.setattr(host_snapshot, 'MAX_REQUEST_LINE', 16)
    import io
    stdin = io.BytesIO(b'{"root": "' + b'x' * 64 + b'"}\n')
    stdout = io.StringIO()
    monkeypatch.setattr(sys, 'stdin', type('S', (), {'buffer': stdin})())
    monkeypatch.setattr(sys, 'stdout', stdout)
    assert host_snapshot.serve() == 0
    assert stdout.getvalue().splitlines()[0] == '{"error":"snapshot_failed"}'
