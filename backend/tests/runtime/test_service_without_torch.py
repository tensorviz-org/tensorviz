"""Exercise the real static service with framework imports prohibited.

Also runnable using ONLY requirements-standalone.txt (no pytest or Torch):
    python tests/runtime/test_service_without_torch.py
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.request


BACKEND = Path(__file__).resolve().parents[2]
TOKEN = "standalone-no-torch-test-not-a-production-secret"


def test_static_service_without_model_framework(failure_type="missing"):
    with tempfile.TemporaryDirectory(prefix="tensorviz-no-torch-") as temporary:
        root = Path(temporary)
        imported = root / "framework-imported"
        model_executed = root / "model-executed"
        runner = r'''
import importlib.abc, pathlib, sys
backend, attempts, failure_type = sys.argv[1:]
class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorflow', 'jax'}:
            pathlib.Path(attempts).write_text(fullname)
            if failure_type == 'broken':
                raise RuntimeError('Test fixture: incompatible framework native dispatcher')
            raise ModuleNotFoundError('Framework imports are forbidden in the static service', name=fullname)
sys.meta_path.insert(0, NoFramework())
sys.path.insert(0, backend)
if (pathlib.Path(backend) / 'native-backend.json').exists():
    import _tensorviz_backend
    sys.exit(_tensorviz_backend.run('standalone_server.py', ['--parent-pipe']))
import standalone_server
sys.argv = ['standalone_server.py', '--parent-pipe']
standalone_server.main()
'''
        process = subprocess.Popen([sys.executable, "-I", "-B", "-u", "-c", runner,
                                    os.environ.get("TENSORVIZ_SERVICE_TEST_BACKEND", str(BACKEND)), str(imported), failure_type], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env={**os.environ, "TENSORVIZ_LOCAL_TOKEN": TOKEN})
        try:
            # Bounded child read avoids a hung test on a broken startup.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                line = executor.submit(process.stdout.readline)
                try:
                    address = json.loads(line.result(timeout=30))
                except Exception:
                    process.kill()
                    raise
            origin = f"http://127.0.0.1:{address['port']}"

            def request(route, payload=None, authorized=True):
                headers = {"X-tensorViz-Token": TOKEN} if authorized else {}
                data = None
                if payload is not None:
                    data = json.dumps(payload).encode()
                    headers["Content-Type"] = "application/json"
                # Do not route local service requests through a user's HTTP proxy.
                client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                for attempt in range(100):
                    try:
                        with client.open(urllib.request.Request(origin + route, data=data, headers=headers), timeout=5) as reply:
                            return reply.status, json.load(reply)
                    except urllib.error.HTTPError as error:
                        return error.code, json.load(error)
                    except (ConnectionError, urllib.error.URLError):
                        if attempt == 99:
                            raise
                        import time
                        time.sleep(0.02)

            assert request("/api/health")[1]["host"] == "standalone"
            assert request("/api/health", authorized=False)[0] == 401
            source = (f"from pathlib import Path\nPath({str(model_executed)!r}).touch()\n"
                      "from torch import nn\nclass Tiny(nn.Module):\n"
                      " def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n"
                      " def forward(self, x):\n  return self.fc(x)\n")
            digest = hashlib.sha256(source.encode()).hexdigest()
            sources = [{"file": "model.py", "source": source, "digest": digest}]
            root_node = {"file": "model.py", "cls": "Tiny", "source_digest": digest}
            body = {"protocol_version": 1, "root": root_node, "sources": sources, "truncated": False}
            status, projection = request("/api/ide/multifile/project", body)
            assert status == 200, projection
            assert projection["traced"] is False
            assert projection["document"]["semantic"]["nodes"]
            status, discovery = request("/api/ide/review/index", {"protocol_version": 1, "sources": sources, "truncated": False})
            assert status == 200, discovery
            assert discovery["models"][0]["cls"] == "Tiny"
            status, setup = request("/api/ide/graph-setup/discover", body)
            assert status == 200, setup
            assert not model_executed.exists(), "Static discovery executed the user's model"
            # Optional signature introspection may attempt an import, but the
            # blocker makes actual framework loading impossible in this process.
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


def test_static_service_with_broken_framework():
    test_static_service_without_model_framework("broken")


def test_service_requirements_and_capture_diagnostics_are_separate():
    runner = r'''
import sys, json
sys.path.insert(0, sys.argv[1])
from services.ide import environment_setup as setup
service = setup.inspect_environment(setup.SERVICE_REQUIREMENTS)
capture = setup.inspect_environment(setup.CAPTURE_REQUIREMENTS)
print(json.dumps({'service': service, 'capture': capture}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", runner, str(BACKEND)],
                            capture_output=True, text=True, check=True, timeout=30)
    status = json.loads(result.stdout)
    assert not any(item.split("=")[0].lower() == "torch" for item in status["service"]["missing"])
    expected_abi = sysconfig.get_config_var('SOABI')
    if expected_abi is None and sys.platform == 'win32':
        expected_abi = sysconfig.get_config_var('EXT_SUFFIX')[1:-4]
    assert status['service']['environment']['abi'] == expected_abi
    # Capture still declares Torch even though static operations no longer do.
    assert "torch" in (BACKEND / "requirements-capture.txt").read_text().splitlines()


if __name__ == "__main__":
    test_static_service_without_model_framework()
    test_static_service_with_broken_framework()
    test_service_requirements_and_capture_diagnostics_are_separate()
    print("Static service health, discovery, projection and setup passed without framework imports.")
