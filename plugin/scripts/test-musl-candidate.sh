#!/bin/sh
# Run only inside the Alpine CI container, with a disposable checkout.
set -eu
# Checkout is owned by the host runner, not container root. Trust only this path.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
apk add --no-cache nodejs npm python3 curl ca-certificates build-base patchelf binutils tar git
asset='cpython-3.12.14%2B20260901-x86_64-unknown-linux-musl-install_only_stripped.tar.gz'
curl --fail --location --proto '=https' --proto-redir '=https' --max-filesize 134217728 \
  "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/$asset" -o /tmp/python.tar.gz
printf '%s  %s\n' '1f37044c8cdbd74d5ee112a753c65ef209fedd169c98f3e4e748a93e27eb27a4' /tmp/python.tar.gz | sha256sum -c -
tar -xzf /tmp/python.tar.gz -C /opt
/opt/python/bin/python3 -m venv /opt/compiler
/opt/python/bin/python3 -m venv /opt/service
/opt/compiler/bin/python -m pip install -r plugin/compiler-requirements.txt
/opt/service/bin/python -m pip install --only-binary=:all: -r backend/requirements-standalone.txt
/opt/service/bin/python -I -c 'import sysconfig,importlib.util; assert sysconfig.get_config_var("SOABI")=="cpython-312-x86_64-linux-musl"; assert importlib.util.find_spec("torch") is None'
npm ci
npm ci --prefix plugin
npm ci --prefix canvas
export TENSORVIZ_COMPILER_PYTHON=/opt/compiler/bin/python
export TENSORVIZ_NO_TORCH_PYTHON=/opt/compiler/bin/python
export TENSORVIZ_SERVICE_TEST_PYTHON=/opt/service/bin/python
export TENSORVIZ_SERVICE_TEST_BACKEND="$PWD/plugin/out/native-runtime/backend"
export TENSORVIZ_TEST_BACKEND="$TENSORVIZ_SERVICE_TEST_BACKEND"
node --test plugin/test/unit/musl-platform.test.cjs
node plugin/scripts/build-release.mjs --native-archive
node --test plugin/test/native/host.cjs plugin/test/native/no-torch.cjs plugin/test/native/managed-service.cjs
/opt/service/bin/python backend/tests/runtime/test_service_without_torch.py
