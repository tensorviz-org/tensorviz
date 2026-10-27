"""Decide whether this process may listen outside loopback.

The standalone backend executes model code. Binding ``0.0.0.0`` with auth
disabled is unauthenticated remote code execution. Loopback is the default.
A non-loopback host is allowed only when ``TENSORVIZ_LOCAL_TOKEN`` is set.
``AUTH_DISABLED`` is not authentication and never satisfies this check.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys


def is_loopback_bind(host: str) -> bool:
    """True for names and addresses that stay on this machine."""
    name = (host or "").strip().lower()
    if name in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def public_bind_error(host: str, *, token: str) -> str | None:
    """A startup error when ``host`` is reachable beyond this machine without a token."""
    if is_loopback_bind(host) or token.strip():
        return None
    return (
        f"Refusing to listen on {host!r}. This backend runs model code and has no "
        "request authentication unless TENSORVIZ_LOCAL_TOKEN is set. "
        "Use 127.0.0.1 (the default), or set TENSORVIZ_LOCAL_TOKEN before binding "
        "a non-loopback host. AUTH_DISABLED does not make a public bind safe."
    )


def ensure_safe_bind(host: str | None = None) -> str:
    """Return the bind host, or exit with a clear message."""
    chosen = (host if host is not None else os.environ.get("HOST", "127.0.0.1")).strip() or "127.0.0.1"
    assert_bind_allowed([chosen])
    return chosen


def assert_bind_allowed(hosts: list[str]) -> None:
    """Exit when any address is reachable beyond this machine without a launch token."""
    token = os.environ.get("TENSORVIZ_LOCAL_TOKEN", "")
    for host in hosts:
        error = public_bind_error(host, token=token)
        if error:
            print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(1)


def _socket_hosts(sockets) -> list[str]:
    hosts: list[str] = []
    for sock in sockets:
        try:
            address = sock.getsockname()
        except OSError:
            hosts.append("0.0.0.0")
            continue
        if isinstance(address, tuple) and address:
            hosts.append(str(address[0]))
    return hosts


def _fd_hosts(fd: int) -> list[str]:
    """Read an inherited socket without closing the caller's descriptor."""
    try:
        borrowed = socket.socket(fileno=fd)
    except OSError:
        return ["0.0.0.0"]
    try:
        return _socket_hosts([borrowed])
    finally:
        borrowed.detach()


def candidate_bind_hosts(
    host: str | None,
    sockets=None,
    *,
    uds: str | None = None,
    fd: int | None = None,
) -> list[str]:
    """Addresses uvicorn is about to accept on.

    An explicit socket list (including an empty one) is the listen set. A unix
    socket stays on this machine. Otherwise the configured host is what
    ``Server.startup`` binds, and uvicorn's own default is loopback.
    """
    if sockets is not None:
        return _socket_hosts(sockets)
    if uds:
        return []
    if fd is not None:
        return _fd_hosts(fd)
    return [((host or "").strip() or "127.0.0.1")]


_GUARD_ATTR = "_tensorviz_bind_guarded"


def install_uvicorn_bind_guard() -> None:
    """Refuse a public listen when uvicorn starts this app.

    ``python -m uvicorn server_v2:app --host 0.0.0.0`` never executes
    ``server_v2.py``'s ``__main__`` block. ``Server.startup`` binds that host
    after the app module has loaded, including when ``--lifespan off`` skips
    FastAPI startup hooks. ``Config.bind_socket`` is wrapped for a process that
    has already imported this module. ``start.sh`` still calls
    ``ensure_safe_bind`` before it launches a reload supervisor, because that
    supervisor binds in the parent before this module is imported.
    """
    try:
        import uvicorn.config as uvicorn_config
        import uvicorn.server as uvicorn_server
    except ImportError:
        return

    original_bind = uvicorn_config.Config.bind_socket
    if not getattr(original_bind, _GUARD_ATTR, False):

        def guarded_bind(self):
            assert_bind_allowed(candidate_bind_hosts(self.host, uds=self.uds, fd=self.fd))
            return original_bind(self)

        setattr(guarded_bind, _GUARD_ATTR, True)
        uvicorn_config.Config.bind_socket = guarded_bind

    original_startup = uvicorn_server.Server.startup
    if not getattr(original_startup, _GUARD_ATTR, False):

        async def guarded_startup(self, *args, **kwargs):
            sockets = kwargs.get("sockets")
            if sockets is None and args:
                sockets = args[0]
            config = getattr(self, "config", None)
            assert_bind_allowed(candidate_bind_hosts(
                getattr(config, "host", None),
                sockets,
                uds=getattr(config, "uds", None),
                fd=getattr(config, "fd", None) if sockets is None else None,
            ))
            return await original_startup(self, *args, **kwargs)

        setattr(guarded_startup, _GUARD_ATTR, True)
        uvicorn_server.Server.startup = guarded_startup
