"""End an opt-in standalone worker tree when its owning launcher pipe closes.

The host creates a separate process group and keeps stdin open. This is lifecycle
ownership, not containment of deliberately hostile model/package code.
"""
import os
import signal
import subprocess
import threading


def _wait_for_parent_pipe() -> None:
    if os.name != 'nt':
        while os.read(0, 1024):
            pass
        return

    # Avoid a blocking CRT stdin read while native libraries initialize.
    # This watcher is the pipe's only reader; consume only available bytes.
    import ctypes
    from ctypes import wintypes
    import msvcrt
    import time

    handle = msvcrt.get_osfhandle(0)
    peek = ctypes.WinDLL('kernel32', use_last_error=True).PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                     wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    peek.restype = wintypes.BOOL
    while True:
        available = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error in (109, 233):  # ERROR_BROKEN_PIPE / ERROR_PIPE_NOT_CONNECTED
                return
            raise ctypes.WinError(error)
        if available.value:
            os.read(0, min(1024, available.value))
        else:
            time.sleep(0.05)


def guard_parent_pipe() -> None:
    if os.environ.pop('TENSORVIZ_PARENT_PIPE', None) != '1':
        return

    def watch():
        try:
            _wait_for_parent_pipe()
        finally:
            if os.name != 'nt' and os.getpgrp() == os.getpid():
                os.killpg(os.getpgrp(), signal.SIGKILL)
            elif os.name == 'nt':
                executable = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'taskkill.exe')
                try:
                    subprocess.run([executable, '/PID', str(os.getpid()), '/T', '/F'],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
