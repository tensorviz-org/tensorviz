"""End an opt-in standalone worker tree when its owning launcher pipe closes.

The host creates a separate process group and keeps stdin open. This is lifecycle
ownership, not containment of deliberately hostile model/package code.
"""
import os
import signal
import subprocess
import threading


def guard_parent_pipe() -> None:
    if os.environ.pop('TENSORVIZ_PARENT_PIPE', None) != '1':
        return

    def watch():
        try:
            while os.read(0, 1024):
                pass
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
