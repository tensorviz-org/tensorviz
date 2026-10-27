"""Parent-pipe ownership is opt-in and ends the owned group on launcher death."""
import os
from pathlib import Path
import subprocess
import sys


def test_parent_pipe_is_consumed_only_when_enabled():
    root = Path(__file__).parent.resolve().parents[1]
    program = ('from services.ingest.worker_lifetime import guard_parent_pipe\n'
               'import os,sys,time\n'
               'guard_parent_pipe()\n'
               'print(os.environ.get("TENSORVIZ_PARENT_PIPE"), flush=True)\n'
               'time.sleep(60)\n')
    with subprocess.Popen([sys.executable, '-c', program], cwd=root,
                          env={**os.environ, 'TENSORVIZ_PARENT_PIPE': '1'},
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          start_new_session=os.name != 'nt') as child:
        assert child.stdout.readline().strip() == b'None', 'the flag must not leak to model children'
        assert child.poll() is None
        child.stdin.close()
        assert child.wait(timeout=10) != 0

    disabled = subprocess.run([sys.executable, '-c', program.replace('time.sleep(60)', 'print("finished")')],
                              cwd=root, env={**os.environ, 'TENSORVIZ_PARENT_PIPE': '0'},
                              input=b'', capture_output=True, timeout=10)
    assert disabled.returncode == 0
    assert b'finished' in disabled.stdout
