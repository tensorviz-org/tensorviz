"""The per-file memo in ``_Provenance._source_frame`` must answer exactly what the
uncached walk did: realpath'd user files, keyed so a relative ``co_filename``
resolved under one cwd is never reused under another."""
import os

from services.ingest.trace import _Provenance


def _frame_from(code_file, roots, cwd=None):
    provenance = _Provenance(roots)
    namespace = {'provenance': provenance}
    previous = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        exec(compile('def probe():\n    return provenance._source_frame()\n', code_file, 'exec'), namespace)
        return provenance, namespace['probe']
    finally:
        os.chdir(previous)


def test_symlinked_workspace_frames_resolve_to_the_real_file(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'model.py').write_text('')
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    _, probe = _frame_from(str(link / 'model.py'), [str(link)])
    first, second = probe(), probe()
    assert first == second
    assert first[0] == os.path.realpath(str(real / 'model.py'))
    assert first[1:] == (2, 'probe')


def test_relative_filename_is_resolved_against_the_cwd_it_runs_under(tmp_path):
    inside, outside = tmp_path / 'workspace', tmp_path / 'elsewhere'
    inside.mkdir(); outside.mkdir()
    provenance, probe = _frame_from('model.py', [str(inside)])
    previous = os.getcwd()
    try:
        os.chdir(inside)
        assert probe()[0] == os.path.realpath(str(inside / 'model.py'))
        os.chdir(outside)  # same co_filename, now outside every root
        assert probe() is None
        os.chdir(inside)
        assert probe()[0] == os.path.realpath(str(inside / 'model.py'))
    finally:
        os.chdir(previous)


def test_frames_outside_every_root_are_not_attributed(tmp_path):
    (tmp_path / 'root').mkdir()
    _, probe = _frame_from(str(tmp_path / 'other.py'), [str(tmp_path / 'root')])
    assert probe() is None
    assert probe() is None
