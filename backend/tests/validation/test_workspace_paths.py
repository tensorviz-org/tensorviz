"""Execution paths must preserve environment-file exclusions after symlink resolution."""
from pathlib import Path
import tempfile
import unittest

from services.validation import setup

class WorkspacePaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'workspace'
        self.root.mkdir()
        (self.root / 'model.py').write_text('# synthetic model, never executed')

    def test_lexical_bad_paths(self):
        for path in ('../model.py', '/model.py', 'C:/model.py', r'dir\model.py',
                     './model.py', 'dir//model.py', '.env', '.env.private/setup.py',
                     'nested/.env/setup.py', 'nul\0.py', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                setup.workspace_file(str(self.root), path)

    def test_environment_file_aliases(self):
        for i, hidden in enumerate(('.env', '.env.local', '.environment')):
            (self.root / hidden).write_text('synthetic fixture only')
            alias = f'fixture{i}.pt'
            (self.root / alias).symlink_to(hidden)
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                setup.workspace_file(str(self.root), alias)

    def test_environment_directory_aliases(self):
        for i, hidden in enumerate(('.env', '.env.private', '.environment')):
            folder = self.root / hidden
            folder.mkdir()
            (folder / 'setup.py').write_text('# synthetic fixture only')
            alias = f'safe{i}'
            (self.root / alias).symlink_to(hidden, target_is_directory=True)
            with self.subTest(hidden=hidden, kind='directory'), self.assertRaises(ValueError):
                setup.workspace_directory(str(self.root), alias)
            with self.subTest(hidden=hidden, kind='file'), self.assertRaises(ValueError):
                setup.workspace_file(str(self.root), f'{alias}/setup.py')

    def test_outside_aliases(self):
        (self.base / 'outside.py').write_text('# synthetic')
        (self.root / 'alias.py').symlink_to(self.base / 'outside.py')
        (self.root / 'outside').symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            setup.workspace_file(str(self.root), 'alias.py')
        with self.assertRaises(ValueError):
            setup.workspace_directory(str(self.root), 'outside')

    def test_normal_inside_aliases_and_workspace_alias(self):
        (self.root / 'alias.py').symlink_to('model.py')
        (self.root / 'folder').mkdir()
        (self.root / 'aliasdir').symlink_to('folder', target_is_directory=True)
        (self.base / 'work-alias').symlink_to(self.root, target_is_directory=True)
        for root in (self.root, self.base / 'work-alias'):
            with self.subTest(root=root.name):
                self.assertEqual(setup.workspace_file(str(root), 'alias.py'), self.root / 'model.py')
                self.assertEqual(setup.workspace_directory(str(root), 'aliasdir'), self.root / 'folder')
                self.assertEqual(setup.workspace_directory(str(root), '.'), self.root)

    def test_size_and_wrong_kind(self):
        with self.assertRaises(ValueError):
            setup.workspace_file(str(self.root), 'model.py', max_bytes=1)
        (self.root / 'folder').mkdir()
        with self.assertRaises(ValueError):
            setup.workspace_file(str(self.root), 'folder')
        with self.assertRaises(ValueError):
            setup.workspace_directory(str(self.root), 'model.py')

if __name__ == '__main__':
    unittest.main()
