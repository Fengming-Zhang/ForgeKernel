import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class StartTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('start', SCRIPTS / 'start.py')
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_start_runs_after_creation(self):
        create = Mock(return_value=Path('/run/one'))
        run = Mock(return_value=0)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.m.run_once(Path(d) / 'session.json', '/task', create, run), 0)
        create.assert_called_once()
        run.assert_called_once_with(Path('/run/one'))

    def test_repeat_resumes_same_run(self):
        create = Mock(return_value=Path('/run/one'))
        run = Mock(return_value=2)
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'session.json'
            self.m.run_once(state, '/task', create, run)
            self.m.run_once(state, '/task', create, run)
        create.assert_called_once()
        self.assertEqual(run.call_count, 2)

    def test_different_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'session.json'
            self.m.run_once(state, '/task', lambda: Path('/run/one'), lambda _: 0)
            with self.assertRaises(ValueError):
                self.m.run_once(state, '/other', Mock(), Mock())

    def test_corrupt_state_is_not_new_run(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'session.json'; state.write_text('broken')
            create = Mock()
            with self.assertRaises(ValueError):
                self.m.run_once(state, '/task', create, Mock())
            create.assert_not_called()

    def test_gpu_probe_reports_remote_error(self):
        import subprocess
        failure = subprocess.CalledProcessError(1, ['ssh'], stderr='Missing remote environment')
        with patch.object(self.m.subprocess, 'run', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'Missing remote environment'):
                self.m.probe_gpu(['ssh'])

    def test_arch_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.m.check_arch('sm90', [8, 0])
        self.m.check_arch('sm90', [9, 0])


if __name__ == '__main__':
    unittest.main()
