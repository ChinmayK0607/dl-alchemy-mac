"""CPU portability checks without importing MLX, torch, or course modules."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import paths
from .runtime import configure_memory, configure_test_device, validate_memory_gb


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.scratch = paths.REPO_ROOT / ".build/tmp"
        self.scratch.mkdir(parents=True, exist_ok=True)

    def test_reference_precedence_and_errors(self):
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="alchemy paths ") as tmp:
            base = Path(tmp)
            root = base / "clone with spaces"
            submodule, sibling, override = root / "reference", base / "dl-alchemy-a1", base / "override"
            for directory in (submodule, sibling, override):
                directory.mkdir(parents=True)
                for name in paths._REFERENCE_FILES:
                    (directory / name).touch()
            with patch.object(paths, "REPO_ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.reference_path(), submodule.resolve())
                with patch.dict(os.environ, DL_ALCHEMY_REFERENCE=str(override)):
                    self.assertEqual(paths.reference_path(), override.resolve())
                (submodule / "modeling.py").unlink()
                self.assertEqual(paths.reference_path(), sibling.resolve())
                for value in (str(base / "missing"), "", "  "):
                    with patch.dict(os.environ, DL_ALCHEMY_REFERENCE=value):
                        with self.assertRaisesRegex(FileNotFoundError, "git submodule update"):
                            paths.reference_path()
                (sibling / "train.py").unlink()
                with self.assertRaisesRegex(FileNotFoundError, paths.REFERENCE_COMMIT):
                    paths.reference_path()

    def test_memory_scales_and_override_wins(self):
        for recommended in (20 * 1024**3, 48 * 1024**3, 96 * 1024**3):
            calls = []
            mx = SimpleNamespace(device_info=lambda: {"max_recommended_working_set_size": recommended},
                                 set_memory_limit=lambda n: calls.append(("allocation", n)),
                                 set_wired_limit=lambda n: calls.append(("wired", n)))
            limit = configure_memory(mx)
            self.assertEqual(limit, int(.75 * recommended))
            self.assertEqual(calls, [("allocation", limit), ("wired", limit)])
            # An explicit cap must not query device-specific defaults.
            mx.device_info = lambda: self.fail("Override queried the device default")
            self.assertEqual(configure_memory(mx, 12.5), int(12.5 * 1024**3))

    def test_invalid_or_unavailable_memory_limits(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_memory_gb(value)
        for info in ({}, {"max_recommended_working_set_size": 0}):
            with self.assertRaisesRegex(RuntimeError, "--memory-gb"):
                configure_memory(SimpleNamespace(device_info=lambda: info))

    def test_test_device_preserves_default_unless_requested(self):
        calls = []
        mx = SimpleNamespace(cpu="cpu", gpu="gpu", default_device=lambda: "gpu",
                             set_default_device=calls.append)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configure_test_device(mx), "gpu")
            self.assertEqual(calls, [])
            with patch.dict(os.environ, MLX_DEVICE="cpu"):
                configure_test_device(mx)
                self.assertEqual(calls, ["cpu"])
            with patch.dict(os.environ, MLX_DEVICE="invalid"):
                with self.assertRaises(ValueError):
                    configure_test_device(mx)

    def test_relocated_training_imports_and_queue_without_reference_or_torch(self):
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="alchemy clone ") as tmp:
            clone = Path(tmp) / "fresh clone"
            shutil.copytree(paths.REPO_ROOT / "mlx_port", clone / "mlx_port",
                            ignore=shutil.ignore_patterns("__pycache__"))
            (clone / "plans").mkdir()
            (clone / "plans/popquiz.json").write_text(json.dumps([
                dict(preset="local", variant="reference", question_group="q1")]))
            source = '''
import importlib.abc, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'modeling', 'model_config', 'optimizers', 'lr_schedules', 'utils'}:
            raise AssertionError('Training imported an optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
from mlx_port.paths import reference_path
try:
    reference_path()
except FileNotFoundError:
    pass
else:
    raise AssertionError('Fixture accidentally found a reference')
from mlx_port.run import parser, config_from_args
assert config_from_args(parser().parse_args(['--preset', 'reference'])).total_steps == 9375
from mlx_port.queue import execute, parser as queue_parser
execute(queue_parser().parse_args(['--dry-run', '--memory-gb', '12.5']))
'''
            proc = subprocess.run([sys.executable, "-c", source], cwd=clone, capture_output=True, text=True,
                                  env=dict(os.environ, DL_ALCHEMY_REFERENCE="missing-reference"), timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            row = json.loads(proc.stdout)
            self.assertIn("--memory-gb", row["command"])
            self.assertIn("12.5", row["command"])
            self.assertFalse((clone / "results").exists())

    def test_setup_is_repeatable_and_forwards_options(self):
        # Test shell quoting and command order using stubs, never git/network/GPU.
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="alchemy setup ") as tmp:
            root = Path(tmp) / "repo with spaces"
            (root / "scripts").mkdir(parents=True)
            shutil.copy(paths.REPO_ROOT / "scripts/setup.sh", root / "scripts/setup.sh")
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            (root / ".venv/bin").mkdir(parents=True)
            events = root / "events.txt"
            for name, file in (("git", fake_bin / "git"), ("uv", fake_bin / "uv"),
                               ("python", root / ".venv/bin/python")):
                file.write_text(f'''#!/bin/sh
printf '%s|' '{name}' "$@" >> "$T5_EVENTS"
printf '\\n' >> "$T5_EVENTS"
if [ "$1" = '-' ]; then cat >/dev/null; fi
''')
                file.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.defpath, T5_EVENTS=str(events))
            for _ in range(2):
                proc = subprocess.run(["sh", str(root / "scripts/setup.sh")], cwd=tmp, env=env,
                                      capture_output=True, text=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr)
            rows = events.read_text().splitlines()
            self.assertEqual(rows[:5], rows[5:])
            self.assertEqual(rows[0], "git|submodule|update|--init|--recursive|")
            self.assertIn("uv|sync|--locked|--python|3.12|", rows)
            self.assertIn("--num-train-sequences|100000|--data-seed|42|--merge-gap|0|", rows[3])
            proc = subprocess.run(["sh", str(root / "scripts/setup.sh"), "--parity", "--num-train-sequences",
                                   "600000", "--data-dir", "data/full set"], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=10)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            last = events.read_text().splitlines()[-5:]
            self.assertIn("--extra|parity|", last[1])
            self.assertIn("--num-train-sequences|600000|", last[3])
            self.assertIn("--data-dir|data/full set|", last[3])

    def test_gpu_check_reports_worker_failure_and_continues(self):
        spec = importlib.util.spec_from_file_location("check_gpu", paths.REPO_ROOT / "scripts/check_gpu.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        calls = []
        def fake_run(cmd, **kwargs):
            self.assertGreater(kwargs["timeout"], 0)
            self.assertNotIn("cpu", cmd)
            self.assertNotIn("parity_check", cmd)
            calls.append(cmd)
            if "mlx_port.bench" in cmd:
                output = Path(cmd[cmd.index("--output")+1])
                output.write_text(''.join(json.dumps(dict(status=s))+'\n' for s in ("ok", "ok", "crash")))
            status = "verification_complete" if "mlx_port.run" in cmd else "ok"
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(status=status))+"\n", stderr="")
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="alchemy checks ") as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.object(sys, "argv", ["check_gpu.py"]), patch.object(check.subprocess, "run", fake_run), patch.object(check.platform, "platform", return_value="test-mac"), redirect_stdout(io.StringIO()):
                    self.assertEqual(check.main(), 1)
                self.assertEqual(len(calls), 3)
                summary = json.loads(next(Path("results").glob("*/*/summary.json")).read_text())
                self.assertEqual([r["status"] for r in summary["checks"]], ["ok", "error", "ok"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
