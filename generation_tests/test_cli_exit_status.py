"""Process-level checks on the ``stlmc-x`` console entry point exit status.

Runs the installed console script as a subprocess and reads its exit code: a
malformed model reaches a handled-error branch and exits nonzero, while
``--version`` exits zero. Skipped when the console script is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

_STLMC_X = shutil.which("stlmc-x")

pytestmark = pytest.mark.skipif(
    _STLMC_X is None, reason="stlmc-x console script not on PATH"
)


def _run(args: list[str], cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_STLMC_X, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_version_flag_exits_zero(tmp_path):
    proc = _run(["--version"], tmp_path)
    assert proc.returncode == 0
    assert proc.stdout.strip()


def test_malformed_model_exits_nonzero(tmp_path):
    model = tmp_path / "broken.model"
    model.write_text("this is not a valid model\n")
    proc = _run([str(model), "-bound", "1", "-goal", "f1", "-solver", "z3"], tmp_path)
    assert proc.returncode != 0


def test_missing_model_exits_nonzero(tmp_path):
    missing = tmp_path / "does_not_exist.model"
    proc = _run([str(missing), "-bound", "1", "-goal", "f1", "-solver", "z3"], tmp_path)
    assert proc.returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
