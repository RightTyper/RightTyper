"""Verify target consistency with files produced by the real collector."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("scripts", [("first.py", "second.py"), ("second.py", "first.py")])
def test_process_rejects_collections_from_different_targets(tmp_path, scripts):
    source = "def identity(value):\n    return value\nidentity(1)\n"
    for name in scripts:
        (tmp_path / name).write_text(source)
        result = subprocess.run(
            [
                sys.executable, "-m", "righttyper", "run",
                "--no-use-multiprocessing", "--allow-runtime-exceptions",
                "--only-collect", name,
            ],
            cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(tmp_path.glob("righttyper-*.rt"))) == 2
    result = subprocess.run(
        [sys.executable, "-m", "righttyper", "process"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "was collected for" in result.stdout
    assert "previous file(s) were for" in result.stdout
    for name in scripts:
        assert (tmp_path / name).read_text() == source
