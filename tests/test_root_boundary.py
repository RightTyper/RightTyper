"""Exercise root filtering through collection and source rewriting."""

import subprocess
import sys

from righttyper.options import run_options
from righttyper.righttyper_utils import skip_this_file


def test_root_filter_excludes_sibling_prefix(tmp_path, monkeypatch):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n")
    sibling = tmp_path / "pkg_extra" / "module.py"
    sibling.parent.mkdir()
    sibling.write_text("value = 2\n")
    monkeypatch.setattr(run_options, "script_dir", str(root.resolve()))
    skip_this_file.cache_clear()
    try:
        assert not skip_this_file(str(root / "module.py"))
        assert skip_this_file(str(sibling))
    finally:
        skip_this_file.cache_clear()


def test_root_excludes_sibling_prefix_during_rewrite(tmp_path):
    source = "def identity(value):\n    return value\n"
    for package in ("pkg", "pkg_extra"):
        directory = tmp_path / package
        directory.mkdir()
        (directory / "__init__.py").write_text("")
        (directory / "module.py").write_text(source)
    (tmp_path / "pkg" / "main.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from pkg.module import identity\n"
        "from pkg_extra.module import identity as sibling\n"
        "identity(1)\nsibling(2)\n"
    )
    collected = subprocess.run(
        [
            sys.executable, "-m", "righttyper", "run",
            "--no-use-multiprocessing", "--allow-runtime-exceptions",
            "--only-collect", "--root", str(tmp_path / "pkg"), "pkg/main.py",
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    processed = subprocess.run(
        [sys.executable, "-m", "righttyper", "process"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert processed.returncode == 0, processed.stdout + processed.stderr
    assert "value: int" in (tmp_path / "pkg" / "module.py").read_text()
    assert (tmp_path / "pkg_extra" / "module.py").read_text() == source
