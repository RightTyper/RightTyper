import textwrap
import subprocess
import sys
from pathlib import Path


def test_issue_22(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def extracted_function(A):
            return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(A[i] >=
                A[i + 1] for i in range(len(A) - 1))

        def optimized(A):
            return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) and all(A[i] >=
                A[i + 1] for i in range(len(A) - 1))

        def main():
            assert extracted_function([6, 5, 4, 4]) == True
            assert extracted_function([1, 2, 2, 3]) == True
            assert extracted_function([1, 3, 2]) == False

        if __name__ == '__main__':
            main()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', 'run', 't.py'])

    assert "def extracted_function(A: list[int]) -> bool" in Path("t.py").read_text()


def test_issue_189_no_output_when_pytest_never_collected(tmp_path, monkeypatch):
    """A pytest run that died before collecting must not rewrite the tree.

    0.1.0-era flags get forwarded to pytest, which rejects them and exits 4 -- but
    only *after* importing conftest, so observations are not empty and "did we see
    anything?" cannot tell the two apart.  The output phase then rewrote 568 files
    in the reporter's tree, leaving .bak copies a gitignore hid.  See #189.
    """
    monkeypatch.chdir(tmp_path)
    m = textwrap.dedent("""\
        def f(x):
            return x + 1

        CONST = f(1)
        """)
    Path("m.py").write_text(m)
    Path("conftest.py").write_text("import m\n")

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.',
         '-m', 'pytest', '--no-such-option'],
        capture_output=True, text=True, timeout=60,
    )

    assert p.returncode != 0
    assert Path("m.py").read_text() == m, "source was rewritten after an aborted run"
    assert not Path("m.py.bak").exists()


def test_issue_189_failing_tests_still_annotate(tmp_path, monkeypatch):
    """The gate must not catch a real run that merely exited non-zero."""
    monkeypatch.chdir(tmp_path)
    Path("m.py").write_text(textwrap.dedent("""\
        def f(x):
            return x + 1
        """))
    Path("test_m.py").write_text(textwrap.dedent("""\
        from m import f

        def test_ok():
            assert f(1) == 2

        def test_fails():
            assert False
        """))

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.', '-m', 'pytest'],
        capture_output=True, text=True, timeout=60,
    )

    assert p.returncode != 0, "expected pytest to report the failing test"
    assert "def f(x: int) -> int:" in Path("m.py").read_text()


def test_issue_189_no_pickle_when_pytest_never_collected(tmp_path, monkeypatch):
    """--only-collect must not defer the same rewrite to `process`."""
    monkeypatch.chdir(tmp_path)
    m = textwrap.dedent("""\
        def f(x):
            return x + 1

        CONST = f(1)
        """)
    Path("m.py").write_text(m)
    Path("conftest.py").write_text("import m\n")

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.', '--only-collect',
         '-m', 'pytest', '--no-such-option'],
        capture_output=True, text=True, timeout=60,
    )

    assert p.returncode != 0
    assert not list(Path(".").glob("righttyper-*.rt")), "collected an aborted run"
    assert Path("m.py").read_text() == m
