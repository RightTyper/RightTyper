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


def test_issue_197_unhashable_class_on_recording_path(tmp_path, monkeypatch):
    """The recording path keys tables by type too, ahead of lub() and TypeMap.

    get_value_type/get_type_name look the observed type up in _BUILTINS and
    _type2handler. Those are plain dicts, so an unhashable class raised there
    before generalize or typemap ever saw it, and the function was left with no
    annotation at all -- not even the return type, which has nothing to do with
    the offending argument.
    """
    t = textwrap.dedent("""\
        class Meta(type):
            def __eq__(cls, other):
                return NotImplemented
            __hash__ = None

        class Unhashable(metaclass=Meta):
            pass

        def f(x):
            return 1

        f(Unhashable())
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'])

    log = Path("righttyper.log")
    assert not log.exists() or "as a dict key" not in log.read_text()
    # The argument stays unannotated -- TypeMap cannot name a class it could not
    # enter -- but the rest of the signature must still be inferred.
    annotated = Path("t.py").read_text()
    assert "def f(x) -> int:" in annotated, annotated
