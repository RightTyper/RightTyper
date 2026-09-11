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


def test_issue_200(tmp_path, monkeypatch):
    """A TypedDict annotation on an overridden method must not kill `process`.

    A TypedDict subclass satisfies isinstance(x, type) but raises on issubclass
    by design. It reaches lub() because _propagate_to_parents merges the child's
    observed argument type with the parent's *declared* one, so the TypedDict
    arrives as a type_obj even though no runtime value ever has that type.

    The crash is intermittent by nature: Rule 4 is gated on both operands being
    argless, so calling the override with a dict yields dict[str, int]|Payload
    and survives. Passing a str keeps the observed type argless, which is what
    makes it reach issubclass.
    """
    t = textwrap.dedent("""\
        from typing import TypedDict

        class Payload(TypedDict):
            a: int

        class Base:
            def handle(self, p: Payload) -> None: ...

        class Child(Base):
            def handle(self, p):
                return len(p)

        Child().handle("xy")
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'])

    # Silent failure: exit 0 with no output. Assert on the artifact.
    log = Path("righttyper.log")
    assert not log.exists() or "TypedDict does not support" not in log.read_text()
    # Union member order differs between the rewrite and the diff paths, so
    # assert on the members rather than on a rendered ordering.
    annotated = Path("t.py").read_text()
    child_sig = next(
        line for line in annotated.splitlines()
        if line.strip().startswith("def handle") and "-> int" in line
    )
    assert "Payload" in child_sig and "str" in child_sig, child_sig


def test_issue_200_self_compatibility_check(tmp_path, monkeypatch):
    """The Self-compatibility check in _clone_for_context needs the same guard.

    Routing _is_subtype through safe_issubclass covered generalize, but
    _CloneForContextT.__init__ has its own `issubclass(source, dest)` whose second
    operand is caller-derived too -- and it sits in _propagate_to_parents, the very
    path by which such classes arrive. A non-runtime_checkable Protocol with data
    members raises there, and the run failed the same silent way: exit 0, nothing
    rewritten.
    """
    t = textwrap.dedent("""\
        from typing import Protocol

        class Base(Protocol):
            x: int
            def handle(self, p): ...

        class Child(Base):
            x = 1
            def handle(self, p):
                return len(p)

        Child().handle("xy")
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'])

    log = Path("righttyper.log")
    assert not log.exists() or "runtime_checkable" not in log.read_text()
    annotated = Path("t.py").read_text()
    assert "-> int" in annotated, annotated
