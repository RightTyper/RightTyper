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


def test_issue_193(tmp_path, monkeypatch):
    # mock's _Call synthesizes __code__ as an unhashable child _Call
    t = textwrap.dedent("""\
        from unittest.mock import call

        def build():
            return [call(a=1), call(a=2)]

        build()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--only-collect', 't.py'],
        capture_output=True, text=True
    )

    assert p.returncode == 0, p.stderr
    assert "_Call" not in p.stderr


def test_issue_193_dataclass_init(tmp_path, monkeypatch):
    # the dataclass/attrs/NamedTuple branch reads __code__ off __init__/__new__
    t = textwrap.dedent("""\
        import dataclasses
        from unittest.mock import call

        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        class FakeInit:
            __code__ = call.something   # truthy, non-code, unhashable

            def __call__(self, *args, **kwargs):
                pass

        Point.__init__ = FakeInit()

        print(type(Point(1, 2)).__name__)
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--only-collect', 't.py'],
        capture_output=True, text=True
    )

    assert p.returncode == 0, p.stderr
    assert "code must be a code object" not in p.stderr
    assert "Point" in p.stdout      # the script really did run to completion


def test_issue_193_mock_in_class_dict(tmp_path, monkeypatch):
    """mock.patch.object leaves a __wrapped__-synthesizing object in a class __dict__."""
    t = textwrap.dedent("""\
        from unittest.mock import call

        class Base:
            def m(self):
                return 0

        class Child(Base):
            def m(self):
                return 1

        Base.m = call.patched

        print(Child().m())
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'],
        capture_output=True, text=True, timeout=20,
    )

    assert p.returncode == 0, f"exit {p.returncode} (137 = OOM-killed)\n{p.stderr}"
    annotated = Path("t.py").read_text()
    assert "-> int" in annotated, annotated


def test_issue_193_raising_getattr(tmp_path, monkeypatch):
    """A __getattr__ raising a non-AttributeError must not escape the CALL handler."""
    t = textwrap.dedent("""\
        class Proxy:
            "__code__ probe: __getattr__ raises something getattr won't suppress"
            def __getattr__(self, name):
                raise ImportError(f"cannot resolve {name}")

            def __call__(self, x):
                return x + 1

        class RaisingLink:
            def __getattr__(self, name):
                raise KeyError(name)

            def __call__(self, x):
                return x

        class Wrapper:
            "__wrapped__ probe: in __dict__, so unwrap() walks into RaisingLink"
            def __init__(self, inner):
                self.__wrapped__ = inner

            def __call__(self, x):
                return self.__wrapped__(x)

        def observed(v):
            return v * 2

        print(Proxy()(1))
        print(Wrapper(RaisingLink())(2))
        print(observed(3))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'],
        capture_output=True, text=True, timeout=20,
    )

    assert p.returncode == 0, f"exit {p.returncode}\n{p.stderr}"
    assert "ImportError" not in p.stderr and "KeyError" not in p.stderr, p.stderr

    annotated = Path("t.py").read_text()
    assert "def observed(v: int) -> int:" in annotated, annotated


def _proxied_base_finder():
    import inspect
    from righttyper.recorder import OverrideFinder

    class Proxy:
        "as mock.patch.object leaves in a class __dict__"
        def __getattr__(self, name):
            raise ImportError(f"cannot resolve {name}")

        def __call__(self, *a, **kw):
            return 0

    class Base:
        def m(self): return 0

    class Child(Base):
        def m(self): return 1

    setattr(Base, "m", Proxy())
    code = Child.m.__code__
    return Base, OverrideFinder(Child.__mro__, "m", 1), code, inspect.getargs(code)


def test_issue_193_hostile_override_is_still_the_override():
    """A proxy in Base's __dict__ is what Child.m overrides, even if it won't say its name."""
    Base, finder, code, arg_info = _proxied_base_finder()

    result = finder.find_next(code, arg_info)

    assert result is not None
    overridden, parent_func, defining_class = result
    assert defining_class is Base
    assert overridden.qualname == f"{Base.__qualname__}.m"
    assert parent_func is None


def test_issue_193_find_next_does_not_hide_parent_arg_errors(monkeypatch):
    """Only the attribute reads are guarded; a real bug must not redirect to the grandparent."""
    import pytest
    import righttyper.recorder as recorder

    def broken(*args, **kwargs):
        raise AttributeError("bug in hint handling")

    monkeypatch.setattr(recorder, "get_parent_arg_types", broken)
    _, finder, code, arg_info = _proxied_base_finder()

    with pytest.raises(AttributeError, match="bug in hint handling"):
        finder.find_next(code, arg_info)


def test_issue_193_raising_getattr_anywhere_in_call_handler(tmp_path, monkeypatch):
    """No probe in the CALL handler may let a hostile __getattr__ out, not just __code__'s."""
    t = textwrap.dedent("""\
        class Meta(type):
            "dataclass/attrs/NamedTuple probe: hasattr on the class"
            def __getattr__(cls, name):
                raise RuntimeError(f"cannot resolve {name}")

        class Hostile(metaclass=Meta):
            pass

        class Slotted:
            "__dict__ probe: no __dict__, so getattr falls through to __getattr__"
            __slots__ = ()
            def __getattr__(self, name):
                raise RuntimeError(f"cannot resolve {name}")

            def __call__(self, x):
                return x + 1

        def observed(v):
            return v * 2

        print(type(Hostile()).__name__)
        print(Slotted()(1))
        print(observed(3))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'],
        capture_output=True, text=True, timeout=20,
    )

    assert p.returncode == 0, f"exit {p.returncode}\n{p.stderr}"
    assert "RuntimeError" not in p.stderr, p.stderr
    assert p.stdout.split() == ["Hostile", "2", "6"], p.stdout

    annotated = Path("t.py").read_text()
    assert "def observed(v: int) -> int:" in annotated, annotated


def test_issue_193_raising_metaclass_getattr_on_first_argument(tmp_path, monkeypatch):
    """A function whose first argument's class raises from a metaclass __getattr__ is still annotated."""
    t = textwrap.dedent("""\
        class Meta(type):
            def __getattr__(cls, name):
                raise RuntimeError(f"cannot resolve {name}")

        class Hostile(metaclass=Meta):
            pass

        def helper(obj, n):
            return n + 1

        print(helper(Hostile(), 1))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    p = subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'],
        capture_output=True, text=True, timeout=20,
    )

    assert p.returncode == 0, f"exit {p.returncode}\n{p.stderr}"
    assert p.stdout.split() == ["2"], p.stdout
    assert "def helper(obj: Hostile, n: int) -> int:" in Path("t.py").read_text()
