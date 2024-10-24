import textwrap
import subprocess
import sys
from pathlib import Path
import pytest
import importlib.util


@pytest.mark.xfail(reason="value introspection doesn't currently work")
def test_generator(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def func(gen):
            return ((0, el) for el in gen)

        print(list(func(i for i in range(10))))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    
    assert "def func(iter: Generator[int, None, None]) -> Generator[int, None, None]" in Path("t.py").read_text()


@pytest.mark.xfail(reason="value introspection doesn't currently work")
def test_iterable(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def func(iter):
            return enumerate(iter)

        print(list(func(range(10))))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    
    assert "def func(iter: Iterable[int]) -> Iterable[Tuple[int, int]]" in Path("t.py").read_text()


def test_builtins(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def func(s):
            return range(s.start, s.stop)

        print(list(func(slice(1,3))))

        def func2(t):
            return t.__name__

        print(func2(type(str)))

        def func3(t):
            pass

        func3(super(str))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    output = Path("t.py").read_text()
    
    assert "import slice" not in output
    assert "def func(s: slice) -> Iterable[int]" in output

    assert "import type" not in output
    assert "def func2(t: type) -> str" in output

    assert "import super" not in output
    assert "def func3(t: super) -> None" in output


@pytest.mark.skipif(importlib.util.find_spec('ml_dtypes') is None, reason='missing module ml_dtypes')
def test_numpy_dtype_name(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        import numpy as np
        import ml_dtypes

        def func(p):
            return str(p)

        bfloat16 = np.dtype(ml_dtypes.bfloat16)
        func(np.array([0], bfloat16))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    output = Path("t.py").read_text()

    assert "import bfloat16" not in output
    assert "def func(p: \"numpy.ndarray[Any, numpy.dtype[ml_dtypes.bfloat16]]\") -> str" in output


def test_call_with_none_default(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def func(n=None):
            return n+1 if n is not None else 0

        func()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    output = Path("t.py").read_text()
    
    assert "def func(n=None) -> int" in output


def test_default_arg(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def func(n=None):
            return n+1 if n is not None else 0

        def func2(n=5):
            return n+1

        func(1)
        func2(1.0)
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    output = Path("t.py").read_text()
    
    assert "def func(n: Optional[int]=None) -> int" in output

    # FIXME Union arguments may change order
    assert "def func2(n: Union[float, int]=5) -> float" in output


def test_function_lookup_for_defaults(tmp_path, monkeypatch):
    # if it confuses time.time for C.time, an exception is raised, as inspect cannot
    # introspect into time.time
    t = textwrap.dedent("""\
        from time import time

        class C:
            def time(self):
                return 0

        C().time()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    # FIXME we lack class support
#    output = Path("t.py").read_text()
#    assert "def time(self) -> int" in output


@pytest.mark.xfail(reason="inner functions/classes not yet supported")
def test_inner_function(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def f(x):
            def g(y):
                return y+1

            return g(x)

        class C:
            def h(self, n):
                return n+1

        f(1)
        C().h(1)
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    output = Path("t.py").read_text()
    
    assert "def g(y: int) -> int" in output
    assert "def h(self, n: int) -> int" in output   # FIXME type for 'self'?


@pytest.mark.xfail(reason="inner functions/classes not yet supported")
def test_default_inner_function(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def f(x):
            def g(y=None):
                return int(y)+1

            return g(x)

        class C:
            def h(self, n=5):
                return n+1

        f(1)
        C().h()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'])
    output = Path("t.py").read_text()
    
    assert "def g(y: Optional[int]=None) -> int" in output
    assert "def h(self, n: int=5) -> int" in output   # FIXME type for 'self'?
