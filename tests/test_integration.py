import textwrap
import subprocess
import sys
from pathlib import Path
import pytest
import importlib.util


@pytest.mark.xfail(reason="value introspection doesn't currently work")
def test_iterable(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def func(iter):
            return enumerate(iter)

        print(list(func(range(10))))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    
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

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
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

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
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

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
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

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
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

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    # FIXME we lack class support
#    output = Path("t.py").read_text()
#    assert "def time(self) -> int" in output


def test_inner_function(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def f(x):
            def g(y):
                return y+1

            return g(x)

        f(1)
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def g(y: int) -> int" in output


def test_class_method(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        class C:
            def f(self, n):
                return n+1

        def g(x):
            class gC:
                def h(self, x):
                    return x/2

            return gC().h(x)

        C().f(1)
        g(1)
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def f(self: Self, n: int) -> int" in output
    assert "import Self" not in output

    assert "def h(self: Self, x: int) -> float" in output


def test_class_method_imported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("m.py").write_text(textwrap.dedent("""\
        class C:
            def f(self, n):
                return n+1

        def g(x):
            class gC:
                def h(self, x):
                    return x/2

            return gC().h(x)
        """
    ))

    Path("t.py").write_text(textwrap.dedent("""\
        import m
        m.C().f(1)
        m.g(1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("m.py").read_text()
    
    assert "import Self" not in output

    assert "def f(self: Self, n: int) -> int" in output
    assert "import C" not in output

    assert "def g(x: int) -> float" in output
    assert "def h(self: Self, x: int) -> float" in output
    assert "import gC" not in output


def test_return_private_class(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(textwrap.dedent("""\
        def f():
            class fC:
                pass
            return fC()

        def g(x):
            pass

        g(f())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    # that local class name is "f.<locals>.fC"; this yields a CST ParserSyntaxError
    assert "import fC" not in output
    assert "def f():" in output # FIXME what is a good way to express the return type?
    assert "def g(x) -> None:" in output # FIXME what is a good way to express the type?


def test_default_inner_function(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def f(x):
            def g(y=None):
                return int(y)+1

            return g(x)

        f(1)
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def g(y: Optional[int]=None) -> int" in output


def test_default_class_method(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        class C:
            def f(self, n=5):
                return n+1

        def g():
            class gC:
                def h(self, x=1):
                    return x/2

            return gC().h()

        C().f()
        g()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def f(self: Self, n: int=5) -> int" in output
    assert "def h(self: Self, x: int=1) -> float" in output


def test_generator(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def gen():
            yield 10

        def main():
            for _ in gen():
                pass

        def g(f):
            pass

        main()
        g(gen())
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    # FIXME should be Generator[int] or Iterator[int]
    assert "def gen() -> Generator[int, Any, Any]:" in output
    assert "def g(f: Generator[Any, Any, Any]) -> None" in output


def test_async_generator(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        import asyncio

        async def gen():
            yield 10

        async def main():
            async for _ in gen():
                pass

        def g(f):
            pass

        asyncio.run(main())
        g(gen())
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    # FIXME should be AsyncGenerator[int] or AsyncIterator[int]
    assert "def gen() -> AsyncGenerator[Any, Any]:" in output
    assert "def g(f: AsyncGenerator[Any, Any]) -> None" in output
