import textwrap
import subprocess
import sys
from pathlib import Path
import pytest
import importlib.util

@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.mark.xfail(reason="value introspection doesn't currently work")
def test_iterable(tmp_cwd):
    t = textwrap.dedent("""\
        def func(iter):
            return enumerate(iter)

        print(list(func(range(10))))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    
    assert "def func(iter: Iterable[int]) -> Iterable[Tuple[int, int]]" in Path("t.py").read_text()


def test_builtins(tmp_cwd):
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


@pytest.mark.skipif((importlib.util.find_spec('ml_dtypes') is None or
                     importlib.util.find_spec('numpy') is None),
                    reason='missing modules')
def test_numpy_dtype_name(tmp_cwd):
    t = textwrap.dedent("""\
        import numpy as np
        import ml_dtypes

        def func(p):
            return str(p)

        bfloat16 = np.dtype(ml_dtypes.bfloat16)
        func(np.array([0], bfloat16))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "import bfloat16" not in output
    assert "def func(p: \"numpy.ndarray[Any, numpy.dtype[ml_dtypes.bfloat16]]\") -> str" in output


def test_call_with_none_default(tmp_cwd):
    t = textwrap.dedent("""\
        def func(n=None):
            return n+1 if n is not None else 0

        func()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def func(n=None) -> int" in output


def test_default_arg(tmp_cwd):
    t = textwrap.dedent("""\
        def func(n=None):
            return n+1 if n is not None else 0

        def func2(n=5):
            return n+1

        func(1)
        func2(1.0)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def func(n: Optional[int]=None) -> int" in output

    # FIXME Union arguments may change order
    assert "def func2(n: Union[float, int]=5) -> float" in output


def test_function_lookup_for_defaults(tmp_cwd):
    # if it confuses time.time for C.time, an exception is raised, as inspect cannot
    # introspect into time.time
    t = textwrap.dedent("""\
        from time import time

        class C:
            def time(self):
                return 0

        C().time()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    # FIXME we lack class support
#    output = Path("t.py").read_text()
#    assert "def time(self) -> int" in output


def test_inner_function(tmp_cwd):
    t = textwrap.dedent("""\
        def f(x):
            def g(y):
                return y+1

            return g(x)

        f(1)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def g(y: int) -> int" in output


def test_class_method(tmp_cwd):
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

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def f(self: Self, n: int) -> int" in output
    assert "import Self" not in output

    assert "def h(self: Self, x: int) -> float" in output


def test_class_method_imported(tmp_cwd):
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


def test_class_name_imported(tmp_cwd):
    Path("m.py").write_text(textwrap.dedent("""\
        class C:
            pass

        def f(x):
            pass

        def g():
            f(C())
        """
    ))

    Path("t.py").write_text(textwrap.dedent("""\
        import m
        m.g()
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("m.py").read_text()
    
    assert "def f(x: C) -> None" in output
    assert "import C" not in output


def test_class_name_in_test(tmp_cwd):
    (tmp_cwd / "tests").mkdir()
    (tmp_cwd / "tests" / "test_foo.py").write_text(textwrap.dedent("""\
        class C:
            pass

        def f(x):
            pass

        def test_foo():
            f(C())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '-m', 'pytest', '-s', 'tests'], check=True)
    output = (tmp_cwd / "tests" / "test_foo.py").read_text()
    
    assert "def f(x: C) -> None" in output
    assert "import test_foo" not in output


@pytest.mark.xfail(reason="Doesn't work yet")
def test_local_class_name(tmp_cwd):
    (tmp_cwd / "t.py").write_text(textwrap.dedent("""\
        def f():
            class C:
                pass

            def g(x):
                return 0

            return g(C())

        f()
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = (tmp_cwd / "t.py").read_text()
    
    assert "def g(x: C) -> int" in output


def test_return_private_class(tmp_cwd):
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


def test_default_inner_function(tmp_cwd):
    t = textwrap.dedent("""\
        def f(x):
            def g(y=None):
                return int(y)+1

            return g(x)

        f(1)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def g(y: Optional[int]=None) -> int" in output


def test_default_class_method(tmp_cwd):
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

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    assert "def f(self: Self, n: int=5) -> int" in output
    assert "def h(self: Self, x: int=1) -> float" in output


def test_generator(tmp_cwd):
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

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    # FIXME should be Generator[int] or Iterator[int]
    assert "def gen() -> Generator[int, Any, Any]:" in output
    assert "def g(f: Generator[Any, Any, Any]) -> None" in output


def test_async_generator(tmp_cwd):
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

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()
    
    # FIXME should be AsyncGenerator[int] or AsyncIterator[int]
    assert "def gen() -> AsyncGenerator[Any, Any]:" in output
    assert "def g(f: AsyncGenerator[Any, Any]) -> None" in output


def test_generate_stubs(tmp_cwd):
    Path("m.py").write_text(textwrap.dedent("""\
        import sys

        CONST = 42
        CALC = 1+1

        class C:
            PI = 314

            def __init__(self, x):  # initializes me
                self.x = x

            def f(self):
                return self.x

        def f(x):
            return C(x).f()
        """
    ))

    Path("t.py").write_text(textwrap.dedent("""\
        import m
        m.f(1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--generate-stubs',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("m.pyi").read_text()
    # FIXME this assertion is brittle
    assert output == textwrap.dedent("""\
        from typing import TYPE_CHECKING, Self
        import sys
        from typing import Any
        CONST: int
        CALC: Any
        class C:
            PI: int
            def __init__(self: Self, x: int) -> None: ...
            def f(self: Self) -> int: ...
        def f(x: int) -> int: ...
        """)


def test_coroutine_type(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        def foo():
            async def coro():
                import asyncio
                await asyncio.sleep(1)
            return coro()

        foo()
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo() -> Coroutine[Any, Any, Any]:" in output


def test_module_type(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        import sys

        def foo(m):
            pass

        foo(sys.modules['__main__'])
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo(m: \"types.ModuleType\") -> None:" in output
    assert "import types" in output


def test_function_type(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x: int) -> float:
            return x/2

        class C:
            def foo2(self: "C", x: int) -> float:
                return x*.5

        def bar(f, g, x):
            return f(x) + g(C(), x)

        def baz(h, x):
            return h(x)

        bar(foo, C.foo2, 1)
        baz(C().foo2, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def bar(f: Callable[[int], float], g: Callable[["C", int], float], x: int) -> float:' in output
    assert 'def baz(h: Callable[[int], float], x: int) -> float:' in output # bound method


def test_function_type_future_annotations(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        from __future__ import annotations

        def foo(x: int) -> float:
            return x/2

        class C:
            def foo2(self: "C", x: int) -> float:
                return x*.5

        def bar(f, g, x):
            return f(x) + g(C(), x)

        def baz(h, x):
            return h(x)

        bar(foo, C.foo2, 1)
        baz(C().foo2, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def bar(f: Callable[["int"], "float"], g: Callable[["C", "int"], "float"], x: int) -> float:' in output
    assert 'def baz(h: Callable[["int"], "float"], x: int) -> float:' in output # bound method


def test_function_type_in_annotation(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        from types import FunctionType

        def foo(x: int) -> float:
            return x/2

        def bar(g: FunctionType, x):
            return g(x)

        def baz(f, g, x):
            return bar(g, x)

        baz(bar, foo, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def bar(g: FunctionType, x: int) -> float:' in output
    # FIXME the quotes around the type are because UnifiedTransformer doesn't realize it's already known
    assert 'def baz(f: "Callable[[types.FunctionType, Any], Any]", g: Callable[[int], float], x: int) -> float:' in output


@pytest.mark.xfail(reason="our function types are all annotation-based so far")
def test_discovered_function_type(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x):
            return x/2

        def bar(f, x):
            return f(x)

        bar(foo, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo(x: int) -> float:" in output
    assert "def bar(f: Callable[[int], float], x: int) -> float:" in output
