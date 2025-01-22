import textwrap
import subprocess
import sys
from pathlib import Path
import pytest
import importlib.util
import re


@pytest.fixture(scope='function', autouse=True)
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.mark.xfail(reason="Iterable/Iterator introspection doesn't currently work")
def test_iterable():
    t = textwrap.dedent("""\
        def func(iter):
            return enumerate(iter)

        print(list(func(range(10))))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    assert "def func(iter: Iterable[int]) -> Iterable[tuple[int, int]]" in Path("t.py").read_text()


def test_builtins():
    t = textwrap.dedent("""\
        def func(s):
            return range(s.start, s.stop)

        print(list(func(slice(1,3))))

        def func2(t):
            return t.__name__

        print(func2(str))

        def func3(t):
            pass

        func3(super(str))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "import slice" not in output
    assert "def func(s: slice) -> range" in output

    assert "import type" not in output
    assert "def func2(t: type[str]) -> str" in output

    assert "import super" not in output
    assert "def func3(t: super) -> None" in output


def test_type_from_generic_alias_annotation():
    t = textwrap.dedent("""\
        def f() -> list[int]: ...   # list[int] is a GenericAlias

        def g():
            return f

        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable[[], list[int]]:" in output


def test_type_from_annotation_none_return():
    t = textwrap.dedent("""\
        def f() -> None: ...

        def g():
            return f

        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable[[], None]:" in output


@pytest.mark.skipif((importlib.util.find_spec('ml_dtypes') is None or
                     importlib.util.find_spec('numpy') is None),
                    reason='missing modules')
def test_numpy_type_name():
    t = textwrap.dedent("""\
        import numpy as np
        import ml_dtypes

        def f(t):
            pass

        f(np.dtype(ml_dtypes.bfloat16))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "import bfloat16" not in output
    assert "def f(t: np.dtype[ml_dtypes.bfloat16]) -> None" in output


@pytest.mark.skipif((importlib.util.find_spec('ml_dtypes') is None or
                     importlib.util.find_spec('numpy') is None),
                    reason='missing modules')
def test_numpy_ndarray_dtype_name():
    t = textwrap.dedent("""\
        import numpy as np
        import ml_dtypes

        def f(p):
            return str(p)

        bfloat16 = np.dtype(ml_dtypes.bfloat16)
        f(np.array([0], bfloat16))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "import bfloat16\n" not in output
    assert "def f(p: np.ndarray[Any, np.dtype[ml_dtypes.bfloat16]]) -> str" in output


@pytest.mark.skipif((importlib.util.find_spec('ml_dtypes') is None or
                     importlib.util.find_spec('numpy') is None),
                    reason='missing modules')
def test_annotation_with_numpy_dtype_name():
    t = textwrap.dedent("""\
        from typing import Any
        import numpy as np
        from ml_dtypes import bfloat16 as bf16

        def f() -> np.ndarray[Any, np.dtype[bf16]]: ...

        def g():
            return f

        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable[[], np.ndarray[Any, np.dtype[bf16]]]:" in output


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None,
                    reason='missing module numpy')
def test_internal_numpy_type():
    t = textwrap.dedent("""\
        import numpy as np
        from numpy.core.overrides import array_function_dispatch

        def sum_dispatcher(arg):
            return (arg,)

        @array_function_dispatch(sum_dispatcher)
        def my_sum(arg):
            return np.sum(arg)

        class MyArray:
            def __init__(self, data):
                self.data = np.array(data)

            def __array_function__(self, func, types, args, kwargs):
                return func(self.data, *args[1:], **kwargs)

        my_sum(MyArray([1, 2, 3, 4]))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert (m := re.search(r'__array_function__\(self[^,]*, (func[^,]*),', output))
    assert m.group(1).startswith('func: "numpy.')
    assert m.group(1).endswith('_ArrayFunctionDispatcher"')


@pytest.mark.skipif((importlib.util.find_spec('jaxtyping') is None or
                     importlib.util.find_spec('numpy') is None),
                    reason='missing modules')
def test_jaxtyping_annotation():
    t = textwrap.dedent("""\
        import numpy as np

        def f(x):
            return x

        f(np.array([[1],[1]], dtype=np.int64))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--infer-shapes', '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert 'def f(x: "jaxtyping.Int64[np.ndarray, \\"2 1\\"]") ' +\
           '-> "jaxtyping.Int64[np.ndarray, \\"2 1\\"]"' in output


def test_call_with_none_default():
    t = textwrap.dedent("""\
        def func(n=None):
            return n+1 if n is not None else 0

        func()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def func(n: None=None) -> int" in output


def test_default_arg():
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

    assert "def func(n: int|None=None) -> int" in output

    assert "def func2(n: float|int=5) -> float" in output


def test_inner_function():
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


def test_class_method():
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
    assert "\nimport Self" not in output

    assert "def h(self: Self, x: int) -> float" in output


def test_class_method_imported():
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

    assert "\nimport Self" not in output

    assert "def f(self: Self, n: int) -> int" in output
    assert "import C" not in output

    assert "def g(x: int) -> float" in output
    assert "def h(self: Self, x: int) -> float" in output
    assert "import gC" not in output


def test_class_name_imported():
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
    Path("t.py").write_text(textwrap.dedent("""\
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


def test_return_private_class():
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


def test_default_inner_function():
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

    assert "def g(y: int|None=None) -> int" in output


def test_default_class_method():
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


def test_generator():
    t = textwrap.dedent("""\
        def gen():
            yield 10
            yield 1.2

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

    assert "def gen() -> Iterator[float|int]:" in output
    assert "def g(f: Iterator[float|int]) -> None" in output


def test_generator_with_return():
    t = textwrap.dedent("""\
        def gen():
            yield 10
            return "done"

        def main():
            g = gen()
            next(g)
            try:
                next(g)
            except StopIteration:
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

    assert "def gen() -> Generator[int, None, str]:" in output
    assert "def g(f: Generator[int, None, str]) -> None" in output


@pytest.mark.xfail(reason="Doesn't currently work")
def test_generator_from_annotation():
    t = textwrap.dedent("""\
        from typing import Generator

        def gen() -> Generator[int|str, None, None]:
            yield ""

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

    assert "def g(f: Generator[int|str, None, None]) -> None" in output


def test_async_generator():
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

    assert "def gen() -> AsyncGenerator[int, None]:" in output
    assert "def g(f: AsyncGenerator[int, None]) -> None" in output


@pytest.mark.parametrize('as_module', [False, True])
def test_send_generator(as_module):
    t = textwrap.dedent("""\
        def gen():
            sum = 0.0
            while True:
                value = yield sum
                if value is not None:
                    sum += value

        def f(g):
            return [
                g.send(10),
                g.send(5)
            ]

        g = gen()
        next(g) # prime generator
        print(f(g))
        """)

    Path("t.py").write_text(t)

    p = subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                       '--no-use-multiprocessing',
                       *(('-m', 't') if as_module else ('t.py',))],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert '[10.0, 15.0]' in str(p.stdout, 'utf-8')

    output = Path("t.py").read_text()

    assert "def gen() -> Generator[float, int, None]:" in output
    assert "def f(g: Generator[float, int, None]) -> list[float]" in output


@pytest.mark.parametrize('as_module', [False, True])
def test_send_async_generator(as_module):
    t = textwrap.dedent("""\
        import asyncio

        async def gen():
            sum = 0.0
            while True:
                value = yield sum
                if value is not None:
                    sum += value

        async def f(g):
            return [
                await g.asend(10),
                await g.asend(5)
            ]

        async def main():
            g = gen()
            await anext(g)
            print(await f(g))

        asyncio.run(main())
        """)

    Path("t.py").write_text(t)

    p = subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                       *(('-m', 't') if as_module else ('t.py',))],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert '[10.0, 15.0]' in str(p.stdout, 'utf-8')

    output = Path("t.py").read_text()

    assert "def gen() -> AsyncGenerator[float, int]:" in output
    assert "def f(g: AsyncGenerator[float, int]) -> list[float]" in output


def test_send_not_generator():
    t = textwrap.dedent("""\
        class C:
            def send(self, x):
                return str(x)

            def asend(self, x):
                return str(x)

        print(C().send(10))
        print(C().asend(10.0))
        """)

    Path("t.py").write_text(t)

    p = subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                       '--no-use-multiprocessing', 't.py'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert '10\n10.0\n' in str(p.stdout, 'utf-8')

    output = Path("t.py").read_text()

    assert "def send(self: Self, x: int) -> str:" in output
    assert "def asend(self: Self, x: float) -> str:" in output


def test_send_bound():
    t = textwrap.dedent("""\
        def gen():
            sum = 0.0
            while True:
                value = yield sum
                if value is not None:
                    sum += value

        def f(s):
            return [
                s(10),
                s(5)
            ]

        g = gen()
        next(g) # prime generator
        print(f(g.send))
        """)

    Path("t.py").write_text(t)

    p = subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                       '--no-use-multiprocessing', 't.py'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert '[10.0, 15.0]' in str(p.stdout, 'utf-8')

    output = Path("t.py").read_text()

    assert "def gen() -> Generator[float, int, None]:" in output
    # TODO the Callable here is our wrapper for the 'g.send' method... can we do better?
    assert "def f(s: Callable) -> list[float]" in output


def test_coroutine():
    Path("t.py").write_text(textwrap.dedent("""\
        import asyncio

        def foo():
            async def coro():
                return "did it"

            return coro()

        asyncio.run(foo())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo() -> Coroutine[None, None, str]:" in output


def test_generate_stubs():
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
        from typing import Self
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


def test_type_from_main():
    Path("m.py").write_text(textwrap.dedent("""\
        def f(x):
            return str(x)
        """
    ))

    Path("t.py").write_text(textwrap.dedent("""\
        import m

        class C:
            def __str__(self):
                return "hi!"

        m.f(C())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)
    output = Path("m.py").read_text()
    assert "def f(x: \"t.C\") -> str:" in output

    subprocess.run([sys.executable, '-m', 'mypy', 'm.py', 't.py'], check=True)


def test_module_type():
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


def test_function_type():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x: int) -> float:
            return x/2

        def bar(f, g, x):
            return f(x) + g(C(), x)

        def baz(h, x):
            return h(x)

        class C:
            def foo2(self: "C", x: int) -> float:
                return x*.5

        bar(foo, C.foo2, 1)
        baz(C().foo2, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def bar(f: Callable[[int], float], g: "Callable[[C, int], float]", x: int) -> float:' in output
    assert 'def baz(h: Callable[[int], float], x: int) -> float:' in output # bound method


def test_function_type_future_annotations():
    Path("t.py").write_text(textwrap.dedent("""\
        from __future__ import annotations

        def foo(x: int) -> float:
            return x/2

        def bar(f, g, x):
            return f(x) + g(C(), x)

        def baz(h, x):
            return h(x)

        class C:
            def foo2(self: "C", x: int) -> int:
                return x//2

        bar(foo, C.foo2, 1)
        baz(C().foo2, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def bar(f: Callable[[int], float], g: Callable[[C, int], int], x: int) -> float:" in output
    assert 'def baz(h: Callable[[int], int], x: int) -> int:' in output # bound method


def test_function_type_in_annotation():
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
    assert 'def baz(f: Callable[[FunctionType, Any], Any], g: Callable[[int], float], x: int) -> float:' in output


def test_discovered_function_type_in_args():
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


def test_discovered_function_type_in_return():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x):
            return x/2

        def bar(f):
            return f

        bar(foo)(1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo(x: int) -> float:" in output
    assert "def bar(f: Callable[[int], float]) -> Callable[[int], float]:" in output


def test_discovered_function_type_in_yield():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x):
            return x/2

        def bar():
            yield foo

        for a in bar(): a(1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo(x: int) -> float:" in output
    assert "def bar() -> Iterator[Callable[[int], float]]:" in output


@pytest.mark.parametrize('ignore_ann', [False, True])
def test_discovered_function_annotated(ignore_ann):
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x: int | float) -> float:
            return x/2

        def bar(f, x):
            return f(x)

        bar(foo, 1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing',
                    *(('--ignore-annotations',) if ignore_ann else()),
                    't.py'], check=True)

    output = Path("t.py").read_text()

    if ignore_ann:
        assert "def bar(f: Callable[[int], float], x: int) -> float:" in output
    else:
        assert "def bar(f: Callable[[int | float], float], x: int) -> float:" in output


def test_discovered_generator():
    Path("t.py").write_text(textwrap.dedent("""\
        def g(x):
            yield from range(x)

        def f(x):
            for _ in x:
                pass

        f(g(10))
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def f(x: Iterator[int]) -> None:" in output


def test_discovered_genexpr():
    Path("t.py").write_text(textwrap.dedent("""\
        def f(x):
            for _ in x:
                pass

        f((i for i in range(10)))
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def f(x: Iterator[int]) -> None:" in output


def test_discovered_genexpr_two_in_same_line():
    # TODO this is a bit risky: we identify the functions (and genexpr) by filename,
    # first code line and name.  These two genexpr have thus the same name!
    # We could add the first code column...
    Path("t.py").write_text(textwrap.dedent("""\
        def f(x):
            return sum(1 for _ in x)

        def g(x):
            return sum(1 for _ in x)

        f((i for i in range(10))) + g((s for s in ['a', 'b']))
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def f(x: Iterator[int]) -> int:" in output
    assert "def g(x: Iterator[str]) -> int:" in output


def test_module_list_not_lost_with_multiprocessing():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(t):
            pass

        from xml.dom.minidom import Element as E
        foo(E('foo'))
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(t: "xml.dom.minidom.Element") -> None:' in output

    assert 'import xml.dom.minidom\n' in output


def test_posonly_and_kwonly():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x, /, *, y):
            pass

        foo(10, y=.1)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: int, /, *, y: float) -> None:' in output


def test_varargs():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x, *args):
            pass

        foo(True, 10, 's', 0.5)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: bool, *args: float|int|str) -> None:' in output


def test_varargs_empty():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x, *args):
            pass

        foo(True)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: bool, *args: None) -> None:' in output


def test_kwargs():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x, **kwargs):
            pass

        foo(True, a=10, b='s', c=0.5)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: bool, **kwargs: float|int|str) -> None:' in output


def test_kwargs_empty():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x, **kwargs):
            pass

        foo(True)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: bool, **kwargs: None) -> None:' in output


def test_none_arg():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x):
            pass

        foo(None)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: None) -> None:' in output


def test_self():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(self):
            return self/2

        class C:
            def bar(self, x):
                class D:
                    def __init__(self):
                        pass

                D()
                return x/2

            class E:
                def baz(me):
                    return me

        foo(10)
        C().bar(1)
        C.E().baz()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(self: int) -> float:' in output
    assert 'def bar(self: Self, x: int) -> float:' in output
    assert 'def __init__(self: Self) -> None:' in output
    assert 'def baz(me: Self) -> Self:' in output


@pytest.mark.xfail(reason="Doesn't currently work")
def test_self_with_wrapped_method():
    Path("t.py").write_text(textwrap.dedent("""\
        import functools

        class C:
            @functools.cache
            def foo(self, x):
                return self

        C().foo(1)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(self: Self, x: int) -> Self:' in output


def test_rich_is_messed_up():
    # running rich's test suite leaves it unusable... simulate that situation.
    Path("t.py").write_text(textwrap.dedent("""\
        import sys
        import rich.progress

        def foo():
            rich.progress.Progress = None  # just something to break it

        foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', 't.py'], check=True)


@pytest.mark.parametrize('as_module', [False, True])
@pytest.mark.parametrize('use_mp', [False, True])
def test_nonzero_SystemExit(as_module, use_mp):
    Path("t.py").write_text(textwrap.dedent("""\
        raise SystemExit("something")
    """))

    p = subprocess.run([sys.executable, '-m', 'righttyper',
                        *(() if use_mp else ('--no-use-multiprocessing',)),
                        *(('-m', 't') if as_module else ('t.py',))],
                        check=False)
    assert p.returncode != 0


@pytest.mark.parametrize('as_module', [False, True])
@pytest.mark.parametrize('use_mp', [False, True])
def test_zero_SystemExit(as_module, use_mp):
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(x):
            return x

        foo(10)
        raise SystemExit()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    *(() if use_mp else ('--no-use-multiprocessing',)),
                    *(('-m', 't') if as_module else ('t.py',))],
                   check=True)

    assert "def foo(x: int) -> int:" in Path("t.py").read_text()


def test_arg_parsing(tmp_cwd):
    Path("t.py").write_text(textwrap.dedent("""\
        import json
        import sys

        with open("out.json", "w") as f:
            json.dump(sys.argv, f)
    """))

    def test_args(*args):
        import json
        subprocess.run([sys.executable, '-m', 'righttyper', *args], check=True)
        with open("out.json", "r") as f:
            return json.load(f)

    assert ['t.py'] == test_args('t.py')
    assert ['t.py', 'a', 'b'] == test_args('t.py', 'a', 'b')

    assert [str(tmp_cwd / "t.py")] == test_args('-m', 't')
    assert [str(tmp_cwd / "t.py"), 'a'] == test_args('-m', 't', 'a')

    assert ['t.py', 'a', '-m'] == test_args('t.py', '--', 'a', '-m')


def test_mocked_function():
    Path("t.py").write_text(textwrap.dedent("""\
        from unittest.mock import create_autospec

        class C:
            def m(self, x):
                return x*2

        def test_it():
            mocked = create_autospec(C)
            mocked.m.return_value = -1

            assert -1 == mocked.m(2)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--no-use-multiprocessing', '-m', 'pytest', 't.py'], check=True)


@pytest.mark.parametrize('as_module', [False, True])
def test_union_superclass(as_module):
    Path("t.py").write_text(textwrap.dedent("""\
        class A: pass
        class B(A): pass
        class C(A): pass

        def foo(x):
            pass

        foo(B())
        foo(C())
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', *(('-m', 't') if as_module else ('t.py',))],
                   check=True)

    assert "def foo(x: A) -> None:" in Path("t.py").read_text()


def test_sampling_overlaps():
    # While sampling, the function is started twice, with the first invocation outlasting
    # the second.  We'll get a START and YIELD events for the first invocation and then
    # a RETURN event for the second... if we don't leave the event enabled, we may not
    # see the first invocation's RETURN.
    t = textwrap.dedent("""\
        def gen(more: bool):
            if more:
                yield 0
            yield 1

        a = gen(True)
        b = gen(False)
        next(a)
        next(b)
        for _ in a:
            pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def gen(more: bool) -> Iterator[int]:" in output


def test_no_return():
    # A function for which we never see a RETURN: can we still type it?
    t = textwrap.dedent("""\
        def gen():
            while True:
                yield 0

        g = gen()
        next(g)
        next(g)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def gen() -> Iterator[int]:" in output


def test_generic_simple():
    t = textwrap.dedent(
        """\
        def add(a, b):
            return a + b
        add(1, 2)
        add("a", "b")
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert 'rt_T1 = TypeVar("rt_T1", int, str)' in output
    assert "def add(a: rt_T1, b: rt_T1) -> rt_T1" in output


def test_generic_name_conflict():
    t = textwrap.dedent("""\
        rt_T1 = None
        rt_T2 = None

        def add(a, b):
            return a + b

        add(1, 2)
        add("a", "b")
        """
    )

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert 'rt_T3 = TypeVar("rt_T3", int, str)' in output
    assert "def add(a: rt_T3, b: rt_T3) -> rt_T3" in output


def test_generic_yield():
    t = textwrap.dedent("""\
        def y(a):
            yield a
        for _ in y(1): pass
        for _ in y("a"): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert 'rt_T1 = TypeVar("rt_T1", int, str)' in output
    assert "def y(a: rt_T1) -> Iterator[rt_T1]" in output


def test_generic_yield_generator():
    t = textwrap.dedent("""\
        def y(a, b):
            yield a
            return b
        for _ in y(1, "a"): pass
        for _ in y("a", 1): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    print(output)
    assert 'rt_T1 = TypeVar("rt_T1", int, str)' in output
    assert 'rt_T2 = TypeVar("rt_T2", int, str)' in output
    assert "def y(a: rt_T1, b: rt_T2) -> Generator[rt_T1, None, rt_T2]" in output


def test_generic_typevar_location():
    t = textwrap.dedent("""\
        ...
        # comment and emptyline
        def add(a, b):
            return a + b
        add(1, 2)
        add("a", "b")
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    res = textwrap.dedent("""\
        rt_T1 = TypeVar("rt_T1", int, str)
        # comment and emptyline
        def add(a: rt_T1, b: rt_T1) -> rt_T1:
        """)

    assert res in output


def test_generic_and_defaults():
    t = textwrap.dedent("""\
        def f(a, b=None, c=None):
            pass

        f(10, 10, 5)
        f(10.0, 2, 10.0)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', '--inline-generics', 't.py'], check=True)
    output = Path("t.py").read_text()

    print(output)
    assert "def f[T1: (float, int)](a: T1, b: int|None=None, c: T1|None=None) -> None" in output


@pytest.mark.parametrize('superclass, expected', [
    ("list", "MyContainer[Never]"),
    ("set", "MyContainer[Never]"),
    ("dict", "MyContainer[Never, Never]"),
    ("KeysView", "KeysView[Never]"),
    ("ValuesView", "ValuesView[Never]"),
    ("ItemsView", "ItemsView[Never, Never]"),
    ("tuple", "tuple")
])
def test_custom_collection_len_error(superclass, expected):
    Path("t.py").write_text(textwrap.dedent(f"""\
        from collections.abc import *

        class MyContainer({superclass}):
            def __init__(self):
                super()

            def __len__(self):
                raise Exception("Oops, something went wrong!")


        def foo(bar):
            pass


        my_object = MyContainer()
        foo(my_object)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '-m', 't'], check=True)

    assert f"def foo(bar: {expected}) -> None" in Path("t.py").read_text()


@pytest.mark.parametrize('superclass, expected', [
    ("list", "MyContainer[Never]"),
    ("set", "MyContainer[Never]"),
    ("dict", "MyContainer[Never, Never]"),
    ("KeysView", "KeysView[Never]"),
    ("ValuesView", "ValuesView[Never]"),
    ("ItemsView", "ItemsView[Never, Never]"),
    ("tuple", "tuple")
])
def test_custom_collection_sample_error(superclass, expected):
    Path("t.py").write_text(textwrap.dedent(f"""\
        from collections.abc import *

        class MyContainer({superclass}):
            def __init__(self):
                super()

            def __len__(self):
                return 1

            def __getitem__(self, key):
                raise Exception("Oops, something went wrong!")

            def __contains__(self, key):
                raise Exception("Oops, something went wrong!")

            def __iter__(self):
                raise Exception("Oops, something went wrong!")


        def foo(bar):
            pass


        my_object = MyContainer()
        foo(my_object)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '-m', 't'], check=True)

    assert f"def foo(bar: {expected}) -> None" in Path("t.py").read_text()


def test_class_properties():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x = None

            @property
            def x(self):
                return str(self._x)

            @x.setter
            def x(self, value):
                self._x = value

            @x.deleter
            def x(self):
                del self._x

        c = C()
        c.x = 10
        y = c.x
        del c.x
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '-m', 't'], check=True)

    output = Path("t.py").read_text()
    print(output)

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def x(self: Self) -> str:" in output                # getter
    assert "def x(self: Self, value: int) -> None:" in output   # setter
    assert "def x(self: Self) -> None:" in output               # deleter


def test_class_properties_no_setter():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x = 10

            @property
            def x(self):
                return str(self._x)

            @x.deleter
            def x(self):
                del self._x

        c = C()
        y = c.x
        del c.x
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '-m', 't'], check=True)

    output = Path("t.py").read_text()

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def x(self: Self) -> str:" in output                # getter
    assert "def x(self: Self) -> None:" in output               # deleter


def test_class_properties_inner_functions():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x = None

            @property
            def x(self):
                def foo():
                    return str(self._x)
                return foo()

            @x.setter
            def x(self, value):
                def foo(v):
                    def bar():
                        pass
                    bar()
                    return int(v)
                self._x = foo(value)

        c = C()
        c.x = 10.0
        y = c.x
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '-m', 't'], check=True)

    output = Path("t.py").read_text()

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def foo() -> str:" in output            # getter's
    assert "def foo(v: float) -> int:" in output    # setter's

    # check for inner function's inner function
    assert "def bar() -> None:" in output


def test_self_simple():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return self

        o = A()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> Self:" in Path("t.py").read_text()


def test_self_wrapped_method():
    Path("t.py").write_text(textwrap.dedent("""\
        import functools

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)

            return wrapper

        class A:
            @decorator
            def foo(self):
                return self

        o = A()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> Self:" in Path("t.py").read_text()


def test_self_bound_method():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self, x):
                return self

        f = A().foo
        f(10)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self, x: int) -> Self:" in Path("t.py").read_text()


def test_self_inherited_method():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return self

        class B(A):
            pass

        o = B()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> Self:" in Path("t.py").read_text()


def test_self_inherited_method_called_indirectly():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return self

        class B(A):
            def bar(self):
                self.foo()

        o = B()
        o.bar()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> Self:" in Path("t.py").read_text()


def test_self_inherited_method_returns_non_self():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return A()

        class B(A):
            pass

        o = B()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> \"A\":" in Path("t.py").read_text()


def test_self_classmethod():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            @classmethod
            def static_initializer(cls):
                return cls()

        o = A.static_initializer()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def static_initializer(cls: type[Self]) -> Self:" in Path("t.py").read_text()


def test_self_inherited_classmethod():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            @classmethod
            def static_initializer(cls):
                return cls()

        class B(A):
            pass

        o = B.static_initializer()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def static_initializer(cls: type[Self]) -> Self:" in Path("t.py").read_text()


def test_self_within_other_types():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return [self, self]

        o = A()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', '--no-use-multiprocessing', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> list[Self]" in Path("t.py").read_text()


def test_self_yield_generator():
    t = textwrap.dedent("""\
        class A:
            def foo(self):
                yield self
                return self

        for _ in A().foo(): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    print(output)
    assert "def foo(self: Self) -> Generator[Self, None, Self]" in output


def test_self_subtyping():
    t = textwrap.dedent("""\
        class NumberAdd:
            def __init__(self, value: float):
                self.value = value

            def operation(self, rhs):
                return self.__class__(self.value + rhs.value)

        class IntegerAdd(NumberAdd):
            def __init__(self, value: int):
                if value != round(value):
                    raise ValueError()
                super().__init__(value)

        a = NumberAdd(0.5)
        b = IntegerAdd(1)

        a.operation(b)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    # IntegerAdd IS-A NumberAdd, the enclosed class; so the argument should be 'Self'
    assert "def operation(self: Self, rhs: Self) -> Self:" in output


def test_self_subtyping_reversed():
    t = textwrap.dedent("""\
        class NumberAdd:
            def __init__(self, value: float):
                self.value = value

            def operation(self, rhs):
                return self.__class__(self.value + rhs.value)

        class IntegerAdd(NumberAdd):
            def __init__(self, value: int):
                super().__init__(round(value))

        a = NumberAdd(0.5)
        b = IntegerAdd(1)

        b.operation(a)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    # The argument isn't Self as (NumberAdd IS-A IntegerAdd) doesn't hold
    assert "def operation(self: Self, rhs: \"NumberAdd\") -> Self:" in output


def test_returns_or_yields_generator():
    t = textwrap.dedent("""\
        def test(a):
            if a < 5:
                return "too small :("
            else:
                for i in range(a):
                    yield a

        for _ in test(3): pass
        for _ in test(10): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def test(a: int) -> Generator[int|None, None, str|None]" in output


def test_generators_merge_into_iterator():
    t = textwrap.dedent("""\
        def test(a):
            if a < 5:
                yield "too small"
            else:
                yield a

        for _ in test(3): pass
        for _ in test(10): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def test(a: int) -> Iterator[int|str]" in output


@pytest.mark.xfail(reason="Temporarily disabled: RandomDict causes issues with rich")
def test_random_dict():
    t = textwrap.dedent("""\
        def f(x):
            return len(x)

        d = {'a': {'b': 2}}
        f(d)

        from righttyper.random_dict import RandomDict
        assert isinstance(d, RandomDict)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-use-multiprocessing', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(x: dict[str, dict[str, int]]) -> int" in output


def test_instrument_test():
    t = textwrap.dedent("""\
        def f():
            x = yield 42
            yield x

        def test_foo():
            g = f()
            next(g)
            r = g.send(10)
            assert r == 10
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                   '-m' 'pytest', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f() -> Generator[int, int, None]" in output
