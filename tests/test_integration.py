import textwrap
import subprocess
import sys
from pathlib import Path
import pytest
import importlib.util
import re
import libcst as cst

from test_transformer import get_function as t_get_function
from functools import partial
get_function = partial(t_get_function, body=False)


@pytest.fixture(scope='function')
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def print_file(file: Path) -> None:
    for lineno, line in enumerate(file.read_text().splitlines(), start=1):
        print(f"{lineno:3}: {line}")


@pytest.fixture(scope='function', autouse=True)
def runmypy(tmp_cwd, request):
    yield
    if (
        not request.node._report.passed
        or 'dont_run_mypy' in request.keywords
        or request.config.getoption("--no-mypy")
    ):
        return

    # if we are specifying a Python version in the test, have mypy check for that as well
    python_version = (
        ('--python-version', request.node.callspec.params.get('python_version'))
        if hasattr(request.node, 'callspec')
        and 'python_version' in request.node.callspec.params
        else ()
    )
    from mypy import api
    result = api.run([*python_version, '.'])
    if result[2]:
        print(result[0])
        filename = result[0].split(':')[0]
        print_file(Path(filename))
        pytest.fail("see mypy errors")


@pytest.mark.parametrize("init, expected", [
    ["iter(b'0')", "Iterator[int]"],
    ["iter(bytearray(b'0'))", "Iterator[int]"],
    ["iter({'a': 0})", "Iterator[str]"],
    ["iter({'a': 0}.values())", "Iterator[int]"],
    ["iter({'a': 0}.items())", "Iterator[tuple[str, int]]"],
    ["iter([0, 1])", "Iterator[int]"],
    ["iter(reversed([0, 1]))", "Iterator[int]"],
    ["iter(range(1))", "Iterator[int]"],
    ["iter(range(1 << 1000))", "Iterator[int]"],
    ["iter({'a'})", "Iterator[str]"],
    ["iter('ab')", "Iterator[str]"],
    ["iter(('a', 'b'))", "Iterator[str]"],
    ["iter(tuple(c for c in ('a', 'b')))", "Iterator[str]"],
    ["iter(zip([0], ('a',)))", "Iterator[tuple[int, str]]"],
    ["zip([0])", "Iterator[tuple[int]]"],
    ["enumerate(('a', 'b'))", "enumerate[str]"],
    ["iter(zip([0], (c for c in ('a',))))", "Iterator[tuple[int, str]]"],
    ["enumerate(c for c in ('a', 'b') if c)", "enumerate[str]"],
])
def test_builtin_iterator(init, expected):
    t = textwrap.dedent(f"""\
        def f():
            return {init}

        next(f())
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent(f"""\
        def f() -> {expected}: ...
    """)


@pytest.mark.parametrize("init, expected", [
    ["zip([0], ())", "Iterator[tuple[int, Never]]"],
    ["enumerate(())", "enumerate[Never]"],
])
def test_builtin_iterator_of_empty(init, expected):
    t = textwrap.dedent(f"""\
        def f():
            return {init}

        f()     # don't call next(), as it'd yield StopIteration
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent(f"""\
        def f() -> {expected}: ...
    """)


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='missing module')
def test_numpy_iterator():
    t = textwrap.dedent("""\
        import numpy as np

        def f():
            return enumerate(np.arange(1).flat)

        f()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    # TODO this could follow '.base' and retrieve the dtype
    assert get_function(code, 'f') == textwrap.dedent(f"""\
        def f() -> enumerate[Any]: ...
    """)


def test_getitem_iterator():
    t = textwrap.dedent(f"""\
        class X:
            def __getitem__(self, n):
                if n < 10:
                    return (n & 2) == 0
                raise IndexError()

        def f(it):
            next(it)

        f(iter(X()))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(it: Iterator[bool]) -> None: ...
    """)


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='missing module')
def test_getitem_iterator_numpy():
    t = textwrap.dedent(f"""\
        import numpy as np

        def f(it):
            return next(it)

        f(iter(np.arange(10, dtype=np.int64)))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(it: Iterator[np.int64]) -> np.int64: ...
    """)


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='missing module')
def test_getitem_iterator_numpy_empty():
    t = textwrap.dedent(f"""\
        import numpy as np

        def f(it):
            pass

        f(iter(np.arange(0, dtype=np.int64)))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(it: Iterator) -> None: ...
    """)


def test_getitem_iterator_from_annotation():
    t = textwrap.dedent(f"""\
        class X:
            def __getitem__(self, n) -> float:
                if n < 10:
                    return n
                raise IndexError()

        def f(it):
            pass

        f(iter(X()))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(it: Iterator[float]) -> None: ...
    """)


def test_custom_iterator():
    t = textwrap.dedent(f"""\
        class X:
            def __iter__(self):
                return self

            def __next__(self):
                return 42

        def f(it):
            next(it)

        f(iter(X()))
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(it: X) -> None: ...
    """)

    assert get_function(code, 'X.__iter__') == textwrap.dedent("""\
        def __iter__(self: Self) -> Self: ...
    """)

    assert get_function(code, 'X.__next__') == textwrap.dedent("""\
        def __next__(self: Self) -> int: ...
    """)


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
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "import slice" not in output
    assert "def func(s: slice) -> range" in output

    assert "import type" not in output
    assert "def func2(t: type[str]) -> str" in output

    assert "import super" not in output
    assert "def func3(t: super) -> None" in output


def test_callable_from_annotations():
    t = textwrap.dedent("""\
        def f(x: int | float, y) -> float:
            return x/2

        def g():
            return f

        f(1.0, None)
        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable[[int|float, Any], float]:" in output
    # TODO is it ok for 'y' to be typed as observed, while the Callable uses 'Any' ?
    assert "def f(x: int | float, y: None) -> float:" in output


def test_callable_from_annotations_typing_special():
    t = textwrap.dedent("""\
        import typing

        class C:
            def f(self, x: int, y) -> typing.NoReturn:
                while True:
                    pass

            def g(self):
                return self.f

        C().g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'C.g') == textwrap.dedent("""\
        def g(self: Self) -> Callable[[int, Any], NoReturn]: ...
    """)


def test_callable_from_annotation_generic_alias():
    t = textwrap.dedent("""\
        def f() -> list[int]:   # list[int] is a GenericAlias
            return [1,2,3]

        def g():
            return f

        f()
        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable[[], list[int]]:" in output


@pytest.mark.dont_run_mypy # fails because of SomethingUnknown
def test_callable_annotation_errors():
    t = textwrap.dedent("""\
        from __future__ import annotations

        def f(x: SomethingUnknown) -> None:
            pass

        def g():
            return f

        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable:" in output


@pytest.mark.dont_run_mypy # fails because of SomethingUnknown
def test_generator_annotation_errors():
    t = textwrap.dedent("""\
        from __future__ import annotations
        from collections.abc import Generator

        def f() -> Generator[SomethingUnknown, None, None]:
            yield 1

        def g(x):
            pass

        g(f())
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g(x: Generator) -> None" in output


def test_callable_from_annotation_none_return():
    t = textwrap.dedent("""\
        def f() -> None:
            pass

        def g():
            return f

        f()
        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
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
                    't.py'], check=True)
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
                    't.py'], check=True)
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

        def f() -> np.ndarray[Any, np.dtype[bf16]]: ... # type: ignore[empty-body]

        def g():
            return f

        g()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g() -> Callable[[], np.ndarray[Any, np.dtype[bf16]]]:" in output


@pytest.mark.dont_run_mypy # it lacks definitions for checking
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
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    f = get_function(code, 'MyArray.__array_function__')
    assert re.search(r'func: "numpy.[\w\.]*_ArrayFunctionDispatcher"', str(f))


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
                    '--infer-shapes', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert 'def f(x: "jaxtyping.Int64[np.ndarray, \\"2 1\\"]") ' +\
           '-> "jaxtyping.Int64[np.ndarray, \\"2 1\\"]"' in output


def test_default_arg():
    t = textwrap.dedent("""\
        def func(n=None):
            return n+1 if n else 0

        def func2(n=5):
            return n+1

        func(1)
        func2(1.0)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def func(n: int|None=None) -> int" in output

    assert "def func2(n: float|int=5) -> float" in output


def test_default_in_private_method():
    t = textwrap.dedent("""\
        class C:
            def __f(self, x=None):
                return x+1 if x else 0

            def f(self, x):
                return self.__f(x)

        C().f(1)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def __f(self: Self, x: int|None=None) -> int" in output


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
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g(y: int) -> int" in output


def test_method():
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
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def f(self: Self, n: int) -> int" in output
    assert "\nimport Self" not in output

    assert "def h(self: Self, x: int) -> float" in output


def test_method_imported():
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
                    't.py'], check=True)
    output = Path("m.py").read_text()

    assert "\nimport Self" not in output

    assert "def f(self: Self, n: int) -> int" in output
    assert "import C" not in output

    assert "def g(x: int) -> float" in output
    assert "def h(self: Self, x: int) -> float" in output
    assert "import gC" not in output


def test_method_overriding():
    t = textwrap.dedent("""\
        class A:
            def foo(self, x):
                return x/2

        class B(A):
            def foo(self, x):
                return int(x//2)

        A().foo(1.0)
        B().foo(10)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'A.foo') == textwrap.dedent("""\
        def foo(self: Self, x: float) -> float: ...
    """)

    # contravariant for parameters, covariant for return value;
    # so that while 'x' must not be 'int', but the return value may be 'int'
    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: float|int) -> int: ...
    """)


def test_method_overriding_annotated():
    t = textwrap.dedent("""\
        from typing import Self, List

        class A:
            def foo(self: Self, x: List[int]):
                return len(x)

        class B(A):
            def foo(self, x):
                return len(x)

        B().foo([1.0])
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: list[float]|list[int]) -> int: ...
    """)


def test_method_overriding_annotated_with_literal():
    t = textwrap.dedent("""\
        from typing import Self, Literal

        class A:
            def foo(self: Self, x: Literal[10, 20]):
                return x // 10

        class B(A):
            def foo(self, x):
                return int(x) // 10 + 1

        B().foo(1.0)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: float|int) -> int: ...
    """)


def test_method_overriding_init_irrelevant():
    t = textwrap.dedent("""\
        class A:
            def __init__(self, x):
                pass

        class B(A):
            def __init__(self):
                super().__init__('x')
                pass

        A(1)
        B()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'A.__init__') == textwrap.dedent("""\
       def __init__(self: Self, x: int|str) -> None: ...
    """)

    assert get_function(code, 'B.__init__') == textwrap.dedent("""\
        def __init__(self: Self) -> None: ...
    """)


def test_method_overriding_new_irrelevant():
    t = textwrap.dedent("""\
        class A:
            def __new__(cls, x: int):
                return super().__new__(cls)

        class B(A):
            def __new__(cls, x):
                return super().__new__(cls, 0)

        B("")
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'B.__new__') == textwrap.dedent("""\
        def __new__(cls: type[Self], x: str) -> Self: ...
    """)


def test_method_overriding_classmethod():
    t = textwrap.dedent("""\
        class A:
            @classmethod
            def foo(cls, x):
                pass

        class B(A):
            @classmethod
            def foo(cls, x):
                pass

        A.foo('')
        B.foo(1)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'A.foo') == textwrap.dedent("""\
        @classmethod
        def foo(cls: type[Self], x: str) -> None: ...
    """)

    # Somewhat unexpectedly, @classmethod matters for LSP (I think because they
    # are also available in subclasses).  At least according to mypy 1.15.0.
    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        @classmethod
        def foo(cls: type[Self], x: int|str) -> None: ...
    """)


def test_method_overriding_private():
    t = textwrap.dedent("""\
        class A:
            def __foo(self, x):
                return x/2

            def _bar(self, x):
                return x/2

        class B(A):
            def __foo(self, x):
                return int(x//2)

            def _bar(self, x):
                return int(x//2)

        A()._A__foo(1.0)    # type: ignore[attr-defined]
        A()._bar(1.0)
        B()._B__foo(10)     # type: ignore[attr-defined]
        B()._bar(10)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'A.__foo') == textwrap.dedent("""\
        def __foo(self: Self, x: float) -> float: ...
    """)

    assert get_function(code, 'A._bar') == textwrap.dedent("""\
        def _bar(self: Self, x: float) -> float: ...
    """)

    assert get_function(code, 'B.__foo') == textwrap.dedent("""\
        def __foo(self: Self, x: int) -> int: ...
    """)

    assert get_function(code, 'B._bar') == textwrap.dedent("""\
        def _bar(self: Self, x: float|int) -> int: ...
    """)


def test_method_overriding_method_called_indirectly():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self, x):
                return self

        class B(A):
            def bar(self, x):
                self.foo(x)

        o = B()
        o.bar(1)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', 't.py'],
                   check=True)

    assert "def foo(self: Self, x: int) -> Self:" in Path("t.py").read_text()


def test_method_overriding_inherited():
    Path("t.py").write_text(textwrap.dedent("""\
        from typing import Self

        class A:
            def foo(self: Self, x: float):
                return self

        class B(A):
            def foo(self, x):
                return self

        class C(B):
            def bar(self, x):
                self.foo(x)

        o = C()
        o.bar(1)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', 't.py'],
                   check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: float|int) -> Self: ...
    """)


def test_method_overriding_arg_names_change():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def foo(self, a: float, b: float, *, c: str) -> tuple[float, str]:
                return (a/b, c)

        class D(C):
            def foo(self, x, y, *, d=None, c) -> tuple[float, str]:
                return (0.0, '')


        o = D()
        o.foo(1, 2.0, d=4, c='*')
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', 't.py'],
                   check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'D.foo') == textwrap.dedent("""\
        def foo(self: Self, x: float|int, y: float, *, d: int|None=None, c: str) -> tuple[float, str]: ...
    """)


def test_method_overriding_annotation():
    t = textwrap.dedent("""\
        class A:
            def foo(self, x: float) -> float:
                return x/2

        class B(A):
            def foo(self, x):
                return int(x//2)

        B().foo(10)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    # contravariant for parameters, covariant for return value;
    # so that while 'x' must not be 'int', but the return value may be 'int'
    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: float|int) -> int: ...
    """)


def test_method_overriding_annotation_ignored():
    t = textwrap.dedent("""\
        class A:
            def foo(self, x: float) -> float:
                return x/2

        class B(A):
            def foo(self, x):
                return int(x//2)

        A().foo(10)
        B().foo(10)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--ignore-annotations', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: int) -> int: ...
    """)


@pytest.mark.dont_run_mypy # fails because of SomethingUnknown
def test_method_overriding_annotation_errors():
    t = textwrap.dedent("""\
        from __future__ import annotations

        class A:
            def foo(self, x: SomethingUnknown) -> float:
                return x/2

        class B(A):
            def foo(self, x):
                return int(x//2)

        B().foo(10)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, x: int) -> int: ...
    """)


def test_method_overriding_typeshed():
    t = textwrap.dedent("""\
        class C:
            def __eq__(self, other):
                if not isinstance(other, C):
                    return False
                return self is other

        C() == C()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'C.__eq__') == textwrap.dedent("""\
        def __eq__(self: Self, other: object|Self) -> bool: ...
    """)

    assert "\nimport Self" not in output


def test_method_overriding_inherited_typeshed():
    Path("t.py").write_text(textwrap.dedent("""\
        class Comparable:
            def __eq__(self, other):
                return False

        class C(Comparable):
            pass

        C() == C()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', 't.py'],
                   check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'Comparable.__eq__') == textwrap.dedent("""\
        def __eq__(self: Self, other: object|Self) -> bool: ...
    """)


@pytest.mark.dont_run_mypy  # this results in incompatible signatures... TODO could we resolve it?
def test_method_overriding_different_signature():
    t = textwrap.dedent("""\
        class A:
            def foo(self, x):
                pass

        class B(A):
            def foo(self, y, z):
                pass

        A().foo(1)
        B().foo(1.0, 2)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'A.foo') == textwrap.dedent("""\
       def foo(self: Self, x: int) -> None: ...
    """)

    assert get_function(code, 'B.foo') == textwrap.dedent("""\
        def foo(self: Self, y: float, z: int) -> None: ...
    """)


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
                    't.py'], check=True)
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
                    '-m', 'pytest', '-s', 'tests'], check=True)
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
                    't.py'], check=True)
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
                    't.py'], check=True)
    output = Path("t.py").read_text()

    # that local class name is "f.<locals>.fC"; this yields a CST ParserSyntaxError
    assert "import fC" not in output
    assert "def f():" in output # FIXME what is a good way to express the return type?
    assert "def g(x) -> None:" in output # FIXME what is a good way to express the type?


def test_default_inner_function():
    t = textwrap.dedent("""\
        def f(x):
            def g(y='0'):
                return int(y)+1

            return g(x)

        f(1)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def g(y: int|str='0') -> int" in output


def test_default_method():
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
                    't.py'], check=True)
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
                    't.py'], check=True)
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
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def gen() -> Generator[int, None, str]:" in output
    assert "def g(f: Generator[int, None, str]) -> None" in output


def test_generator_from_annotation():
    t = textwrap.dedent("""\
        from collections.abc import Generator

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

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    # we know it's from the annotation because we never observed an 'int' yield
    assert (
        get_function(code, 'g') == textwrap.dedent("""\
            def g(f: Generator[int|str, None, None]) -> None: ...
        """)
        or 
        get_function(code, 'g') == textwrap.dedent("""\
            def g(f: Iterator[int|str]) -> None: ...
        """)
    )


def test_generator_ignore_annotation():
    t = textwrap.dedent("""\
        from collections.abc import Generator

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
                    '--ignore-annotations', 't.py'], check=True)
    output = Path("t.py").read_text()

    # If from annotation, it'll include an 'int' yield
    assert "def g(f: Iterator[str]) -> None" in output


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
                    't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def gen() -> AsyncGenerator[int, None]:" in output
    assert "def g(f: AsyncGenerator[int, None]) -> None" in output


def test_generator_with_self():
    t = textwrap.dedent("""\
        class C:
            def f(self):
                yield self

        def f(g):
            for _ in g:
                pass

        f(C().f())
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(self: Self) -> Iterator[Self]" in output
    assert "def f(g: Iterator[C]) -> None" in output


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
                return float(x)

            def asend(self, x):
                return int(x)

        print([
            C().send(10),
            C().asend(10.0)
        ])
        """)

    Path("t.py").write_text(t)

    p = subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                       't.py'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    assert "[10.0, 10]" in str(p.stdout, 'utf-8')

    output = Path("t.py").read_text()

    assert "def send(self: Self, x: int) -> float:" in output
    assert "def asend(self: Self, x: float) -> int:" in output


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
                       't.py'],
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
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def foo() -> Coroutine[None, None, str]:" in output


def test_coroutine_with_self():
    Path("t.py").write_text(textwrap.dedent("""\
        import asyncio

        class C:
            async def coro(self):
                return self

        def f(g):
            asyncio.run(g)

        f(C().coro())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def coro(self: Self) -> Self:" in output
    assert "def f(g: Coroutine[None, None, C]) -> None:" in output


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
                    't.py'], check=True)

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
                    't.py'], check=True)
    output = Path("m.py").read_text()
    assert "def f(x: \"t.C\") -> str:" in output


def test_module_type():
    Path("t.py").write_text(textwrap.dedent("""\
        import sys

        def foo(m):
            pass

        foo(sys.modules['__main__'])
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

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
                    't.py'], check=True)

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
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def bar(f: Callable[[int], float], g: Callable[[C, int], int], x: int) -> float:" in output
    assert 'def baz(h: Callable[[int], int], x: int) -> int:' in output # bound method


# TODO this leads to an error: FunctionType IS-A Callable, so typing baz's g as
# a Callable is too general. Should we be narrowing baz's g to FunctionType ?
@pytest.mark.dont_run_mypy
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
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def bar(g: FunctionType, x: int) -> float:' in output
    assert 'def baz(f: Callable[[FunctionType, Any], Any], g: Callable[[int], float], x: int) -> float:' in output


def test_callable_varargs():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(*args):
            return args[0]/2

        def bar(f):
            f(1, 2)

        bar(foo)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(*args: int) -> float:' in output
    assert 'def bar(f: Callable[..., float]) -> None:' in output
    # or VarArg(int) from mypy_extensions


def test_callable_kwargs():
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(**kwargs):
            return kwargs['a']/2

        def bar(f):
            f(a=1, b=2)

        bar(foo)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(**kwargs: int) -> float:' in output
    assert 'def bar(f: Callable[..., float]) -> None:' in output
    # or KwArg(int) from mypy_extensions, or Unpack + TypedDict


def test_callable_with_self():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def f(self):
                pass

        def g(f):
            C().f()

        g(C.f)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def f(self: Self) -> None:' in output
    assert 'def g(f: Callable[[C], None]) -> None:' in output


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
                    't.py'], check=True)

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
                    't.py'], check=True)

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
                    't.py'], check=True)

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
        assert "def bar(f: Callable[[int|float], float], x: int) -> float:" in output


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
                    't.py'], check=True)

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
                    't.py'], check=True)

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
                    't.py'], check=True)

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


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self(python_version):
    Path("t.py").write_text(textwrap.dedent("""\
        def foo(self):
            return self/2

        class C:
            def bar(self, x):
                class D:
                    def __private(moi): ...

                    def __init__(self):
                        self.__private()

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
                    f'--python-version={python_version}', 't.py'], check=True)

    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    if python_version == '3.10':
        assert get_function(code, 'foo') == textwrap.dedent("""\
            def foo(self: int) -> float: ...
        """)
        assert get_function(code, 'C.bar') == textwrap.dedent("""\
            def bar(self: "C", x: int) -> float: ...
        """)
        # TODO we're not annotating because of <locals> in the name, but we could annotate "D"
        assert get_function(code, 'C.bar.<locals>.D.__private') == textwrap.dedent("""\
            def __private(moi) -> None: ...
        """)
        # TODO we're not annotating because of <locals> in the name, but we could annotate "D"
        assert get_function(code, 'C.bar.<locals>.D.__init__') == textwrap.dedent("""\
            def __init__(self) -> None: ...
        """)
        assert get_function(code, 'C.E.baz') == textwrap.dedent("""\
            def baz(me: "C.E") -> "C.E": ...
        """)
    else:
        assert get_function(code, 'foo') == textwrap.dedent("""\
            def foo(self: int) -> float: ...
        """)
        assert get_function(code, 'C.bar') == textwrap.dedent("""\
            def bar(self: Self, x: int) -> float: ...
        """)
        assert get_function(code, 'C.bar.<locals>.D.__private') == textwrap.dedent("""\
            def __private(moi: Self) -> None: ...
        """)
        assert get_function(code, 'C.bar.<locals>.D.__init__') == textwrap.dedent("""\
            def __init__(self: Self) -> None: ...
        """)
        assert get_function(code, 'C.E.baz') == textwrap.dedent("""\
            def baz(me: Self) -> Self: ...
        """)


def test_cached_function():
    Path("t.py").write_text(textwrap.dedent("""\
        import functools

        @functools.cache
        def foo(x=None):
            return x/2 if x else 0

        foo(1)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(x: int|None=None) -> float:' in output


def test_self_with_cached_method():
    Path("t.py").write_text(textwrap.dedent("""\
        import functools

        class C:
            @functools.cache
            def foo(self, x):
                return self

        C().foo(1)
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def foo(self: Self, x: int) -> Self:' in output


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self_in_hierarchy(python_version):
    Path("t.py").write_text(textwrap.dedent(f"""\
        class A:
            def f(self):
                return self

        class B(A):
            def g(self):
                ...

        class C(A):
            def h(self):
                ...

        A().f()
        {"B().f().g()" if python_version != "3.10" else "B().f(); B().g()"}
        {"C().f().h()" if python_version != "3.10" else "C().f(); C().h()"}
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', f'--python-version={python_version}', 't.py'], check=True)

    output = Path("t.py").read_text()
    print(output)
    code = cst.parse_module(output)

    if python_version == '3.10':
        assert get_function(code, 'A.f') == textwrap.dedent("""\
            def f(self: "A") -> "A": ...
        """)

        assert get_function(code, 'B.g') == textwrap.dedent("""\
            def g(self: "B") -> None: ...
        """)

        assert get_function(code, 'C.h') == textwrap.dedent("""\
            def h(self: "C") -> None: ...
        """)
    else:
        assert get_function(code, 'A.f') == textwrap.dedent("""\
            def f(self: Self) -> Self: ...
        """)

        assert get_function(code, 'B.g') == textwrap.dedent("""\
            def g(self: Self) -> None: ...
        """)

        assert get_function(code, 'C.h') == textwrap.dedent("""\
            def h(self: Self) -> None: ...
        """)


@pytest.mark.dont_run_mypy  # unnecessary for this test
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
                    't.py'], check=True)


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

    subprocess.run([sys.executable, '-m', 'righttyper', '-m', 'pytest', 't.py'], check=True)


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
    output = Path("t.py").read_text()
    print(output)

    assert "def foo(x: A) -> None:" in output


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
                    '--sampling', 't.py'], check=True)
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
                    '--sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert "def gen() -> Iterator[int]:" in output


@pytest.mark.parametrize("python_version", ["3.9", "3.11", "3.12"])
def test_generic_simple(python_version):
    t = textwrap.dedent(
        """\
        def add(a, b):
            return a + b
        add(1, 2)
        add("a", "b")
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    f'--python-version={python_version}', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    if python_version != "3.12":
        assert re.search('from typing import.*TypeVar', output)
        assert 'rt_T1 = TypeVar("rt_T1", int, str)' in output
        assert "def add(a: rt_T1, b: rt_T1) -> rt_T1" in output
    else:
        assert "def add[T1: (int, str)](a: T1, b: T1) -> T1" in output


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
                    '--python-version=3.11', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert 'rt_T3 = TypeVar("rt_T3", int, str)' in output
    assert "def add(a: rt_T3, b: rt_T3) -> rt_T3" in output


@pytest.mark.parametrize("python_version", ["3.11", "3.12"])
def test_generic_yield(python_version):
    t = textwrap.dedent("""\
        from typing import Any

        def y(a):
            yield a

        _: Any
        for _ in y(1): pass
        for _ in y("a"): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    f'--python-version={python_version}', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    if python_version == '3.11':
        assert 'rt_T1 = TypeVar("rt_T1", int, str)' in output
        assert "def y(a: rt_T1) -> Iterator[rt_T1]" in output
    else:
        assert "def y[T1: (int, str)](a: T1) -> Iterator[T1]" in output


@pytest.mark.parametrize("python_version", ["3.11", "3.12"])
def test_generic_yield_generator(python_version):
    t = textwrap.dedent("""\
        from typing import Any

        def y(a, b):
            yield a
            return b

        _: Any
        for _ in y(1, "a"): pass
        for _ in y("a", 1): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    f'--python-version={python_version}', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    if python_version == '3.11':
        assert re.search('from typing import.*TypeVar', output)
        assert 'rt_T1 = TypeVar("rt_T1", int, str)' in output
        assert 'rt_T2 = TypeVar("rt_T2", int, str)' in output
        assert "def y(a: rt_T1, b: rt_T2) -> Generator[rt_T1, None, rt_T2]" in output
    else:
        assert "def y[T1: (int, str), T2: (int, str)](a: T1, b: T2) -> Generator[T1, None, T2]" in output


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
                    '--python-version=3.11', '--no-sampling', 't.py'], check=True)
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
                    '--no-sampling', '--python-version=3.12', 't.py'], check=True)
    output = Path("t.py").read_text()

    assert not re.search('from typing import.*TypeVar', output)
    assert "def f[T1: (float, int)](a: T1, b: int|None=None, c: T1|None=None) -> None" in output


def test_inline_generics_no_variables():
    t = textwrap.dedent("""\
        def f(x):
            pass

        f([10])
        f(['10'])
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(x: list[int|str]) -> None" in output


@pytest.mark.parametrize('superclass', [
    'list', 'set', 'dict', 'tuple', 'KeysView', 'ValuesView', 'ItemsView'
])
def test_custom_collection_typing(superclass):
    Path("t.py").write_text(textwrap.dedent(f"""\
        from collections.abc import *

        class MyContainer({superclass}):
            def __init__(self):
                super()

        def foo(x): pass

        foo(MyContainer())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)
    output = Path("t.py").read_text()
    assert "def foo(x: MyContainer) -> None:" in output

@pytest.mark.parametrize('init, expected', [
    ['[1,2,3]', 'list[int]'],
    ['{"a", "b"}', 'set[str]'],
    ['frozenset({"a", "b"})', 'frozenset[str]'],
    ['{"a": 1, "b": 2}', 'dict[str, int]'],
    ['defaultdict(int, {"a": 1, "b": 2})', 'defaultdict[str, int]'],
    ['OrderedDict({"a": 1, "b": 2})', 'OrderedDict[str, int]'],
    ['ChainMap({"a": 1}, {"b": 2})', 'ChainMap[str, int]'],
    ['Counter(["foo", "bar"])', 'Counter[str]'],
    ['deque([1,2,3])', 'deque[int]'],
    ['RandomDict({"a": 1, "b": 2})', 'dict[str, int]'],
])
def test_collection_typing(init, expected):
    Path("t.py").write_text(textwrap.dedent(f"""\
        from collections import defaultdict, OrderedDict, ChainMap, Counter, deque
        from righttyper.random_dict import RandomDict

        def foo(x): pass

        foo({init})
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)
    output = Path("t.py").read_text()
    assert f"def foo(x: {expected}) -> None:" in output


@pytest.mark.parametrize('pattern, matching, notmatching, expected', [
    ['r"\\d+"', '"123"', '""', 'str'],
    ['rb"\\d+"', 'b"123"', 'b""', 'bytes'],
])
def test_pattern_match(pattern, matching, notmatching, expected):
    Path("t.py").write_text(textwrap.dedent(f"""\
        import re

        def foo(p, data):
            return re.match(p, data)

        foo(re.compile({pattern}), {matching})
        foo(re.compile({pattern}), {notmatching})
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert f"def foo(p: re.Pattern[{expected}], data: {expected}) -> re.Match[{expected}]|None:" in output


def test_namedtuple():
    Path("t.py").write_text(textwrap.dedent(f"""\
        from collections import namedtuple

        P = namedtuple('P', [])

        def foo(x):
            return x

        foo(P())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)
    output = Path("t.py").read_text()
    assert (
        "def foo(x: P) -> P:" in output or
        "def foo(x: \"P\") -> \"P\":" in output
    )


def test_class_properties():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x: int|None = None

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
        c.x = 10  # type: ignore[assignment]
        y = c.x
        del c.x
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)

    output = Path("t.py").read_text()
    print(output)

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def x(self: Self) -> str:" in output                # getter
    assert "def x(self: Self, value: int) -> None:" in output   # setter
    assert "def x(self: Self) -> None:" in output               # deleter


def test_class_properties_private():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x: int|None = None

            @property
            def __x(self):
                return str(self._x)

            @__x.setter
            def __x(self, value):
                self._x = value

            @__x.deleter
            def __x(self):
                del self._x

        c = C()
        c._C__x = 10  # type: ignore[assignment, attr-defined]
        y = c._C__x   # type: ignore[attr-defined]
        del c._C__x   # type: ignore[attr-defined]
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)

    output = Path("t.py").read_text()
    print(output)

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def __x(self: Self) -> str:" in output                # getter
    assert "def __x(self: Self, value: int) -> None:" in output   # setter
    assert "def __x(self: Self) -> None:" in output               # deleter


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
                    '-m', 't'], check=True)

    output = Path("t.py").read_text()

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def x(self: Self) -> str:" in output                # getter
    assert "def x(self: Self) -> None:" in output               # deleter


def test_class_properties_inner_functions():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x: int|None = None

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
        c.x = 10.0  # type: ignore[assignment]
        y = c.x
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)

    output = Path("t.py").read_text()

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def foo() -> str:" in output            # getter's
    assert "def foo(v: float) -> int:" in output    # setter's

    # check for inner function's inner function
    assert "def bar() -> None:" in output


def test_class_properties_inherited():
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def __init__(self):
                self._x: int|None = None

            @property
            def x(self):
                return str(self._x)

            @x.setter
            def x(self, value):
                self._x = value

            @x.deleter
            def x(self):
                del self._x

        class D(C):
            pass

        d = D()
        d.x = 10  # type: ignore[assignment]
        y = d.x
        del d.x
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)
    output = Path("t.py").read_text()
    print(output)

    assert "def __init__(self: Self) -> None:" in output

    # TODO parse functions out so that the annotation is included
    assert "def x(self: Self) -> str:" in output                # getter
    assert "def x(self: Self, value: int) -> None:" in output   # setter
    assert "def x(self: Self) -> None:" in output               # deleter


@pytest.mark.skip(reason="Doesn't currently work")  # FIXME
def test_class_properties_from_metaclass():
    Path("t.py").write_text(textwrap.dedent("""\
        class Meta(type):
            @property
            def my_property(cls):
                return cls()

        class C(metaclass=Meta):
            pass

        x = C.my_property
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '-m', 't'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)
    print(output)

    # According to mypy, typing.Self can't be used in metaclasses.
    # Also, typing the return value is problematic.  Perhaps use a typevar, as in
    #   def my_property[T](cls: type[T]) -> T
    # ?

#    assert get_function(code, 'Meta.my_property') == textwrap.dedent("""\
#        @property
#        def my_property(cls: "Meta"): ...
#    """)


def test_self_simple():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return self

        o = A()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', 't.py'],
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
                    '--no-sampling', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> Self:" in Path("t.py").read_text()


def test_self_bound_method():
    # It's important to use a separate function (h() below) to make sure a bound method
    # object is created... Python 3.12 seems to optimize things like "(C().f)()"
    Path("t.py").write_text(textwrap.dedent("""\
        class C:
            def f(self):
                pass

            @classmethod
            def g(cls):
                pass

        def h(f):
            f()

        h(C().f)
        h(C.g)
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    't.py'], check=True)

    output = Path("t.py").read_text()
    assert 'def f(self: Self) -> None:' in output
    assert 'def g(cls: type[Self]) -> None:' in output


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
                    '--no-sampling', 't.py'],
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
                    '--no-sampling', 't.py'],
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
                    '--no-sampling', 't.py'],
                   check=True)

    assert "def foo(self: Self) -> \"A\":" in Path("t.py").read_text()


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self_classmethod(python_version):
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            @classmethod
            def static_initializer(cls):
                return cls()

        o = A.static_initializer()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    f'--python-version={python_version}', '--no-sampling', 't.py'],
                   check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    if python_version == '3.10':
        assert get_function(code, 'A.static_initializer') == textwrap.dedent("""\
            @classmethod
            def static_initializer(cls: "type[A]") -> "A": ...
        """)
    else:
        assert get_function(code, 'A.static_initializer') == textwrap.dedent("""\
            @classmethod
            def static_initializer(cls: type[Self]) -> Self: ...
        """)


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self_inherited_classmethod(python_version):
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
                    f'--python-version={python_version}', '--no-sampling', 't.py'],
                   check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    if python_version == '3.10':
        assert get_function(code, 'A.static_initializer') == textwrap.dedent("""\
            @classmethod
            def static_initializer(cls: "type[A]") -> "A": ...
        """)
    else:
        assert get_function(code, 'A.static_initializer') == textwrap.dedent("""\
            @classmethod
            def static_initializer(cls: type[Self]) -> Self: ...
        """)


def test_self_within_other_types():
    Path("t.py").write_text(textwrap.dedent("""\
        class A:
            def foo(self):
                return [self, self]

        o = A()
        o.foo()
    """))

    subprocess.run([sys.executable, '-m', 'righttyper', '--output-files', '--overwrite',
                    '--no-sampling', 't.py'],
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
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()

    print(output)
    assert "def foo(self: Self) -> Generator[Self, None, Self]" in output


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self_subtyping(python_version):
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
                    '--no-sampling', f'--python-version={python_version}', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    # IntegerAdd IS-A NumberAdd, the enclosed class; so the argument should be 'Self'
    if python_version == '3.10':
        assert get_function(code, 'NumberAdd.operation') == textwrap.dedent("""\
            def operation(self: "NumberAdd", rhs: "NumberAdd") -> "NumberAdd": ...
        """)
    else:
        assert get_function(code, 'NumberAdd.operation') == textwrap.dedent("""\
            def operation(self: Self, rhs: Self) -> Self: ...
        """)


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self_subtyping_reversed(python_version):
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
        a.operation(b)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', f'--python-version={python_version}', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    # The argument isn't Self as (NumberAdd IS-A IntegerAdd) doesn't hold
    if python_version == '3.10':
        assert get_function(code, 'NumberAdd.operation') == textwrap.dedent("""\
            def operation(self: "NumberAdd", rhs: "NumberAdd") -> "NumberAdd": ...
        """)
    else:
        # FIXME the rhs should ideally just be "NumberAdd"
        assert get_function(code, 'NumberAdd.operation') == textwrap.dedent("""\
            def operation(self: Self, rhs: "NumberAdd|Self") -> Self: ...
        """)


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_self_subtyping_reversed_too(python_version):
    # This is the same as test_self_subtyping_reversed,
    # but with "operation" calls in opposite order.
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

        a.operation(b)
        b.operation(a)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', f'--python-version={python_version}', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    # The argument isn't Self as (NumberAdd IS-A IntegerAdd) doesn't hold
    if python_version == '3.10':
        assert get_function(code, 'NumberAdd.operation') == textwrap.dedent("""\
            def operation(self: "NumberAdd", rhs: "NumberAdd") -> "NumberAdd": ...
        """)
    else:
        # FIXME the rhs should ideally just be "NumberAdd"
        assert get_function(code, 'NumberAdd.operation') == textwrap.dedent("""\
            def operation(self: Self, rhs: "NumberAdd|Self") -> Self: ...
        """)


def test_returns_or_yields_generator():
    t = textwrap.dedent("""\
        def test(a):
            if a < 5:
                return "too small :("
            else:
                for i in range(a):
                    yield a
            return None

        for _ in test(3): pass
        for _ in test(10): pass
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)

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
                    '--no-sampling', 't.py'], check=True)

    output = Path("t.py").read_text()
    assert "def test(a: int) -> Iterator[int|str]" in output


@pytest.mark.parametrize('replace_dict', [False, True])
def test_random_dict(replace_dict):
    t = textwrap.dedent(f"""\
        def f(x):
            return len(x)

        d = {{'a': {{'b': 2}}}}
        f(d)

        from righttyper.random_dict import RandomDict
        assert {'' if replace_dict else ' not '} isinstance(d, RandomDict)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    *(('--replace-dict',) if replace_dict else ('--no-replace-dict',)),
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(x: dict[str, dict[str, int]]) -> int" in output


def test_instrument_pytest():
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


@pytest.mark.dont_run_mypy  # mypy fails, but it's not quite clear why
def test_higher_order_functions():
    # Check that we can handle such functions.  Do we need the CALL event to handle them?
    t = textwrap.dedent("""\
        def foo(x):
            return x+x

        def runner(f):
            f(0)
            return f

        runner(foo)("a")
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def foo[T1: (int, str)](x: T1) -> T1" in output
    assert "def runner[T1: (int, str)](f: Callable[[T1], T1]) -> Callable[[T1], T1]" in output


def test_generalize_union_arg_typevar():
    t = textwrap.dedent("""\
        def f(x):
            return x[0]

        f([10])
        f(['10'])
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--python-version=3.12', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f[T1: (int, str)](x: list[T1]) -> T1" in output


def test_generalize_union_arg_not_typevar():
    t = textwrap.dedent("""\
        def f(x):
            x.append(x[0])

        f([10])
        f(['10'])
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(x: list[int|str]) -> None" in output


def test_generalize_union_return_typevar():
    t = textwrap.dedent("""\
        def f(x):
            return [x]

        f(10)
        f('10')
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--python-version=3.12', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f[T1: (int, str)](x: T1) -> list[T1]" in output


def test_generalize_union_return_not_typevar():
    t = textwrap.dedent("""\
        def f(x):
            return ['10' if x else 10]

        f(False)
        f(True)
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(x: bool) -> list[int|str]" in output


@pytest.mark.dont_run_mypy # unnecessary 
def test_object_overridden_getattr():
    # Sometimes dynamic attribute lookups have side effects or, as in the case of
    # tqdm and rich tests, lead to infinite recursions as their __getattr__ call getattr()
    # on the same object.
    t = textwrap.dedent("""\
        class Thing:
            def __getattr__(self, name):
                raise RuntimeError()

        def f(t):
            pass

        f(Thing())
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    # mostly we are checking that it doesn't fail (raises fatal exception)
    output = Path("t.py").read_text()
    assert "def f(t: Thing) -> None" in output


@pytest.mark.dont_run_mypy # unnecessary 
def test_object_with_empty_dir():
    # Derived from 'rich' test case "Issue #1838 - Edge case with Faiss library - object with empty dir()"
    # This leads to an AttributeError if we do any isinstance(obj, X) where X is from collections.abc
    t = textwrap.dedent("""\
        def g(x):
            pass

        class Thing:
            @property
            def __class__(self):
                raise AttributeError

            def f(self):
                pass

        Thing().f()
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    # mostly we are checking that it doesn't fail (raises fatal exception)
    output = Path("t.py").read_text()
    assert "def f(self: Self) -> None" in output


@pytest.mark.parametrize("python_version", ["3.10", "3.11"])
def test_empty_container(python_version):
    t = textwrap.dedent("""\
        def f(x):
            return len(x)

        f([])
        f({})
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    f'--python-version={python_version}', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    if python_version == "3.11":
        assert get_function(code, 'f') == textwrap.dedent("""\
            def f(x: dict[Never, Never]|list[Never]) -> int: ...
        """)
    else:
        assert get_function(code, 'f') == textwrap.dedent("""\
            def f(x: dict[Any, Any]|list[Any]) -> int: ...
        """)


@pytest.mark.skip(reason="just documents an idea")
def test_container_is_modified():
    # TODO should we resample mutable containers upon return?
    t = textwrap.dedent("""\
        def f(x):
            x.append(1)

        f([])
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    assert "def f(x: list[int]) -> None" in output


@pytest.mark.parametrize("python_version", ["3.9"])
def test_typing_union(python_version):
    t = textwrap.dedent("""\
        def f(x):
            return str(x) if x else None

        def g(x):
            pass

        f(10)
        f("foo")
        f(0)
        g(f)
        g([])
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    f'--python-version={python_version}', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(x: Union[int, str]) -> Optional[str]: ...
    """)
    assert get_function(code, 'g') == textwrap.dedent("""\
        def g(x: Union[list[Any], Callable[[Union[int, str]], Optional[str]]]) -> None: ...
    """)


@pytest.mark.parametrize('all_type', ['list', 'tuple'])
def test_typefinder_name_from_all_preferred(all_type):
    # C has 4 names:
    #   - m.foo.C, where it's defined
    #   - m.C, where it's imported into m
    #   - m.xyz, declared in '__all__'
    #   - __main__.m.foo.C
    #
    # we want to see it pick m.xyz

    Path("m").mkdir()
    (Path("m") / "__init__.py").write_text(textwrap.dedent(f"""\
        from .foo import C
        __all__ = {"['xyz']" if all_type == "list" else "('xyz',)"}
        xyz = C
        """
    ))
    (Path("m") / "foo.py").write_text(textwrap.dedent("""\
        class C:
            pass
        """
    ))
    Path("t.py").write_text(textwrap.dedent("""\
        import m.foo

        def f(x): pass

        f(m.foo.C())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite',
                    '--output-files', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(x: m.xyz) -> None: ...
    """)


def test_typefinder_name_without_underscore_preferred():
    # C has 5 names:
    #   - m.foo.C, where it's defined
    #   - m._C, where it's imported (and defined before m.C)
    #   - m.C (the alias)
    #   - __main__.m.foo._C
    #   - __main__.m.foo.C
    #
    # we want to see it pick m.xyzzy

    Path("m").mkdir()
    (Path("m") / "__init__.py").write_text(textwrap.dedent("""\
        from .foo import C as _C
        C = _C
        """
    ))
    (Path("m") / "foo.py").write_text(textwrap.dedent("""\
        class C:
            pass
        """
    ))
    Path("t.py").write_text(textwrap.dedent("""\
        import m.foo

        def f(x): pass

        f(m.foo.C())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite',
                    '--output-files', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(x: m.C) -> None: ...
    """)


def test_typefinder_mod_without_underscore_preferred():
    # C has 5 names:
    #   - _foo.C, where it's defined
    #   - m._foo.C, where it's imported
    #   - m.C (the alias)
    #   - __main__.m._foo.C
    #   - __main__.m.C
    #
    # we want to see it pick m.C

    Path("m").mkdir()
    (Path("m") / "__init__.py").write_text(textwrap.dedent("""\
        import _foo
        C = _foo.C
        """
    ))
    Path("_foo").mkdir()
    (Path("_foo") / "__init__.py").write_text(textwrap.dedent("""\
        class C:
            pass
        """
    ))
    Path("t.py").write_text(textwrap.dedent("""\
        import m

        def f(x): pass

        f(m.C())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite',
                    '--output-files', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)
    print(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(x: m.C) -> None: ...
    """)


def test_typefinder_shorter_name_preferred():
    # C has two names:
    #   - m.foo.C, where it's defined
    #   - m.C, where it's imported
    #
    # we want to see it pick m.C
    Path("m").mkdir()
    (Path("m") / "__init__.py").write_text(textwrap.dedent("""\
        from .foo import C
        """
    ))
    (Path("m") / "foo.py").write_text(textwrap.dedent("""\
        class C:
            pass
        """
    ))
    Path("t.py").write_text(textwrap.dedent("""\
        import m

        def f(x):
            pass

        f(m.C())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite',
                    '--output-files', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(x: m.C) -> None: ...
    """)


def test_typefinder_defined_in_main():
    # Also check that we don't just build a map the first time we need it
    # by checking for a name that is only defined after that
    Path("t.py").write_text(textwrap.dedent("""\
        class C1(): pass

        def f(x):
            pass

        f(C1())

        class C2:
            pass

        f(C2())
        """
    ))

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite',
                    '--output-files', '--no-sampling', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(x: "C1|C2") -> None: ...
    """)


def test_inconsistent_samples():
    Path("t.py").write_text(textwrap.dedent("""\
        def f():
            def g(a, b):
                return a+b
            return g

        g = f()
        g(1,2)

        # Fake an inconsistent (different arity) sample
        import righttyper.righttyper as rt
        rt.obs.record_start(
            code=g.__code__,
            frame_id=rt.FrameId(0),
            arg_types=(
                rt.get_type_name(int), 
                rt.get_type_name(int), 
                rt.get_type_name(int),
            ),
            self_type=None,
            self_replacement=None
        )
        rt.obs.record_return(
            code=g.__code__, frame_id=rt.FrameId(0), return_value=1
        )
        """
    ))

    p = subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite',
                        '--output-files', '--no-sampling', 't.py'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert b'Error' not in p.stdout

    # no annotation expected
    assert get_function(code, 'f.<locals>.g') == textwrap.dedent("""\
        def g(a, b): ...
    """)


@pytest.mark.dont_run_mypy  # would fail due to f("foo") calls
def test_use_top_pct():
    t = textwrap.dedent(f"""\
        def f(x):
            return x

        for i in range(10):
            f(i)
        f("foo")
        f("foo")
        """)

    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', '--overwrite', '--output-files',
                    '--no-sampling', '--use-top-pct=80', 't.py'], check=True)
    output = Path("t.py").read_text()
    code = cst.parse_module(output)

    assert get_function(code, 'f') == textwrap.dedent(f"""\
        def f(x: int) -> int: ...
    """)
