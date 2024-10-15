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
