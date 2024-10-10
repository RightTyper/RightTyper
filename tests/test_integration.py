import textwrap
import subprocess
import sys
from pathlib import Path
import pytest


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
