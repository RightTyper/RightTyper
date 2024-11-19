import textwrap
import libcst as cst
from righttyper.generate_stubs import PyiTransformer


def generate_stub(orig_code: str) -> str:
    m = cst.parse_module(orig_code)
#    print(m)
    m = m.visit(PyiTransformer())
#    print(m)
    return m.code


def test_stubs(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        import sys

        A = B = 42
        CALC = 1+1
        CALC += 2

        # blah blah blah

        class C:
            '''blah blah blah'''
            class D:
                PI = 314

            def __init__(self: Self, x: int) -> None:  # initializes me
                self.x = x

            def f(self: Self) -> int:
                return self.x

        def f(x: int) -> int:
            return C(x).f()
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        import sys
        from typing import Any
        A: int
        B: int
        CALC: Any
        class C:
            class D:
                PI: int
            def __init__(self: Self, x: int) -> None: ...
            def f(self: Self) -> int: ...
        def f(x: int) -> int: ...
        """)

def test_stubs_no_any(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        import sys

        A = 42

        def f(x: int) -> int:
            return C(x).f()
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        import sys
        A: int
        def f(x: int) -> int: ...
        """)


def test_stubs_assign_tuple(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        X, Y, Z = 'a', 10, .0
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\

        """)


def test_stubs_empty_class(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        class Foo:
            '''Maybe one day we'll write more'''

        def f(x: int) -> int:
            return 42
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        class Foo:
            pass
        def f(x: int) -> int: ...
        """)


def test_stubs_conditional(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            "this should go away"
            import ast

        def f(x: "ast.AST") -> int:
            return 42
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import ast
        def f(x: "ast.AST") -> int: ...
        """)


def test_stubs_context_handler(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        with something():
            "this should go away"
            import ast

        def f(x: "ast.AST") -> int:
            return 42
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        with something():
            import ast
        def f(x: "ast.AST") -> int: ...
        """)



def test_stubs_try(tmp_path, monkeypatch):
    code = textwrap.dedent("""\
        try:
            from foo import bar
        except ImportError:
            import foobar as bar

        def f(x: bar) -> int:
            return 42
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        try:
            from foo import bar
        except ImportError:
            import foobar as bar
        def f(x: bar) -> int: ...
        """)


def test_stubs_all_variable(tmp_path, monkeypatch):
    # __all__ is included in many typeshed "pyi"s.
    code = textwrap.dedent("""\
        __all__ = [
            "foo",
            "Bar"
        ]

        def foo() -> int:
            return 42

        class Bar(object):
            def __init__(self, x):
                pass

        def baz() -> float:
            pass
        """
    )

    output = generate_stub(code)
    assert output == textwrap.dedent("""\
        __all__ = [
            "foo",
            "Bar"
        ]
        def foo() -> int: ...
        class Bar(object):
            def __init__(self, x): ...
        def baz() -> float: ...
        """)
