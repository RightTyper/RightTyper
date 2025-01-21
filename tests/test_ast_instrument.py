import ast
import textwrap
from righttyper.ast_instrument import (
    instrument,
    WRAPPER_NAME,
    WRAPPER_ASNAME,
)


def parse(s: str) -> ast.Module:
    return ast.parse(textwrap.dedent(s))


def unparse(t: ast.Module) -> str:
    return ast.unparse(t) + "\n"


def test_wrap_send():
    t = parse("""\
        def f(x):
            return x.y.g.send(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.righttyper import {WRAPPER_NAME} as {WRAPPER_ASNAME}

        def f(x):
            return {WRAPPER_ASNAME}(x.y.g.send)(10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_wrap_asend():
    t = parse("""\
        async def f(x):
            return await x.y.g.asend(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.righttyper import {WRAPPER_NAME} as {WRAPPER_ASNAME}

        async def f(x):
            return await {WRAPPER_ASNAME}(x.y.g.asend)(10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_nothing_to_wrap():
    t = parse("""\
        def f(x):
            return x/2
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        def f(x):
            return x / 2
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_import_after_from_future():
    t = parse("""\
        from __future__ import annotations
        import sys

        def f(x):
            return x.send(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from __future__ import annotations
        from righttyper.righttyper import {WRAPPER_NAME} as {WRAPPER_ASNAME}
        import sys

        def f(x):
            return {WRAPPER_ASNAME}(x.send)(10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_assignment_context():
    t = parse("""\
        class C:

            def __init__(self):
                self.send = 10
                self.asend, x = (1, 2)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        class C:

            def __init__(self):
                self.send = 10
                self.asend, x = (1, 2)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw
