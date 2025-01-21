import ast
import textwrap
from righttyper.ast_instrument import (
    instrument,
    SEND_HANDLER,
    SEND_WRAPPER,
    ASEND_HANDLER,
    ASEND_WRAPPER
)


def parse(s: str) -> ast.AST:
    return ast.parse(textwrap.dedent(s))


def unparse(t: ast.AST) -> str:
    return ast.unparse(t) + "\n"


def test_wrap_send():
    t = parse("""\
        def f(x):
            return x.y.g.send(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.righttyper import {SEND_HANDLER} as {SEND_WRAPPER}

        def f(x):
            return {SEND_WRAPPER}(x.y.g, 10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_wrap_asend():
    t = parse("""\
        async def f(x):
            return await x.y.g.asend(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.righttyper import {ASEND_HANDLER} as {ASEND_WRAPPER}

        async def f(x):
            return await {ASEND_WRAPPER}(x.y.g, 10)
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
        from righttyper.righttyper import {SEND_HANDLER} as {SEND_WRAPPER}
        import sys

        def f(x):
            return {SEND_WRAPPER}(x, 10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw
