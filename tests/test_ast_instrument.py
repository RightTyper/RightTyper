import ast
import pytest
import textwrap
from righttyper.ast_instrument import (
    instrument,
    SEND_WRAPPER_NAME,
    SEND_WRAPPER_ASNAME,
    RANDOM_DICT_NAME,
    RANDOM_DICT_ASNAME
)


def parse(s: str) -> ast.Module:
    return ast.parse(textwrap.dedent(s))


def unparse(t: ast.Module) -> str:
    return ast.unparse(t) + "\n"


def test_send_wrap_send():
    t = parse("""\
        def f(x):
            return x.y.g.send(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.righttyper import {SEND_WRAPPER_NAME} as {SEND_WRAPPER_ASNAME}

        def f(x):
            return {SEND_WRAPPER_ASNAME}(x.y.g.send)(10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_send_wrap_asend():
    t = parse("""\
        async def f(x):
            return await x.y.g.asend(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.righttyper import {SEND_WRAPPER_NAME} as {SEND_WRAPPER_ASNAME}

        async def f(x):
            return await {SEND_WRAPPER_ASNAME}(x.y.g.asend)(10)
    """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_send_nothing_to_wrap():
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


def test_send_import_after_from_future():
    t = parse("""\
        from __future__ import annotations
        import sys

        def f(x):
            return x.send(10)
        """)

    t = instrument(t)

    assert unparse(t) == textwrap.dedent(f"""\
        from __future__ import annotations
        from righttyper.righttyper import {SEND_WRAPPER_NAME} as {SEND_WRAPPER_ASNAME}
        import sys

        def f(x):
            return {SEND_WRAPPER_ASNAME}(x.send)(10)
    """)


def test_send_assignment_context():
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


def test_dict_literal():
    t = parse("""\
        x = {'a': 0, 'b': 1}
        y = {}
        """)

    t = instrument(t, replace_dict=True)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.random_dict import {RANDOM_DICT_NAME} as {RANDOM_DICT_ASNAME}
        x = {RANDOM_DICT_ASNAME}({{'a': 0, 'b': 1}})
        y = {RANDOM_DICT_ASNAME}({{}})
        """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_dict_comprehension():
    t = parse("""\
        x = {i: i + 1 for i in range(10)}
        """)

    t = instrument(t, replace_dict=True)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.random_dict import {RANDOM_DICT_NAME} as {RANDOM_DICT_ASNAME}
        x = {RANDOM_DICT_ASNAME}({{i: i + 1 for i in range(10)}})
        """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_dict_call():
    t = parse("""\
        x = dict([('a', 1), ('b', 2)])
        """)

    t = instrument(t, replace_dict=True)

    assert unparse(t) == textwrap.dedent(f"""\
        from righttyper.random_dict import {RANDOM_DICT_NAME} as {RANDOM_DICT_ASNAME}
        x = {RANDOM_DICT_ASNAME}([('a', 1), ('b', 2)])
        """)

    compile(t, 'tp.py', 'exec') # ensure it doesn't throw


def test_dict_import_after_from_future():
    t = parse("""\
        from __future__ import annotations
        import sys

        def f(x):
            return dict()
        """)

    t = instrument(t, replace_dict=True)

    assert unparse(t) == textwrap.dedent(f"""\
        from __future__ import annotations
        from righttyper.random_dict import {RANDOM_DICT_NAME} as {RANDOM_DICT_ASNAME}
        import sys

        def f(x):
            return {RANDOM_DICT_ASNAME}()
    """)
