import libcst as cst
import textwrap
from righttyper.typeshed import get_func_params, get_typeshed_func_params
from righttyper.typeinfo import TypeInfo, UnknownTypeInfo
import pytest


def test_names_builtin():
    code = cst.parse_module(textwrap.dedent("""\
        def f(x: int): pass
    """))
    pars = get_func_params(code, "foo", "f")
    assert pars == [TypeInfo('', 'int'), None]
    assert pars[0] is not None and pars[0].type_obj is int


def test_names_import():
    # TODO what about module references?  "import x", then use 'x' ?
    code = cst.parse_module(textwrap.dedent("""\
        import a, a.b, b.c as bc

        def f(x: a.b.c.d, y: bc.d.e): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('a.b', 'c.d'),
        TypeInfo('b.c', 'd.e'),
        None,
    ]


def test_names_import_from():
    code = cst.parse_module(textwrap.dedent("""\
        from a.b import c, d as duh, e as uh

        def f(x: c.d, y: duh.f.g, z: uh): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('a.b', 'c.d'),
        TypeInfo('a.b', 'd.f.g'),
        TypeInfo('a.b', 'e'),
        None,
    ]


def test_names_local():
    code = cst.parse_module(textwrap.dedent("""\
        class A:
            class B: pass

        def f(x: A.B): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo("foo", "A.B"),
        None,
    ]


@pytest.mark.skip(reason="Not yet implemented")
def test_names_local_override():
    code = cst.parse_module(textwrap.dedent("""\
        from bar import int
        class int: pass

        def f(x: int): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo("foo", "int"),
        None,
    ]


def test_names_undefined():
    code = cst.parse_module(textwrap.dedent("""\
        def f(x: "dunno"): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo("foo", "dunno"),
        None,
    ]


def test_type_parsing():
    code = cst.parse_module(textwrap.dedent("""\
        def f(x): pass
    """))
    assert get_func_params(code, "foo", "f") == [None, None]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: int): pass
    """))
    assert get_func_params(code, "foo", "f") == [TypeInfo('', 'int'), None]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: list[int]): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('', 'list', args=(
            TypeInfo('', 'int'),
        )),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: int|str): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo.from_set({
            TypeInfo('', 'int'),
            TypeInfo('', 'str'),
        }),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: "int|str"): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo.from_set({
            TypeInfo('', 'int'),
            TypeInfo('', 'str'),
        }),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: list["int|str"]): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('', 'list', args=(
            TypeInfo.from_set({
                TypeInfo('', 'int'),
                TypeInfo('', 'str'),
            }),
        )),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        from collections.abc import Callable
        def f(x: Callable[[int], None]): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('collections.abc', 'Callable', args=(
            TypeInfo.list([
                TypeInfo('', 'int'),
            ]),
            TypeInfo('', 'None'),
        )),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: tuple[int, ...]): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('', 'tuple', args=(
            TypeInfo('', 'int'),
            ...
        )),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        def f(x: tuple[()]): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('', 'tuple', args=(
            (),
        )),
        None,
    ]

    code = cst.parse_module(textwrap.dedent("""\
        import jaxtyping
        import numpy

        def f(x: jaxtyping.Float16[numpy.ndarray, "1 1 1"]): pass
    """))
    assert get_func_params(code, "foo", "f") == [
        TypeInfo('jaxtyping', 'Float16', args=(
            TypeInfo('numpy', 'ndarray'),
            "1 1 1"
        )),
        None,
    ]


def test_from_typeshed():
    assert get_typeshed_func_params("builtins", "object.__eq__") == [
        None,   # self
        TypeInfo.from_type(object),
        TypeInfo.from_type(bool),  # return type
    ]
