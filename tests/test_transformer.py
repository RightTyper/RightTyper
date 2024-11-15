import libcst as cst
import textwrap
from righttyper.unified_transformer import UnifiedTransformer, types_in_annotation, _namespace_of
from righttyper.righttyper_types import FuncInfo, Filename, FunctionName, Typename, ArgumentName
import typing as T
import pytest
import re


def test_transformer_not_annotated_missing():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x):
            return x/2
    """))

    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                FuncInfo(Filename('foo.py'), FunctionName('foo')):
                (
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                )
            },
            not_annotated = dict(),
            module_name='foo',
            module_names=[
                'foo'
            ]
        )

    code.visit(t)


def get_function(m: cst.Module, name: str) -> str|None:
    class V(cst.CSTVisitor):
        def __init__(self):
            self.found = None
            self.name_stack = []

        def visit_ClassDef(self, node: cst.ClassDef):
            self.name_stack.append(node.name.value)

        def leave_ClassDef(self, node: cst.ClassDef):
            self.name_stack.pop()

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            self.name_stack.append(node.name.value)
            qual_name = ".".join(self.name_stack)
            self.name_stack.append("<locals>")
            if qual_name == name:
                self.found = node
                return False # stop here
            return True

        def leave_FunctionDef(self, node: cst.FunctionDef):
            self.name_stack.pop()
            self.name_stack.pop()

    v = V()
    m.visit(v)
    return cst.Module([v.found]).code.lstrip('\n') if v.found else None


def get_if_type_checking(m: cst.Module) -> str|None:
    class V(cst.CSTVisitor):
        def __init__(self):
            self.found = None

        def visit_If(self, node: cst.If) -> bool:
            if isinstance(node.test, cst.Name) and node.test.value == 'TYPE_CHECKING':
                self.found = node
                return False # stop here
            return True

    v = V()
    m.visit(v)
    return cst.Module([v.found]).code.lstrip('\n') if v.found else None


def test_transform_function():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            return (x+y)/2

        def bar(x):
            return x/2

        def baz(z):
            return z/2
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    baz = FuncInfo(Filename('foo.py'), FunctionName('baz'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
                baz: (
                    [
                        (ArgumentName('z'), Typename('int'))
                    ],
                    Typename('wrong')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('return')},
                baz: {ArgumentName('z')}
            },
            module_name='foo',
            module_names=['foo']
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y) -> float:
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x):
            return x/2
    """)

    assert get_function(code, 'baz') == textwrap.dedent("""\
        def baz(z: int):
            return z/2
    """)

    assert get_if_type_checking(code) == None


def test_transform_method():
    code = cst.parse_module(textwrap.dedent("""\
        class C:
            def foo(self, x, y):
                return (x+y)/2

            @staticmethod
            def bar(x):
                return x/2

            @classmethod
            def baz(cls, z):
                return z/2
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('C.foo'))
    bar = FuncInfo(Filename('foo.py'), FunctionName('C.bar'))
    baz = FuncInfo(Filename('foo.py'), FunctionName('C.baz'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
                bar: (
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
                baz: (
                    [
                        (ArgumentName('z'), Typename('int'))
                    ],
                    Typename('float')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('return')},
                bar: {ArgumentName('x'), ArgumentName('return')},
                baz: {ArgumentName('z'), ArgumentName('return')},
            },
            module_name='foo',
            module_names=['foo']
        )

    code = code.visit(t)
    assert get_function(code, 'C.foo') == textwrap.dedent("""\
        def foo(self, x: int, y) -> float:
            return (x+y)/2
    """)

    assert get_function(code, 'C.bar') == textwrap.dedent("""\
        @staticmethod
        def bar(x: int) -> float:
            return x/2
    """)

    assert get_function(code, 'C.baz') == textwrap.dedent("""\
        @classmethod
        def baz(cls, z: int) -> float:
            return z/2
    """)

    assert get_if_type_checking(code) == None


def test_transform_local_function():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            def bar(z):
                return z/2
            return bar(x+y)
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    bar = FuncInfo(Filename('foo.py'), FunctionName('foo.<locals>.bar'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('float'))
                    ],
                    Typename('float')
                ),
                bar: (
                    [
                        (ArgumentName('z'), Typename('int'))
                    ],
                    Typename('float')
                ),
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')},
                bar: {ArgumentName('z'), ArgumentName('return')}
            },
            module_name='foo',
            module_names=['foo']
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: float) -> float:
            def bar(z: int) -> float:
                return z/2
            return bar(x+y)
    """)

    assert get_if_type_checking(code) == None


def test_transform_adds_typing_import_for_typing_names():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('typing.Optional[int]'))
                    ],
                    Typename('list[typing.Never]')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('return')},
            },
            module_name='foo',
            module_names=['foo']
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: Optional[int]) -> list[Never]: ...
    """)

    code_str = str(code.code)
    assert re.search(r"^ *from typing import .*\bOptional\b", code_str)
    assert re.search(r"^ *from typing import .*\bNever\b", code_str)
    assert get_if_type_checking(code) == None


def test_transform_unknown_type_as_string():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            return x/2
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('x.y.WholeNumber|None'))
                    ],
                    Typename('x.z.FloatingPointNumber')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')}
            },
            module_name = 'foo',
            module_names = [
                "foo",
                "x.y",
                "x"
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: "x.y.WholeNumber|None") -> "x.z.FloatingPointNumber":
            return x/2
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import x
            import x.y
    """)


def test_transform_unknown_type_with_import_annotations():
    code = cst.parse_module(textwrap.dedent("""\
        from __future__ import annotations

        def foo(x, y):
            return x/2
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('x.y.WholeNumber|None'))
                    ],
                    Typename('x.z.FloatingPointNumber')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')}
            },
            module_name = 'foo',
            module_names = [
                "foo",
                "x.y",
                "x"
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: x.y.WholeNumber|None) -> x.z.FloatingPointNumber:
            return x/2
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import x
            import x.y
    """)


def test_transform_deletes_type_hint_comments_in_header():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y): # type: (int, int) -> Any
            return (x+y)/2

        def bar(x):   # type: (Any) -> None
            pass
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('float')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y')}
            },
            module_name = 'foo',
            module_names = [
                'foo'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: int):
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x):   # type: (Any) -> None
            pass
    """)

    assert get_if_type_checking(code) == None


def test_transform_deletes_type_hint_comments_in_parameters():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(
            x,  # type: int
            y   # type: float
        ):
            # type: (...) -> Any
            return (x+y)/2

        def bar(
            x   # type: Any
        ):
            # type: (...) -> None
            pass
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('float')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y')}
            },
            module_name = 'foo',
            module_names = [
                'foo'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(
            x: int,
            y: int
        ):
            # type: (...) -> Any
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(
            x   # type: Any
        ):
            # type: (...) -> None
            pass
    """)

    assert get_if_type_checking(code) == None


def test_transform_deletes_type_hint_comments_for_retval():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(
            x,  # type: int
            y   # type: float
        ):
            # type: (...) -> Any
            return (x+y)/2

        def bar(
            x   # type: Any
        ):
            # type: (...) -> None
            pass
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('float')
                )
            },
            not_annotated = {
                foo: {ArgumentName('return')}
            },
            module_name = 'foo',
            module_names=[
                'foo'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(
            x,  # type: int
            y   # type: float
        ) -> float:
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(
            x   # type: Any
        ):
            # type: (...) -> None
            pass
    """)

    assert get_if_type_checking(code) == None


def test_transform_locally_defined_types():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            return F((x+y)/2)

        class F:
            def __init__(self, v):
                self.v = v

            def foo(self, v):
                return F(v)

        def bar(x, y):
            return F((x+y)/2)
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    f_foo = FuncInfo(Filename('foo.py'), FunctionName('F.foo'))
    bar = FuncInfo(Filename('foo.py'), FunctionName('bar'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('foo.F')
                ),
                f_foo: (
                    [
                        (ArgumentName('v'), Typename('float')),
                    ],
                    Typename('foo.F')
                ),
                bar: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('foo.F')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')},
                f_foo: {ArgumentName('v'), ArgumentName('return')},
                bar: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')}
            },
            module_name = 'foo',
            module_names=[
                'foo'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: int) -> "F":
            return F((x+y)/2)
    """)

    assert get_function(code, 'F.foo') == textwrap.dedent("""\
        def foo(self, v: float) -> "F":
            return F(v)
    """)

    # F is now known, so no quotes are needed
    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x: int, y: int) -> F:
            return F((x+y)/2)
    """)

    assert get_if_type_checking(code) == None


def test_uses_imported_aliases():
    code = cst.parse_module(textwrap.dedent("""\
        from x.y import z as zed
        from y import T
        if True:
            import a.b as A

        def foo(x, y, z): ...

        import r    # imported after 'def foo', so can't be used in annotation
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('x.y.z')),
                        (ArgumentName('y'), Typename('y.T')),
                        (ArgumentName('z'), Typename('a.b.c.T')),
                    ],
                    Typename('r.t.T')
                ),
            },
            not_annotated = {
                foo: {
                    ArgumentName('x'), ArgumentName('y'),
                    ArgumentName('z'), ArgumentName('return')
                },
            },
            module_name = 'foo',
            module_names = [
                'foo',
                'x',
                'x.y',
                'y',
                'a',
                'a.b',
                'r'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: zed, y: T, z: A.c.T) -> "r.t.T": ...
    """)


    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import r
    """)


def test_uses_imported_domains():
    code = cst.parse_module(textwrap.dedent("""\
        if True:
            import x.y

        def foo(x): ...

        import r    # imported after 'def foo', so can't be used in annotation
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('x.y.z')),
                    ],
                    Typename('r.t.T')
                ),
            },
            not_annotated = {
                foo: {
                    ArgumentName('x'), ArgumentName('return')
                },
            },
            module_name = 'foo',
            module_names = [
                'foo',
                'x',
                'x.y',
                'r'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: x.y.z) -> "r.t.T": ...
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import r
    """)


def test_imports_subdomain_if_needed():
    code = cst.parse_module(textwrap.dedent("""\
        import x
        import a.b.c

        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('x.y.z')),
                    ],
                    Typename('a.b')
                ),
            },
            not_annotated = {
                foo: {
                    ArgumentName('x'), ArgumentName('return')
                },
            },
            module_name = 'foo',
            module_names = [
                'foo',
                'x',
                'x.y',
                'a',
                'a.b',
                'a.b.c',
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: "x.y.z") -> a.b: ...
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import x.y
    """)


def test_relative_import():
    code = cst.parse_module(textwrap.dedent("""\
        from .. import b
        from . import c
        from .c import X

        def foo(x, y, z): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('pkg.b.T')),
                        (ArgumentName('y'), Typename('pkg.a.c.T')),
                        (ArgumentName('z'), Typename('pkg.a.c.X')),
                    ],
                    Typename('None')
                ),
            },
            not_annotated = {
                foo: {
                    ArgumentName('x'), ArgumentName('y'), ArgumentName('z')
                },
            },
            module_name = 'pkg.a.a',
            module_names = [
                'pkg.a',
                'pkg.a.a',
                'pkg.b',
                'pkg.a.c',
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: b.T, y: c.T, z: X): ...
    """)

    assert get_if_type_checking(code) == None


@pytest.mark.skip(reason="Not yet supported")
def test_uses_local_imports():
    code = cst.parse_module(textwrap.dedent("""\
        def foo():
            import m.n
            def bar(x): ...

        class C:
            import n
            def foo(self, x): ...

            class D:
                def foo(self, x): ...

        def f(a, b): ...
    """))

    foobar = FuncInfo(Filename('foo.py'), FunctionName('foo.<locals>.bar'))
    Cfoo = FuncInfo(Filename('foo.py'), FunctionName('C.foo'))
    Dfoo = FuncInfo(Filename('foo.py'), FunctionName('C.D.foo'))
    f = FuncInfo(Filename('foo.py'), FunctionName('f'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foobar: (
                    [
                        (ArgumentName('x'), Typename('m.n.T')),
                    ],
                    Typename('None')
                ),
                Cfoo: (
                    [
                        (ArgumentName('x'), Typename('n.o.T')),
                    ],
                    Typename('None')
                ),
                Dfoo: (
                    [
                        (ArgumentName('x'), Typename('n.o.T')),
                    ],
                    Typename('None')
                ),
                f : (
                    [
                        (ArgumentName('a'), Typename('m.n.T')),
                        (ArgumentName('b'), Typename('n.o.T')),
                    ],
                    Typename('None')
                ),
            },
            not_annotated = {
                foobar: {
                    ArgumentName('x'),
                },
                Cfoo: {
                    ArgumentName('x'),
                },
                Dfoo: {
                    ArgumentName('x'),
                },
                f : {
                    ArgumentName('a'), ArgumentName('b')
                },
            },
            module_name = 'foo',
            module_names = [
                'foo',
                'm.n',
                'n',
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo.<locals>.bar') == textwrap.dedent("""\
        def bar(x: m.n.T): ...
    """)

    assert get_function(code, 'C.foo') == textwrap.dedent("""\
        def foo(self, x: n.o.T): ...
    """)
    assert get_function(code, 'C.D.foo') == textwrap.dedent("""\
        def foo(self, x: "n.o.T"): ...
    """)
    assert get_function(code, 'f') == textwrap.dedent("""\
        def f(a: "m.n.T", b: "n.o.T"): ...
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import m.n
            import n
    """)


@pytest.mark.skip(reason="no longer relevant as-is")
def test_assigned_names_are_known():
    code = cst.parse_module(textwrap.dedent("""\
        from typing import Any
        import foo
        if True:
            x = foo
        else:
            _, y = ..., foo.y
            z: Any = foo

        def foo(x):
            pass
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('z')),
                    ],
                    Typename('y')
                ),
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('return')},
            },
            module_name = 'foo',
            module_names = [
                'foo'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: z) -> y:
            pass
    """)

    assert get_if_type_checking(code) == None


def test_nonglobal_imported_modules_are_unknown():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            import a.b

        class C:
            import m as m2

        def bar(x):
            pass
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    bar = FuncInfo(Filename('foo.py'), FunctionName('bar'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('a.T')),
                        (ArgumentName('y'), Typename('a.b.T')),
                    ],
                    Typename('a.c.T')
                ),
                bar: (
                    [
                        (ArgumentName('x'), Typename('m.T')),
                    ],
                    Typename('m.T')
                ),
            },
            not_annotated = {
                foo: {
                    ArgumentName('x'), ArgumentName('y'), ArgumentName('return')
                },
                bar: {
                    ArgumentName('x'), ArgumentName('return')
                },
            },
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'a.b',
                'a.c',
                'm',
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: "a.T", y: "a.b.T") -> "a.c.T":
            import a.b
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x: "m.T") -> "m.T":
            pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import a
            import a.b
            import a.c
            import m
    """)


def test_nonglobal_assignments_are_unknown():
    code = cst.parse_module(textwrap.dedent("""\
        from typing import Any

        def foo(x):
            a = Any

        class C:
            m = Any

        def bar(x):
            pass
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    bar = FuncInfo(Filename('foo.py'), FunctionName('bar'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('a.T')),
                    ],
                    Typename('')
                ),
                bar: (
                    [
                        (ArgumentName('x'), Typename('m.T')),
                    ],
                    Typename('')
                ),
            },
            not_annotated = {
                foo: {
                    ArgumentName('x')
                },
                bar: {
                    ArgumentName('x')
                },
            },
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'm'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: "a.T"):
            a = Any
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x: "m.T"):
            pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import a
            import m
    """)


def test_types_in_annotation():
    def get_types(s):
        return types_in_annotation(cst.parse_expression(s))

    assert {'int'} == get_types('int')
    assert {'Tuple', 'int', 'float'} == get_types('Tuple[int, float]')
    assert {'Dict', 'foo.bar', 'bar.baz'} == get_types('Dict[foo.bar, bar.baz]')
    assert {'Union', 'int', 'float', 'Tuple', 'a.b.c'} == get_types('Union[int, float, Tuple[int, a.b.c]]')


def test_namespace_of():
    assert '' == _namespace_of('foo')
    assert 'foo' == _namespace_of('foo.bar')
    assert 'foo.bar' == _namespace_of('foo.bar.baz')
