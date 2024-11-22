import libcst as cst
import libcst.matchers as cstm
import textwrap
from righttyper.unified_transformer import UnifiedTransformer, types_in_annotation, used_names
from righttyper.righttyper_types import FuncInfo, Filename, FunctionName, Typename, ArgumentName, FuncAnnotation
import typing
import pytest
import re


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


def assert_regex(pattern: str, text: str|None) -> re.Match:
    m = re.search(pattern, text if text else "")
    assert m, f"'{text}' doesn't match '{pattern}'"
    return m


def get_if_type_checking(m: cst.Module) -> str|None:
    stmts = typing.cast(list[cst.If], list(cstm.findall(m, cstm.If(test=cstm.Name("TYPE_CHECKING")))))
    return cst.Module(stmts).code.lstrip('\n') if stmts else None


def _split(s: str) -> tuple[str, str]:
    parts = s.split('.')
    return parts[0], '.'.join(parts[1:])


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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
                baz: FuncAnnotation(
                    [
                        (ArgumentName('z'), Typename('int'))
                    ],
                    None
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
                bar: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
                baz: FuncAnnotation(
                    [
                        (ArgumentName('z'), Typename('int'))
                    ],
                    Typename('float')
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('float'))
                    ],
                    Typename('float')
                ),
                bar: FuncAnnotation(
                    [
                        (ArgumentName('z'), Typename('int'))
                    ],
                    Typename('float')
                ),
            },
            override_annotations=False,
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


def test_override_annotations():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x: int) -> int:
            return x/2

        class C:
            def bar(self, x: float) -> int:
                return x/2
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    bar = FuncInfo(Filename('foo.py'), FunctionName('C.bar'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('float'))
                    ],
                    Typename('float')
                ),
                bar: FuncAnnotation(
                    [
                        (ArgumentName('self'), Typename('typing.Self')),
                        (ArgumentName('x'), Typename('int'))
                    ],
                    Typename('float')
                ),
            },
            override_annotations=True,
            module_name='foo',
            module_names=['foo']
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: float) -> float:
            return x/2
    """)

    assert get_function(code, 'C.bar') == textwrap.dedent("""\
        def bar(self: Self, x: int) -> float:
            return x/2
    """)


def test_transform_adds_typing_import_for_typing_names():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('typing.Optional[int]'))
                    ],
                    Typename('list[typing.Never]')
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('x.y.WholeNumber|None'))
                    ],
                    Typename('x.z.FloatingPointNumber')
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('x.y.WholeNumber|None'))
                    ],
                    Typename('x.z.FloatingPointNumber')
                )
            },
            override_annotations=False,
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

    code_str = str(code.code).strip()
    assert code_str.startswith("from __future__ import annotations")


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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    None
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    None
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                    ],
                    Typename('float')
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('foo.F')
                ),
                f_foo: FuncAnnotation(
                    [
                        (ArgumentName('v'), Typename('float')),
                    ],
                    Typename('foo.F')
                ),
                bar: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('foo.F')
                )
            },
            override_annotations=False,
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('x.y.z')),
                        (ArgumentName('y'), Typename('y.T')),
                        (ArgumentName('z'), Typename('a.b.c.T')),
                    ],
                    Typename('r.t.T')
                ),
            },
            override_annotations=False,
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

        with something():
            import a

        def foo(x, y): ...

        import r    # imported after 'def foo', so can't be used in annotation
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('x.y.z')),
                        (ArgumentName('y'), Typename('a.T')),
                    ],
                    Typename('r.t.T')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'x',
                'x.y',
                'r'
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: x.y.z, y: a.T) -> "r.t.T": ...
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('x.y.z')),
                    ],
                    Typename('a.b')
                ),
            },
            override_annotations=False,
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


def test_existing_typing_imports():
    code = cst.parse_module(textwrap.dedent("""\
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import m
            import ast

        def foo(x): ...

        def bar(x: "ast.For") -> "m.T": ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('ast.If')),
                    ],
                    Typename('typing.Any')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'ast',
                'm',
                'typing',
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: "ast.If") -> Any: ...
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import m
            import ast
            import ast
    """)

    code_str = str(code.code)
    assert code_str.startswith(textwrap.dedent("""\
        from typing import TYPE_CHECKING, Any
        if TYPE_CHECKING:
            import m
    """))


def test_inserts_imports_after_docstring_and_space():
    code = cst.parse_module(textwrap.dedent("""\
        '''blah blah blah
        blah
        '''


        from __future__ import annotations

        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('ast.If')),
                    ],
                    Typename('typing.Any')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'ast',
                'typing',
            ]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: ast.If) -> Any: ...
    """)

    print(code.code)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import ast
    """)

    code_str = str(code.code)
    assert code_str.startswith(textwrap.dedent("""\
        '''blah blah blah
        blah
        '''


        from __future__ import annotations
        from typing import TYPE_CHECKING, Any
        if TYPE_CHECKING:
            import ast
    """))


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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('pkg.b.T')),
                        (ArgumentName('y'), Typename('pkg.a.c.T')),
                        (ArgumentName('z'), Typename('pkg.a.c.X')),
                    ],
                    None
                ),
            },
            override_annotations=False,
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
                foobar: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('m.n.T')),
                    ],
                    Typename('None')
                ),
                Cfoo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('n.o.T')),
                    ],
                    Typename('None')
                ),
                Dfoo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('n.o.T')),
                    ],
                    Typename('None')
                ),
                f : FuncAnnotation(
                    [
                        (ArgumentName('a'), Typename('m.n.T')),
                        (ArgumentName('b'), Typename('n.o.T')),
                    ],
                    Typename('None')
                ),
            },
            override_annotations=False,
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


def test_nonglobal_imported_modules_are_ignored():
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a.T')),
                        (ArgumentName('y'), Typename('a.b.T')),
                    ],
                    Typename('a.c.T')
                ),
                bar: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('m.T')),
                    ],
                    Typename('m.T')
                ),
            },
            override_annotations=False,
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


def test_nonglobal_assignments_are_ignored():
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
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a.T')),
                    ],
                    Typename('')
                ),
                bar: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('m.T')),
                    ],
                    Typename('')
                ),
            },
            override_annotations=False,
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


def test_if_type_checking_insertion():
    code = cst.parse_module(textwrap.dedent("""\
        from a import b
        from typing import Any

        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('c.T')),
                    ],
                    Typename('None')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'c',
                'typing'
            ]
        )

    code = code.visit(t)

    assert str(code.code).startswith(textwrap.dedent("""\
        from typing import TYPE_CHECKING, Any
        if TYPE_CHECKING:
            import c
    """))


def test_import_conflicts_with_import():
    code = cst.parse_module(textwrap.dedent("""\
        from a import b as a
        import b as c

        def foo(x, y): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a.T')),
                        (ArgumentName('y'), Typename('c.d.e.T')),
                    ],
                    None
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'b',
                'c',
                'c.d'
            ]
        )

    code = code.visit(t)

    m = assert_regex(r'def foo\(x: "(.*?)", y: "(.*?)"\): ...', get_function(code, 'foo'))
    m1, t1 = _split(m.group(1))
    m2, t2 = _split(m.group(2))

    assert m1 != 'a'
    assert t1 == 'T'

    assert m2 != 'b'
    assert t2 == 'e.T'

    m1, m2 = sorted((m1, m2))

    print(code.code)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import a as {m1}
            import c.d as {m2}
    """)


def test_import_conflicts_with_definitions():
    code = cst.parse_module(textwrap.dedent("""\
        class a: ...
        def c(): pass

        def foo(x, y): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a.T')),
                        (ArgumentName('y'), Typename('c.d.e.T')),
                    ],
                    None
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'b',
                'c',
                'c.d'
            ]
        )

    code = code.visit(t)

    m = assert_regex(r'def foo\(x: "(.*?)", y: "(.*?)"\): ...', get_function(code, 'foo'))
    m1, t1 = _split(m.group(1))
    m2, t2 = _split(m.group(2))

    assert m1 != 'a'
    assert t1 == 'T'

    assert m2 != 'b'
    assert t2 == 'e.T'

    m1, m2 = sorted((m1, m2))

    print(code.code)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import a as {m1}
            import c.d as {m2}
    """)


def test_import_conflicts_with_assignments():
    code = cst.parse_module(textwrap.dedent("""\
        a, b = (10, 20)

        def foo(x, y): ...

        c: int = 10
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a.T')),
                        (ArgumentName('y'), Typename('c.d.e.T')),
                    ],
                    None
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
                'b',
                'c',
                'c.d'
            ]
        )

    code = code.visit(t)

    m = assert_regex(r'def foo\(x: "(.*?)", y: "(.*?)"\): ...', get_function(code, 'foo'))
    m1, t1 = _split(m.group(1))
    m2, t2 = _split(m.group(2))

    assert m1 != 'a'
    assert t1 == 'T'

    assert m2 != 'b'
    assert t2 == 'e.T'

    m1, m2 = sorted((m1, m2))

    print(code.code)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import a as {m1}
            import c.d as {m2}
    """)


def test_import_conflicts_with_with():
    code = cst.parse_module(textwrap.dedent("""\
        with my_handler() as a:
            pass

        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a.T')),
                    ],
                    None
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
            ]
        )

    code = code.visit(t)

    m = assert_regex(r'def foo\(x: "(.*?)"\): ...', get_function(code, 'foo'))
    m1, t1 = _split(m.group(1))

    assert m1 != 'a'
    assert t1 == 'T'

    print(code.code)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import a as {m1}
    """)


def test_import_conflicts_alias_for_module():
    code = cst.parse_module(textwrap.dedent("""\
        a, b = (10, 20)

        def foo(x): ...
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: FuncAnnotation(
                    [
                        (ArgumentName('x'), Typename('a')), # module "a" meant here, not something in it
                    ],
                    None
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'a',
            ]
        )

    code = code.visit(t)

    m = assert_regex(r'def foo\(x: "(.*?)"\): ...', get_function(code, 'foo'))
    m1, t1 = _split(m.group(1))

    assert m1 != 'a'
    assert t1 == ''

    print(code.code)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import a as {m1}
    """)


def test_builtin_name_conflicts():
    code = cst.parse_module(textwrap.dedent("""\
    def int():
        pass

    class C:
        @property
        def tuple(self):
            pass

        def f(self):
            pass
    """))

    f = FuncInfo(Filename('foo.py'), FunctionName('C.f'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: FuncAnnotation(
                    [
                    ],
                    Typename('tuple[int, float]')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'builtins',
            ]
        )

    code = code.visit(t)

    assert get_function(code, 'C.f') == textwrap.dedent("""\
        def f(self) -> "builtins.tuple[builtins.int, float]":
            pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import builtins
    """)


def test_class_names_dont_affect_body_of_methods():
    code = cst.parse_module(textwrap.dedent("""\
    tuple = 0

    class C:
        int = 0 # doesn't affect 'g'

        def f(self):
            def g(x):
                pass
    """))

    g = FuncInfo(Filename('foo.py'), FunctionName('C.f.<locals>.g'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                g: FuncAnnotation(
                    [
                    ],
                    Typename('tuple[int]')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'builtins',
            ]
        )

    code = code.visit(t)

    assert get_function(code, 'C.f.<locals>.g') == textwrap.dedent("""\
        def g(x) -> "builtins.tuple[int]":
            pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import builtins
    """)


def test_inner_function():
    code = cst.parse_module(textwrap.dedent("""\
    tuple = 0

    class C:
        def f(self):
            int = 0

            def g(x):
                def h(x):
                    pass

            class D:
                float = 0

                def i(self):
                    def j(x):
                        pass
    """))

    g = FuncInfo(Filename('foo.py'), FunctionName('C.f.<locals>.g'))
    h = FuncInfo(Filename('foo.py'), FunctionName('C.f.<locals>.g.<locals>.h'))
    i = FuncInfo(Filename('foo.py'), FunctionName('C.f.<locals>.D.i'))
    j = FuncInfo(Filename('foo.py'), FunctionName('C.f.<locals>.D.i.<locals>.j'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                g: FuncAnnotation([], Typename('tuple[int, float]')),
                h: FuncAnnotation([], Typename('tuple[int, float]')),
                i: FuncAnnotation([], Typename('tuple[int, float]')),
                j: FuncAnnotation([], Typename('tuple[int, float]')),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'builtins',
            ]
        )

    code = code.visit(t)

    assert get_function(code, 'C.f.<locals>.g') == textwrap.dedent("""\
        def g(x) -> "builtins.tuple[builtins.int, float]":
            def h(x) -> "builtins.tuple[builtins.int, float]":
                pass
    """)

    assert get_function(code, 'C.f.<locals>.D.i') == textwrap.dedent("""\
        def i(self) -> "builtins.tuple[builtins.int, builtins.float]":
            def j(x) -> "builtins.tuple[builtins.int, float]":
                pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import builtins
    """)


def test_builtin_name_conflicts_even_module_name():
    code = cst.parse_module(textwrap.dedent("""\
    builtins = 'even this!'

    def int():
        pass

    class C:
        @property
        def tuple(self):
            pass

        def f(self):
            pass
    """))

    f = FuncInfo(Filename('foo.py'), FunctionName('C.f'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: FuncAnnotation(
                    [
                    ],
                    Typename('tuple[int, float]')
                ),
            },
            override_annotations=False,
            module_name = 'foo',
            module_names = [
                'foo',
                'builtins',
            ]
        )

    code = code.visit(t)

    m = assert_regex(r'def f\(self\) -> \"(.*?)\":', get_function(code, 'C.f'))
    types = types_in_annotation(cst.parse_expression(m.group(1)))

    assert 'float' in types # no conflict

    for typ in types - {'float'}:
        assert typ not in ('int', 'tuple'), f"{typ=}"

        mod, rest = _split(typ)
        assert mod not in ('', 'builtins'), f"{typ=}"
        assert rest in ('int', 'tuple'), f"{typ=}"

    assert get_if_type_checking(code) == textwrap.dedent(f"""\
        if TYPE_CHECKING:
            import builtins as {mod}
    """)


def test_used_names():
    code = cst.parse_module(textwrap.dedent("""\
    a, b = 0, 0
    c: int = (d := 0)

    class C:
        e = 0

        class D:
            def f(self):
                pass

        def g(self):
            h = 0
            def i(self): pass

    def j():
        pass

    with handler as (k,):
        pass

    with handler as [l]:
        pass
    """))

    assert {'a', 'b', 'c', 'd', 'C', 'j', 'k', 'l'} == used_names(code)

    C = typing.cast(cst.ClassDef, cstm.findall(code, cstm.ClassDef(name=cstm.Name('C')))[0])
    assert {'D', 'e', 'g'} == used_names(C)

    D = typing.cast(cst.ClassDef, cstm.findall(code, cstm.ClassDef(name=cstm.Name('D')))[0])
    assert {'f'} == used_names(D)

    f = typing.cast(cst.FunctionDef, cstm.findall(code, cstm.FunctionDef(name=cstm.Name('f')))[0])
    assert set() == used_names(f)

    g = typing.cast(cst.FunctionDef, cstm.findall(code, cstm.FunctionDef(name=cstm.Name('g')))[0])
    assert {'h', 'i'} == used_names(g)

    i = typing.cast(cst.FunctionDef, cstm.findall(code, cstm.FunctionDef(name=cstm.Name('i')))[0])
    assert set() == used_names(i)


def test_types_in_annotation():
    def get_types(s):
        return types_in_annotation(cst.parse_expression(s))

    assert {'int'} == get_types('int')
    assert {'Tuple', 'int', 'float'} == get_types('Tuple[int, float]')
    assert {'Dict', 'foo.bar', 'bar.baz'} == get_types('Dict[foo.bar, bar.baz]')
    assert {'Union', 'int', 'float', 'Tuple', 'a.b.c'} == get_types('Union[int, float, Tuple[int, a.b.c]]')
