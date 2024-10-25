import libcst as cst
import textwrap
from righttyper.unified_transformer import UnifiedTransformer, types_in_annotation
from righttyper.righttyper_types import FuncInfo, Filename, FunctionName, Typename, ArgumentName
import typing as T


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
            not_annotated = dict()
        )

    code.visit(t)


def get_function(m: cst.Module, name: str) -> T.Optional[str]:
    class V(cst.CSTVisitor):
        def __init__(self):
            self.found = None
            self.class_stack = []

        def visit_ClassDef(self, node: cst.ClassDef):
            self.class_stack.append(node.name.value)

        def leave_ClassDef(self, node: cst.ClassDef):
            self.class_stack.pop()

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            qual_name = ".".join([*self.class_stack, node.name.value])
            if qual_name == name:
                self.found = node
                return False # stop here
            return True

    v = V()
    m.visit(v)
    return cst.Module([v.found]).code.lstrip('\n') if v.found else None


def get_if_type_checking(m: cst.Module) -> T.Optional[str]:
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
            }
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
            }
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
            }
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: float) -> float:
            def bar(z: int) -> float:
                return z/2
            return bar(x+y)
    """)

    assert get_if_type_checking(code) == None


def test_transform_adds_typing_import_for_typing_name():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x):
            return x/2 if x else 0.0
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [
                        (ArgumentName('x'), Typename('Optional[int]'))   # Optional comes from typing module
                    ],
                    Typename('float')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x')},
            }
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: Optional[int]):
            return x/2 if x else 0.0
    """)

    assert "from typing import " in code.code

    assert get_if_type_checking(code) == None


def test_transform_adds_typing_import_for_typing_name_return():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x):
            return x/2 if x else None
    """))

    foo = FuncInfo(Filename('foo.py'), FunctionName('foo'))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: (
                    [],
                    Typename('Optional[float]') # Optional comes from typing module
                )
            },
            not_annotated = {
                foo: {ArgumentName('return')},
            }
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x) -> Optional[float]:
            return x/2 if x else None
    """)

    # also check only needed names are emitted
    assert "from typing import TYPE_CHECKING, Optional\n" in code.code
    assert get_if_type_checking(code) == None


def test_transform_function_as_string():
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
                        (ArgumentName('y'), Typename('Optional[X.Y.WholeNumber]'))
                    ],
                    Typename('X.Z.FloatingPointNumber')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')}
            }
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: "Optional[X.Y.WholeNumber]") -> "X.Z.FloatingPointNumber":
            return x/2
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import X.Y
            import X.Z
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
            }
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
            }
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
            }
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
                    Typename('F')
                ),
                f_foo: (
                    [
                        (ArgumentName('v'), Typename('float')),
                    ],
                    Typename('F')
                ),
                bar: (
                    [
                        (ArgumentName('x'), Typename('int')),
                        (ArgumentName('y'), Typename('int'))
                    ],
                    Typename('F')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')},
                f_foo: {ArgumentName('v'), ArgumentName('return')},
                bar: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')}
            }
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


def test_types_in_annotation():
    assert {'int'} == types_in_annotation('int')
    assert {'Tuple', 'int', 'float'} == types_in_annotation('Tuple[int, float]')
    assert {'Dict', 'foo.bar', 'bar.baz'} == types_in_annotation('Dict[foo.bar, bar.baz]')
    assert {'Union', 'int', 'float', 'Tuple', 'a.b.c'} == types_in_annotation('Union[int, float, Tuple[int, a.b.c]]')
