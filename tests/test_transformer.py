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
            not_annotated = dict(),
            imports = set()
        )

    code.visit(t)


def get_function(m: cst.Module, name: str) -> T.Optional[str]:
    class V(cst.CSTVisitor):
        def __init__(self):
            self.found = None

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            if node.name.value == name:
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
            imports = set()
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

    # XXX check import


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
                        (ArgumentName('x'), Typename('Integer')),
                        (ArgumentName('y'), Typename('WholeNumber'))
                    ],
                    Typename('FloatingPointNumber')
                )
            },
            not_annotated = {
                foo: {ArgumentName('x'), ArgumentName('y'), ArgumentName('return')}
            },
            imports = set(),
            allowed_types = [Typename('Integer')]
        )

    code = code.visit(t)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: Integer, y: "WholeNumber") -> "FloatingPointNumber":
            return x/2
    """)


def test_types_in_annotation():
    assert {'int'} == types_in_annotation('int')
    assert {'Tuple', 'int', 'float'} == types_in_annotation('Tuple[int, float]')
    assert {'Dict', 'foo.bar', 'bar.baz'} == types_in_annotation('Dict[foo.bar, bar.baz]')
    assert {'Union', 'int', 'float', 'Tuple', 'a.b.c'} == types_in_annotation('Union[int, float, Tuple[int, a.b.c]]')
