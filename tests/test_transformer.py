import libcst as cst
import libcst.matchers as cstm
from libcst.metadata import MetadataWrapper, PositionProvider
import textwrap
from righttyper.unified_transformer import UnifiedTransformer, types_in_annotation
from righttyper.typeinfo import TypeInfo, NoneTypeInfo, AnyTypeInfo
from righttyper.righttyper_types import CodeId, Filename, FunctionName, ArgumentName, VariableName
from righttyper.annotation import FuncAnnotation, ModuleVars
from righttyper.type_id import get_type_name
import typing
import pytest
import re


def find_function(m: cst.Module, name: str) -> tuple[cst.FunctionDef, int]|None:
    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

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
                first_line = min(
                    self.get_metadata(PositionProvider, node).start.line
                    for node in (node, *node.decorators)
                )

                self.found = (node, first_line)
                return False # stop here
            return True

        def leave_FunctionDef(self, node: cst.FunctionDef):
            self.name_stack.pop()
            self.name_stack.pop()

    wrapper = MetadataWrapper(m)
    v = V()
    wrapper.visit(v)

    return v.found


def find_function_all(m: cst.Module, name: str) -> list[tuple[cst.FunctionDef, int]]:
    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self):
            self.found = []
            self.name_stack = []

        def visit_ClassDef(self, node: cst.ClassDef):
            self.name_stack.append(node.name.value)

        def leave_ClassDef(self, node: cst.ClassDef):
            self.name_stack.pop()

        def visit_FunctionDef(self, node: cst.FunctionDef):
            self.name_stack.append(node.name.value)
            qual_name = ".".join(self.name_stack)
            self.name_stack.append("<locals>")
            if qual_name == name:
                first_line = min(
                    self.get_metadata(PositionProvider, node).start.line
                    for node in (node, *node.decorators)
                )

                self.found.append((node, first_line))

        def leave_FunctionDef(self, node: cst.FunctionDef):
            self.name_stack.pop()
            self.name_stack.pop()

    wrapper = MetadataWrapper(m)
    v = V()
    wrapper.visit(v)

    return v.found


def get_function(m: cst.Module, funcname: str, body=True) -> str|None:
    """Returns the given function as a string, if found in 'm'"""
    if (f := find_function(m, funcname)):
        if body:
            return cst.Module([f[0]]).code.lstrip('\n')

        return cst.Module([
                f[0].with_changes(
                    body=cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())]),
                    leading_lines=[]
                )
            ]).code

    return None


def get_function_all(m: cst.Module, funcname: str, body=True) -> list[str]:
    """Returns the given function as a string, if found in 'm'"""
    strings: list[str] = []
    for f, line_no in find_function_all(m, funcname):
        if body:
            strings.append(cst.Module([f]).code.lstrip('\n'))
        else:
            strings.append(
                cst.Module([f.with_changes(
                    body=cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())]),
                    leading_lines=[])]).code)
    return strings


def get_code_id(filename: str, m: cst.Module, funcname: str) -> CodeId:
    """Returns a CodeId for the given function, if found in 'm'"""
    if (f := find_function(m, funcname)):
        return CodeId(
            Filename(filename),
            FunctionName(funcname),
            f[1],
            0
        )

    raise RuntimeError(f"Unable to find {funcname}")


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


def _mkAnnotation(args: list[tuple[ArgumentName, TypeInfo]], retval: TypeInfo) -> FuncAnnotation:
    return FuncAnnotation(args, retval, varargs=None, kwargs=None, variables=[])


def test_transform_function():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            return (x+y)/2

        def bar(x):
            return x/2

        def baz(z):
            return z/2
    """))

    foo = get_code_id('foo.py', code, 'foo')
    baz = get_code_id('foo.py', code, 'baz')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
                baz: _mkAnnotation(
                    [
                        (ArgumentName('z'), TypeInfo.from_type(int, module=''))
                    ],
                    NoneTypeInfo,
                )
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y) -> float:
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x):
            return x/2
    """)

    assert get_function(code, 'baz') == textwrap.dedent("""\
        def baz(z: int) -> None:
            return z/2
    """)

    assert get_if_type_checking(code) is None

    changes = sorted(t.get_changes())
    it = iter(changes)

    name, old, new = next(it)
    assert name == 'baz'
    assert old == 'def baz(z):'
    assert new == 'def baz(z: int) -> None:'

    name, old, new = next(it)
    assert name == 'foo'
    assert old == 'def foo(x, y):'
    assert new == 'def foo(x: int, y) -> float:'

    assert next(it, None) is None


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

    foo = get_code_id('foo.py', code, 'C.foo')
    bar = get_code_id('foo.py', code, 'C.bar')
    baz = get_code_id('foo.py', code, 'C.baz')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
                bar: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
                baz: _mkAnnotation(
                    [
                        (ArgumentName('z'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                )
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    assert get_if_type_checking(code) is None

    changes = sorted(t.get_changes())
    it = iter(changes)

    name, old, new = next(it)
    assert name == 'C.bar'
    assert old == 'def bar(x):'
    assert new == 'def bar(x: int) -> float:'

    name, old, new = next(it)
    assert name == 'C.baz'
    assert old == 'def baz(cls, z):'
    assert new == 'def baz(cls, z: int) -> float:'

    name, old, new = next(it)
    assert name == 'C.foo'
    assert old == 'def foo(self, x, y):'
    assert new == 'def foo(self, x: int, y) -> float:'

    assert next(it, None) is None

def test_transform_local_function():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            def bar(z):
                return z/2
            return bar(x+y)
    """))

    foo = get_code_id('foo.py', code, 'foo')
    bar = get_code_id('foo.py', code, 'foo.<locals>.bar')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_type(float, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
                bar: _mkAnnotation(
                    [
                        (ArgumentName('z'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: float) -> float:
            def bar(z: int) -> float:
                return z/2
            return bar(x+y)
    """)

    assert get_if_type_checking(code) is None


def test_override_annotations():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x: int) -> int:
            return x/2

        class C:
            def bar(self, x: float) -> int:
                return x/2
    """))

    foo = get_code_id('foo.py', code, 'foo')
    bar = get_code_id('foo.py', code, 'C.bar')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(float, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
                bar: _mkAnnotation(
                    [
                        (ArgumentName('self'), TypeInfo(module='typing', name='Self')),
                        (ArgumentName('x'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo.from_type(float, module=''),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='typing', name='Optional', args=(
                            TypeInfo.from_type(int, module=''),
                            )
                        ))
                    ],
                    TypeInfo.from_type(list, module='', args=(TypeInfo(module='typing', name='Never'),)),
                )
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: Optional[int]) -> list[Never]: ...
    """)

    code_str = str(code.code)
    assert re.search(r"^ *from typing import .*\bOptional\b", code_str)
    assert re.search(r"^ *from typing import .*\bNever\b", code_str)
    assert get_if_type_checking(code) is None


def test_transform_unknown_type_as_string():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y):
            return x/2
    """))

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_set({
                            TypeInfo(module='x.y', name='Something', args=('quoted',)),
                            NoneTypeInfo
                        }))
                    ],
                    TypeInfo(module='x', name='z.FloatingPointNumber'),
                )
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: "x.y.Something[\\"quoted\\"]|None") -> "x.z.FloatingPointNumber":
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_set({
                            TypeInfo(module='x.y', name='WholeNumber'),
                            NoneTypeInfo
                        }))
                    ],
                    TypeInfo(module='x', name='z.FloatingPointNumber'),
                )
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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


def test_transform_empty_body_but_from_future():
    code = cst.parse_module(textwrap.dedent("""\
        from __future__ import annotations
    """))

    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {},
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    assert str(code.code) == textwrap.dedent("""\
        from __future__ import annotations
    """)


def test_transform_deletes_type_hint_comments_in_header():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(x, y): # foo boo # type: (int, int) -> Any
            return (x+y)/2

        def bar(x):   # type: (Any) -> None
            pass
    """))

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_type(int, module=''))
                    ],
                    NoneTypeInfo,
                )
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: int, y: int) -> None: # foo boo
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x):   # type: (Any) -> None
            pass
    """)

    assert get_if_type_checking(code) is None


def test_transform_deletes_type_hint_comments_in_parameters():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(
            x,  # typert...  # type: int
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_type(int, module=''))
                    ],
                    NoneTypeInfo,
                )
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(
            x: int,  # typert...
            y: int
        ) -> None:
            return (x+y)/2
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(
            x   # type: Any
        ):
            # type: (...) -> None
            pass
    """)

    assert get_if_type_checking(code) is None


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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                    ],
                    TypeInfo.from_type(float, module=''),
                )
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    assert get_if_type_checking(code) is None


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

    foo = get_code_id('foo.py', code, 'foo')
    f_foo = get_code_id('foo.py', code, 'F.foo')
    bar = get_code_id('foo.py', code, 'bar')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo(module='foo', name='F'),
                ),
                f_foo: _mkAnnotation(
                    [
                        (ArgumentName('v'), TypeInfo.from_type(float, module='')),
                    ],
                    TypeInfo(module='foo', name='F'),
                ),
                bar: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(int, module='')),
                        (ArgumentName('y'), TypeInfo.from_type(int, module=''))
                    ],
                    TypeInfo(module='foo', name='F'),
                )
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    assert get_if_type_checking(code) is None


def test_uses_imported_aliases():
    code = cst.parse_module(textwrap.dedent("""\
        from x.y import z as zed
        from y import T
        if True:
            import a.b as A

        def foo(x, y, z): ...

        import r    # imported after 'def foo', so can't be used in annotation
    """))

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='x.y', name='z')),
                        (ArgumentName('y'), TypeInfo(module='y', name='T')),
                        (ArgumentName('z'), TypeInfo(module='a.b', name='c.T'))
                    ],
                    TypeInfo(module='r', name='t.T'),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: zed, y: T, z: A.c.T) -> "_rt_r.t.T": ...
    """)


    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import r as _rt_r
    """)


def test_uses_imported_domains():
    code = cst.parse_module(textwrap.dedent("""\
        if True:
            import x.y

        with something():
            import a

        def foo(x, y): ...
    """))

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='x.y', name='z')),
                        (ArgumentName('y'), TypeInfo(module='a', name='T'))
                    ],
                    TypeInfo(module='r', name='t.T'),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='x.y', name='z'))
                    ],
                    TypeInfo(module='a', name='b'),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    import ast

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(ast.If))
                    ],
                    TypeInfo(module='typing', name='List'),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: "ast.If") -> List: ...
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import m
            import ast
            import ast
    """)

    code_str = str(code.code)
    assert code_str.startswith(textwrap.dedent("""\
        from typing import TYPE_CHECKING, List
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

    import ast

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo.from_type(ast.If))
                    ],
                    TypeInfo(module='typing', name='List'),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: ast.If) -> List: ...
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
        from typing import TYPE_CHECKING, List
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='pkg.b', name='T')),
                        (ArgumentName('y'), TypeInfo(module='pkg.a.c', name='T')),
                        (ArgumentName('z'), TypeInfo(module='pkg.a.c', name='X')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'pkg.a.a',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: b.T, y: c.T, z: X) -> None: ...
    """)

    assert get_if_type_checking(code) is None


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

    foobar = get_code_id('foo.py', code, 'foo.<locals>.bar')
    Cfoo = get_code_id('foo.py', code, 'C.foo')
    Dfoo = get_code_id('foo.py', code, 'C.D.foo')
    f = get_code_id('foo.py', code, 'f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foobar: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='m.n', name='T')),
                    ],
                    NoneTypeInfo,
                ),
                Cfoo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='n', name='o.T')),
                    ],
                    NoneTypeInfo,
                ),
                Dfoo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='n', name='o.T')),
                    ],
                    NoneTypeInfo,
                ),
                f : _mkAnnotation(
                    [
                        (ArgumentName('a'), TypeInfo(module='m.n', name='T')),
                        (ArgumentName('b'), TypeInfo(module='n', name='o.T')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    foo = get_code_id('foo.py', code, 'foo')
    bar = get_code_id('foo.py', code, 'bar')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='a', name='T')),
                        (ArgumentName('y'), TypeInfo(module='a.b', name='T'))
                    ],
                    TypeInfo(module='a.c', name='T'),
                ),
                bar: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='m', name='T')),
                    ],
                    TypeInfo(module='m', name='T'),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
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

    foo = get_code_id('foo.py', code, 'foo')
    bar = get_code_id('foo.py', code, 'bar')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='a', name='T')),
                    ],
                    NoneTypeInfo,
                ),
                bar: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='m', name='T')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)
    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo(x: "a.T") -> None:
            a = Any
    """)

    assert get_function(code, 'bar') == textwrap.dedent("""\
        def bar(x: "m.T") -> None:
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='c', name='T'))
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='a', name='T')),
                        (ArgumentName('y'), TypeInfo(module='c.d', name='e.T')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    m = assert_regex(r'def foo\(x: "(.*?)", y: "(.*?)"\) -> None: ...', get_function(code, 'foo'))
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='a', name='T')),
                        (ArgumentName('y'), TypeInfo(module='c.d', name='e.T')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    m = assert_regex(r'def foo\(x: "(.*?)", y: "(.*?)"\) -> None: ...', get_function(code, 'foo'))
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='a', name='T')),
                        (ArgumentName('y'), TypeInfo(module='c.d', name='e.T')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    m = assert_regex(r'def foo\(x: "(.*?)", y: "(.*?)"\) -> None: ...', get_function(code, 'foo'))
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

    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo(module='a', name='T')),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    m = assert_regex(r'def foo\(x: "(.*?)"\) -> None: ...', get_function(code, 'foo'))
    m1, t1 = _split(m.group(1))

    assert m1 != 'a'
    assert t1 == 'T'

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

    f = get_code_id('foo.py', code, 'C.f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                    ],
                    TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int, module=''), TypeInfo.from_type(float, module=''))),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    assert get_function(code, 'C.f') == textwrap.dedent("""\
        def f(self) -> "builtins.tuple[builtins.int, float]":
            pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
        if TYPE_CHECKING:
            import builtins
    """)


def test_arg_name_conflicts():
    code = cst.parse_module(textwrap.dedent("""\
    import numpy
    from numpy import dtype, int64

    class C:
        def dtype(self):
            return None

        def f(self, x):
            pass
    """))

    f = get_code_id('foo.py', code, 'C.f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo('numpy', 'dtype', args=(TypeInfo('numpy', 'int64'),)))
                    ],
                    NoneTypeInfo
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
        )

    code = t.transform_code(code)

    assert get_function(code, 'C.f') == textwrap.dedent("""\
        def f(self, x: numpy.dtype[int64]) -> None:
            pass
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

    g = get_code_id('foo.py', code, 'C.f.<locals>.g')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                g: _mkAnnotation(
                    [
                    ],
                    TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int, module=''),)),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    assert get_function(code, 'C.f.<locals>.g') == textwrap.dedent("""\
        def g(x) -> "builtins.tuple[int]":
            pass
    """)

    assert get_if_type_checking(code) == textwrap.dedent("""\
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

    g = get_code_id('foo.py', code, 'C.f.<locals>.g')
    h = get_code_id('foo.py', code, 'C.f.<locals>.g.<locals>.h')
    i = get_code_id('foo.py', code, 'C.f.<locals>.D.i')
    j = get_code_id('foo.py', code, 'C.f.<locals>.D.i.<locals>.j')
    tuple_int_float = TypeInfo.from_type(tuple, args=(
        TypeInfo.from_type(int, module=''),
        TypeInfo.from_type(float, module='')
    ))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                g: _mkAnnotation([], tuple_int_float),
                h: _mkAnnotation([], tuple_int_float),
                i: _mkAnnotation([], tuple_int_float),
                j: _mkAnnotation([], tuple_int_float)
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

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

    assert get_if_type_checking(code) == textwrap.dedent("""\
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

    f = get_code_id('foo.py', code, 'C.f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                    ],
                    TypeInfo.from_type(tuple, args=(
                        TypeInfo.from_type(int, module=''),
                        TypeInfo.from_type(float, module='')
                    )),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

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


def test_types_in_annotation():
    def get_types(s):
        return types_in_annotation(cst.parse_expression(s))

    assert {'int'} == get_types('int')
    assert {'Tuple', 'int', 'float'} == get_types('Tuple[int, float]')
    assert {'Dict', 'foo.bar', 'bar.baz'} == get_types('Dict[foo.bar, bar.baz]')
    assert {'Union', 'int', 'float', 'Tuple', 'a.b.c'} == get_types('Union[int, float, Tuple[int, a.b.c]]')


def make_typevar(args: list, i: int) -> TypeInfo:
    return TypeInfo.from_set(
        {get_type_name(arg) for arg in args},
        typevar_index=i
    )


def test_generics_inline_simple():
    code = cst.parse_module(textwrap.dedent("""\
        def add(a, b):
            return a + b
    """))

    T1 = make_typevar([str, int], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                        (ArgumentName("b"), T1)
                    ],
                    T1,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    assert get_function(code, 'add') == textwrap.dedent("""\
        def add[T1: (int, str)](a: T1, b: T1) -> T1:
            return a + b
    """)


@pytest.mark.parametrize('override', [False, True])
def test_generics_arg_already_annotated(override):
    code = cst.parse_module(textwrap.dedent("""\
        def add(a, b: int|str):
            return a + b
    """))

    T1 = make_typevar([str, int], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                        (ArgumentName("b"), T1)
                    ],
                    T1,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=override,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    if override:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[T1: (int, str)](a: T1, b: T1) -> T1:
                return a + b
        """)
    else:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[T1: (int, str)](a: T1, b: int|str) -> T1:
                return a + b
        """)


@pytest.mark.parametrize('override', [False, True])
def test_generics_ret_already_annotated(override):
    code = cst.parse_module(textwrap.dedent("""\
        def add(a, b) -> int|str:
            return a + b
    """))

    T1 = make_typevar([str, int], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                        (ArgumentName("b"), T1)
                    ],
                    T1,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=override,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    if override:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[T1: (int, str)](a: T1, b: T1) -> T1:
                return a + b
        """)
    else:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[T1: (int, str)](a: T1, b: T1) -> int|str:
                return a + b
        """)


@pytest.mark.parametrize('override', [False, True])
def test_generics_existing_generics(override):
    code = cst.parse_module(textwrap.dedent("""\
        def add[X: (int, bool), Y: (str,), Z: (int,)](a, b: Y) -> Z:
            return a + b
    """))

    T1 = make_typevar([str, int], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                    ],
                    T1,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=override,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    if override:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[Y: (str,), T1: (int, str)](a: T1, b: Y) -> T1:
                return a + b
        """)
    else:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[Y: (str,), Z: (int,), T1: (int, str)](a: T1, b: Y) -> Z:
                return a + b
        """)


def test_generics_existing_unused_generics():
    code = cst.parse_module(textwrap.dedent("""\
        def add[X: (int, bool)](a, b):
            return a + b
    """))

    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), TypeInfo("", "int")),
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    assert get_function(code, 'add') == textwrap.dedent("""\
        def add(a: int, b) -> None:
            return a + b
    """)


@pytest.mark.parametrize('override', [False, True])
def test_generics_existing_generics_nested(override):
    code = cst.parse_module(textwrap.dedent("""\
        def add[X: (int, bool), Y: (str,), Z: (int,)](a, b: tuple[Y]) -> list[Z]:
            return a + b
    """))

    T1 = make_typevar([str, int], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                    ],
                    T1,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=override,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    if override:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[Y: (str,), T1: (int, str)](a: T1, b: tuple[Y]) -> T1:
                return a + b
        """)
    else:
        assert get_function(code, 'add') == textwrap.dedent("""\
            def add[Y: (str,), Z: (int,), T1: (int, str)](a: T1, b: tuple[Y]) -> list[Z]:
                return a + b
        """)


def test_generics_existing_generics_overlaps_name():
    # we don't (yet) attempt to merge inline generics
    code = cst.parse_module(textwrap.dedent("""\
        def add[T1: (bool,)](a, b: T1):
            return a + b
    """))

    T1 = make_typevar([str, int], 1)
    T2 = make_typevar([str, bool], 2)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                    ],
                    T2,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    assert get_function(code, 'add') == textwrap.dedent("""\
        def add[T1: (bool,), T2: (int, str), T3: (bool, str)](a: T2, b: T1) -> T3:
            return a + b
    """)


def test_generics_inline_multiple():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(a, b, c, d):
            pass
    """))

    T1 = make_typevar([int, str], 1)
    T2 = make_typevar([int, str], 2)
    f = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                        (ArgumentName("b"), T1),
                        (ArgumentName("c"), T2),
                        (ArgumentName("d"), T2)
                    ],
                    NoneTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    assert get_function(code, 'foo') == textwrap.dedent("""\
        def foo[T1: (int, str), T2: (int, str)](a: T1, b: T1, c: T2, d: T2) -> None:
            pass
    """)


def test_generics_inline_nested():
    code = cst.parse_module(textwrap.dedent("""\
        def add(a, b):
            return [a + _b for _b in b]
    """))

    T1 = make_typevar([int, str], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                        (ArgumentName("b"), TypeInfo("", "list", (T1,))),
                    ],
                    TypeInfo("", "list", (T1,)),
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=True
        )

    code = t.transform_code(code)

    assert get_function(code, 'add') == textwrap.dedent("""\
        def add[T1: (int, str)](a: T1, b: list[T1]) -> list[T1]:
            return [a + _b for _b in b]
    """)


def test_generics_defined_simple():
    code = cst.parse_module(textwrap.dedent("""\
        def add(a, b):
            return a + b
    """))

    T1 = make_typevar([int, str], 1)
    f = get_code_id('foo.py', code, 'add')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("a"), T1),
                        (ArgumentName("b"), T1),
                    ],
                    T1,
                ),
            },
            module_variables = ModuleVars([]),
            module_name = 'foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False
        )

    code = t.transform_code(code)

    assert 'rt_T1 = TypeVar("rt_T1", int, str)' in code.code
        
    assert get_function(code, 'add') == textwrap.dedent("""\
        def add(a: rt_T1, b: rt_T1) -> rt_T1:
            return a + b
    """)


def test_overload_preserve():
    input_code = cst.parse_module(textwrap.dedent("""\
        from typing import overload

        @overload
        def foo(bar: int) -> str:
            ...
        @overload
        def foo(bar: str) -> int:
            ...
        def foo(bar):
            if isinstance(bar, int):
                return "hello"
            elif isinstance(bar, str):
                return 2
            elif isinstance(bar, bool):
                return not bar
    """))
    foo = get_code_id('foo.py', input_code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName("bar"), TypeInfo.from_type(str, module="")),
                    ],
                    TypeInfo.from_type(int, module=""),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=False,
            only_update_annotations=False,
            inline_generics=False,
        )

    output_code = t.transform_code(input_code)
    assert input_code.code == output_code.code


def test_overload_remove():
    code = cst.parse_module(textwrap.dedent("""\
        from typing import overload

        @overload
        def foo(bar: int) -> str:
            ...
        @overload
        def foo(bar: str) -> int:
            ...
        def foo(bar):
            if isinstance(bar, int):
                return "hello"
            elif isinstance(bar, str):
                return 2
            elif isinstance(bar, bool):
                return not bar
    """))
    
    
    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName("bar"), TypeInfo.from_type(str, module="")),
                    ],
                    TypeInfo.from_type(int, module=""),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    functions = get_function_all(code, "foo")
    assert len(functions) == 1
    assert functions[0].strip() == textwrap.dedent("""
        def foo(bar: str) -> int:
            if isinstance(bar, int):
                return "hello"
            elif isinstance(bar, str):
                return 2
            elif isinstance(bar, bool):
                return not bar
        """).strip()


def test_overload_aliased():
    code = cst.parse_module(textwrap.dedent("""\
        import typing

        @typing.overload
        def foo(bar: int) -> str:
            ...
        @typing.overload
        def foo(bar: str) -> int:
            ...
        def foo(bar: int|str) -> int|str:
            if isinstance(bar, int):
                return "hello"
            elif isinstance(bar, str):
                return 2
    """))
    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName("bar"), TypeInfo.from_type(str, module="")),
                    ],
                    TypeInfo.from_type(int, module=""),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    functions = get_function_all(code, "foo")
    assert len(functions) == 1
    assert functions[0].strip() == textwrap.dedent("""
        def foo(bar: str) -> int:
            if isinstance(bar, int):
                return "hello"
            elif isinstance(bar, str):
                return 2
        """).strip()


def test_dont_annotate_with_any():
    code = cst.parse_module(textwrap.dedent("""\
        def foo(bar: int) -> float:
            ...
    """))
    foo = get_code_id('foo.py', code, 'foo')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                foo: _mkAnnotation(
                    [
                        (ArgumentName("bar"), AnyTypeInfo),
                    ],
                    AnyTypeInfo,
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    function = get_function(code, "foo")
    assert function == textwrap.dedent("""\
        def foo(bar):
            ...
        """)


def test_local_aliases_known():
    code = cst.parse_module(textwrap.dedent("""\
        class _C:
            pass

        D = _C

        def f(x):
            ...
    """))
    f = get_code_id('foo.py', code, 'f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("x"), TypeInfo("foo", "D")),
                    ],
                    TypeInfo("foo", "D"),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    function = get_function(code, "f")
    assert function == textwrap.dedent("""\
        def f(x: D) -> D:
            ...
        """)


def test_local_aliases_known_multiple():
    code = cst.parse_module(textwrap.dedent("""\
        class _C:
            pass

        E, D = None, _C

        def f(x):
            ...
    """))
    f = get_code_id('foo.py', code, 'f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("x"), TypeInfo("foo", "D")),
                    ],
                    TypeInfo("foo", "D"),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    function = get_function(code, "f")
    assert function == textwrap.dedent("""\
        def f(x: D) -> D:
            ...
        """)


def test_local_aliases_known_annotated():
    code = cst.parse_module(textwrap.dedent("""\
        from typing import TypeAlias

        class _C:
            pass

        D: TypeAlias = _C

        def f(x):
            ...
    """))
    f = get_code_id('foo.py', code, 'f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("x"), TypeInfo("foo", "D")),
                    ],
                    TypeInfo("foo", "D"),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    function = get_function(code, "f")
    assert function == textwrap.dedent("""\
        def f(x: D) -> D:
            ...
        """)


def test_local_aliases_known_namedexpr():
    code = cst.parse_module(textwrap.dedent("""\
        class _C:
            pass

        if (D := _C):
            pass

        def f(x):
            ...
    """))
    f = get_code_id('foo.py', code, 'f')
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                f: _mkAnnotation(
                    [
                        (ArgumentName("x"), TypeInfo("foo", "D")),
                    ],
                    TypeInfo("foo", "D"),
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    function = get_function(code, "f")
    assert function == textwrap.dedent("""\
        def f(x: D) -> D:
            ...
        """)


def mk_var_transformer(filename, code, *, override_annotations=True, only_update_annotations=False):
    try:
        C_init = get_code_id(filename, code, 'C.__init__')
    except:
        C_init = None

    return UnifiedTransformer(
            filename=filename,
            type_annotations = {
                C_init: FuncAnnotation(
                    [],
                    TypeInfo("", "None"),
                    varargs=None, kwargs=None,
                    variables=[
                        (VariableName('self.x'), TypeInfo('', 'int')),
                        (VariableName('self.y'), TypeInfo('', 'int'))
                    ]
                ),
            } if C_init else {},
            module_variables = ModuleVars([
                (VariableName('g'), TypeInfo("", "float")),
                (VariableName('C.x'), TypeInfo("", "str")),
                (VariableName('C.y'), TypeInfo("", "bool"))
            ]),
            module_name='foo',
            override_annotations=override_annotations,
            only_update_annotations=only_update_annotations,
            inline_generics=False,
        )


@pytest.mark.parametrize('ignore, update', [
    [False, False],
    [True, False],
    [False, True]
])
def test_variables_assign(ignore, update):
    code = cst.parse_module(textwrap.dedent("""\
        g = 1.0

        class C:
            x = 'foo'
            x = x + x

            def __init__(self, x):
                self.y = x
    """))

    t = mk_var_transformer('foo.py', code,
            override_annotations=ignore,
            only_update_annotations=update)

    code = t.transform_code(code)
    if not update:
        assert code.code == textwrap.dedent("""\
            g: float = 1.0

            class C:
                x: str = 'foo'
                x = x + x

                def __init__(self, x) -> None:
                    self.y: int = x
            """)
    else:
        assert code.code == textwrap.dedent("""\
            g = 1.0

            class C:
                x = 'foo'
                x = x + x

                def __init__(self, x):
                    self.y = x
            """)


@pytest.mark.parametrize('ignore, update', [
    [False, False],
    [True, False],
    [False, True]
])
def test_variables_annassign(ignore, update):
    code = cst.parse_module(textwrap.dedent("""\
        g: str = 1.0

        class C:
            x: int = 'foo'
            x = x + x

            def __init__(self, x):
                self.y: None = x
    """))

    t = mk_var_transformer('foo.py', code,
            override_annotations=ignore,
            only_update_annotations=update)

    code = t.transform_code(code)

    if ignore:
        assert code.code == textwrap.dedent("""\
            g: float = 1.0

            class C:
                x: str = 'foo'
                x = x + x

                def __init__(self, x) -> None:
                    self.y: int = x
            """)
    elif update:
        assert code.code == textwrap.dedent("""\
            g: float = 1.0

            class C:
                x: str = 'foo'
                x = x + x

                def __init__(self, x):
                    self.y: int = x
            """)
    else:
        assert code.code == textwrap.dedent("""\
            g: str = 1.0

            class C:
                x: int = 'foo'
                x = x + x

                def __init__(self, x) -> None:
                    self.y: None = x
            """)


def test_variables_assign_already_defined():
    code = cst.parse_module(textwrap.dedent(f"""\
        (a, g) = 0, 1
        g = 1.0

        class C:
            x = y = 0
            x = 'foo'

            def __init__(self, x):
                self.x, blah = 0, 1
                self.y, foo = 0, 1
                self.x = x
                self.y = x
        """))

    t = mk_var_transformer('foo.py', code)
    code = t.transform_code(code)

    assert code.code == textwrap.dedent(f"""\
        (a, g) = 0, 1
        g = 1.0

        class C:
            x = y = 0
            x = 'foo'

            def __init__(self, x) -> None:
                self.x, blah = 0, 1
                self.y, foo = 0, 1
                self.x = x
                self.y = x
        """)


@pytest.mark.parametrize("var, expr", [
    ("g", "range(2)"),
    ("g, h", "enumerate(range(2))")
])
def test_variables_assign_already_defined_for(var, expr):
    code = cst.parse_module(textwrap.dedent(f"""\
        for {var} in {expr}:
            pass
        g = 1.0
        """))

    t = mk_var_transformer('foo.py', code)
    code = t.transform_code(code)

    assert code.code == textwrap.dedent(f"""\
        for {var} in {expr}:
            pass
        g = 1.0
        """)


@pytest.mark.parametrize('ignore, update', [
    [False, False],
    [True, False],
    [False, True]
])
def test_variables_assign_already_declared(ignore, update):
    code = cst.parse_module(textwrap.dedent(f"""\
        g: None
        g = 1.0

        class C:
            x: None
            x = 'foo'

            def __init__(self, x):
                self.x: None
                self.x = x
                self.y = x
        """))

    t = mk_var_transformer('foo.py', code,
            override_annotations=ignore,
            only_update_annotations=update)
    code = t.transform_code(code)

    if ignore:
        assert code.code == textwrap.dedent(f"""\
            g: float
            g = 1.0

            class C:
                x: str
                x = 'foo'

                def __init__(self, x) -> None:
                    self.x: int
                    self.x = x
                    self.y: int = x
            """)
    elif update:
        assert code.code == textwrap.dedent(f"""\
            g: float
            g = 1.0

            class C:
                x: str
                x = 'foo'

                def __init__(self, x):
                    self.x: int
                    self.x = x
                    self.y = x
            """)
    else:
        assert code.code == textwrap.dedent(f"""\
            g: None
            g = 1.0

            class C:
                x: None
                x = 'foo'

                def __init__(self, x) -> None:
                    self.x: None
                    self.x = x
                    self.y: int = x
            """)


@pytest.mark.parametrize('from_zoo', ['', 'from zoo '])
def test_variables_already_defined_import(from_zoo):
    code = cst.parse_module(textwrap.dedent(f"""\
        {from_zoo}import g
        g = 1.0

        class C:
            {from_zoo}import elephant as x
            x = 'foo'
        """))

    t = mk_var_transformer('foo.py', code)
    code = t.transform_code(code)

    assert code.code == textwrap.dedent(f"""\
        {from_zoo}import g
        g = 1.0

        class C:
            {from_zoo}import elephant as x
            x = 'foo'
        """)


def test_variables_assign_multiple():
    # not yet supported
    code = cst.parse_module(textwrap.dedent("""\
        g = h = 1.0

        class C:
            x, y = 'foo'

            def __init__(self, x):
                self.y = self.z = x
    """))

    t = mk_var_transformer('foo.py', code)

    code = t.transform_code(code)
    assert code.code == textwrap.dedent("""\
        g = h = 1.0

        class C:
            x, y = 'foo'

            def __init__(self, x) -> None:
                self.y = self.z = x
        """)


def test_variables_assign_tuple():
    # not yet supported
    code = cst.parse_module(textwrap.dedent("""\
        h, (g, i) = 1, (1.0, 0)

        class C:
            x, y = 'foo', True

            def __init__(self, x):
                self.y, self.z = x, None
    """))

    t = mk_var_transformer('foo.py', code)

    code = t.transform_code(code)
    assert code.code == textwrap.dedent("""\
        h, (g, i) = 1, (1.0, 0)

        class C:
            x, y = 'foo', True

            def __init__(self, x) -> None:
                self.y, self.z = x, None
        """)


def test_attribute_of_subscript():
    """Replicates a bug not sufficiently checking assignment targets in unified_transformer.
       Note that "self.x[0].x: int = 2" is not a valid annotation.
    """
    code = cst.parse_module(textwrap.dedent("""\
        from typing import NamedTuple, Self
        T = NamedTuple("T", ['x'])

        class C:
            def __init__(self):
                self.x = [T(0), T(1)]
                self.x[0].x: int = 2
                self.x[0].x = 2

        s = C()
        """))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                get_code_id('foo.py', code, 'C.__init__'): FuncAnnotation(
                    [],
                    TypeInfo("", "None"),
                    varargs=None, kwargs=None,
                    variables=[
                        (VariableName('self.x'), TypeInfo('', 'list', args=(TypeInfo("foo", "T"),)))
                    ]
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
            override_annotations=True,
            only_update_annotations=False,
            inline_generics=False,
        )

    code = t.transform_code(code)
    function = get_function(code, "C.__init__")
    assert function == textwrap.dedent("""\
        def __init__(self) -> None:
            self.x: list[T] = [T(0), T(1)]
            self.x[0].x: int = 2
            self.x[0].x = 2
        """)


@pytest.mark.parametrize('wrap_in, wrap_out', [
    ('Final', 'Final[{t}]'),
    ('ClassVar', 'ClassVar[{t}]'),
    ('Final[ClassVar]', 'Final[ClassVar[{t}]]'),
    ('ClassVar[Final]', 'ClassVar[Final[{t}]]'),
    ('Final[{t}]',) * 2,
    ('ClassVar[{t}]',) * 2,
    ('Final[ClassVar[{t}]]',) * 2,
    ('ClassVar[Final[{t}]]',) * 2
])
def test_variables_retains_final_and_classvar(wrap_in, wrap_out):
    code = cst.parse_module(textwrap.dedent(f"""\
        from typing import ClassVar, Final

        g: Final = 1.0

        class C:
            x: {wrap_in.format(t='int')} = 'foo'
        """))

    t = mk_var_transformer('foo.py', code)
    code = t.transform_code(code)

    assert code.code == textwrap.dedent(f"""\
        from typing import ClassVar, Final

        g: Final[float] = 1.0

        class C:
            x: {wrap_out.format(t='str')} = 'foo'
        """)


def test_type_keyword_is_known():
    code = cst.parse_module(textwrap.dedent("""\
        type MyStr = str

        def foo(x):
            pass
        """))
    t = UnifiedTransformer(
            filename='foo.py',
            type_annotations = {
                get_code_id('foo.py', code, 'foo'): FuncAnnotation(
                    [
                        (ArgumentName('x'), TypeInfo("foo", "MyStr"))
                    ],
                    TypeInfo("", "None"),
                    varargs=None, kwargs=None,
                    variables=[]
                ),
            },
            module_variables = ModuleVars([]),
            module_name='foo',
        )

    code = t.transform_code(code)
    function = get_function(code, "foo")
    assert function == textwrap.dedent("""\
        def foo(x: MyStr) -> None:
            pass
        """)

def test_transform_deletes_type_hint_for_variable():
    code = cst.parse_module(textwrap.dedent("""\
        g = 1.0  # typeepty something...  #  type: None
        h = 0 # not modified, leave as-is # type: int
    """))

    t = mk_var_transformer('foo.py', code)

    code = t.transform_code(code)
    assert code.code == textwrap.dedent("""\
        g: float = 1.0  # typeepty something...
        h = 0 # not modified, leave as-is # type: int
        """)
