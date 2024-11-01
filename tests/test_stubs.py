import textwrap
import libcst as cst

from typing import List, Self, Sequence

class PyiTransformer(cst.CSTTransformer):
    def __init__(self: Self) -> None:
        self._needs_any = False

    def leave_FunctionDef(
        self: Self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(
            body=cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())]),
            leading_lines=[]
        )

    def leave_Comment(    # type: ignore[override]
        self: Self,
        original_node: cst.Comment,
        updated_node: cst.Comment
        ) -> cst.RemovalSentinel:
        return cst.RemoveFromParent()

    def value2type(self: Self, value: cst.CSTNode) -> str:
        # FIXME not exhaustive; this should come from RightTyper typing
        if isinstance(value, cst.BaseString):
            return str.__name__
        elif isinstance(value, cst.Integer):
            return int.__name__
        elif isinstance(value, cst.Float):
            return float.__name__
        elif isinstance(value, cst.Tuple):
            return tuple.__name__
        elif isinstance(value, cst.BaseList):
            return list.__name__
        elif isinstance(value, cst.BaseDict):
            return dict.__name__
        elif isinstance(value, cst.BaseSet):
            return set.__name__
        self._needs_any = True
        return "Any"

    def handle_body(self: Self, body: Sequence[cst.CSTNode]) -> List[cst.CSTNode]:
        result: List[cst.CSTNode] = []
        for stmt in body:
            if isinstance(stmt, (cst.FunctionDef, cst.ClassDef, cst.If)):
                result.append(stmt)
            elif (isinstance(stmt, cst.SimpleStatementLine) and
                  isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))):
                result.append(stmt)
            elif (isinstance(stmt, cst.SimpleStatementLine) and isinstance(stmt.body[0], cst.Assign)):
                for target in stmt.body[0].targets:
                    type_ann = self.value2type(stmt.body[0].value)

                    result.append(cst.SimpleStatementLine(body=[
                        cst.AnnAssign(
                            target=target.target,
                            annotation=cst.Annotation(cst.Name(type_ann)),
                            value=None
                        )
                    ]))

        return result

    def leave_ClassDef(
        self: Self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=self.handle_body(updated_node.body.body)
            ),
            leading_lines=[]
        )

    def leave_If(
        self: Self,
        original_node: cst.If,
        updated_node: cst.If
    ) -> cst.If:
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=self.handle_body(updated_node.body.body)
            ),
            leading_lines=[]
        )

    def leave_Module(
        self: Self,
        original_node: cst.Module,
        updated_node: cst.Module
    ) -> cst.Module:
        updated_node = updated_node.with_changes(
            body=self.handle_body(updated_node.body)
        )

        if self._needs_any:
            imports = [
                i for i, stmt in enumerate(updated_node.body)
                if (isinstance(stmt, cst.SimpleStatementLine) and
                    isinstance(stmt.body[0], (cst.Import, cst.ImportFrom)))
            ]

            # TODO could check if it's already there
            position = imports[-1]+1 if imports else 0

            updated_node = updated_node.with_changes(
                body=(*updated_node.body[:position],
                      cst.SimpleStatementLine([
                          cst.ImportFrom(
                            module=cst.Name('typing'),
                            names=[cst.ImportAlias(cst.Name('Any'))]
                          ),
                      ]),
                      *updated_node.body[position:]
                )
            )

        return updated_node


def generate_stub(orig_code: str) -> str:
    m = cst.parse_module(orig_code)
#    print(m)
    m = m.visit(PyiTransformer())
#    print(m)
    return m.code


def test_generate_stubs(tmp_path, monkeypatch):
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

def test_generate_stubs_no_any(tmp_path, monkeypatch):
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


def test_generate_stubs_empty_class(tmp_path, monkeypatch):
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


def test_generate_stubs_conditional(tmp_path, monkeypatch):
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
