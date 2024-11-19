from typing import Self
import collections.abc as abc
import libcst as cst


class PyiTransformer(cst.CSTTransformer):
    def __init__(self: Self) -> None:
        self._needs_any = False

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

    def handle_body(self: Self, body: abc.Sequence[cst.CSTNode]) -> list[cst.CSTNode]:
        result: list[cst.CSTNode] = []
        for stmt in body:
            if isinstance(stmt, (cst.FunctionDef, cst.ClassDef, cst.If, cst.Try, cst.With)):
                result.append(stmt)
            elif (isinstance(stmt, cst.SimpleStatementLine) and
                  isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))):
                result.append(stmt)
            elif (isinstance(stmt, cst.SimpleStatementLine) and isinstance(stmt.body[0], cst.Assign)):
                if not all(isinstance(target.target, cst.Name) for target in stmt.body[0].targets):
                    # can't handle tuples... do we need to?
                    continue

                if any (isinstance(target.target, cst.Name) and target.target.value == '__all__'
                        for target in stmt.body[0].targets):
                    result.append(stmt)
                    continue

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
            )
        )

    def leave_With(
        self: Self,
        original_node: cst.With,
        updated_node: cst.With
    ) -> cst.With:
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=self.handle_body(updated_node.body.body)
            )
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
