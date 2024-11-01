import ast
from typing import TextIO

from righttyper.righttyper_types import FunctionName

from typing import List, Self, Sequence
import libcst as cst

def generate_stub(
    input_file: str,
    output_file: str,
    exclude_functions: list[FunctionName],
) -> None:
    if exclude_functions is None:
        exclude_functions = []

    with open(input_file, "r") as file:
        tree = ast.parse(file.read())

    with open(output_file, "a") as stub_file:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                if node.name not in exclude_functions:
                    generate_function_stub(node, stub_file)
            elif isinstance(node, ast.ClassDef):
                generate_class_stub(node, stub_file)
            elif isinstance(node, ast.Assign):
                generate_variable_stub(node, stub_file)


def generate_function_stub(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    stub_file: TextIO,
) -> None:
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation).strip()}"
        args.append(arg_str)
    if node.args.vararg:
        vararg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg_str += (
                f": {ast.unparse(node.args.vararg.annotation).strip()}"
            )
        args.append(vararg_str)
    if node.args.kwarg:
        kwarg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(node.args.kwarg.annotation).strip()}"
        args.append(kwarg_str)
    args_str = ", ".join(args)
    if node.returns:
        returns = ast.unparse(node.returns).strip()
    else:
        returns = "None"
    stub_file.write(f"def {node.name}({args_str}) -> {returns}: ...\n")


def generate_class_stub(node: ast.ClassDef, stub_file: TextIO) -> None:
    stub_file.write(f"class {node.name}:\n")
    empty_body = True
    for elem in node.body:
        if isinstance(elem, ast.FunctionDef):
            if elem.name not in [
                "__init__",
                "__new__",
            ]:
                stub_file.write("    ")
                generate_function_stub(elem, stub_file)
                empty_body = False
        elif isinstance(elem, ast.Assign):
            stub_file.write("    ")
            generate_variable_stub(elem, stub_file)
            empty_body = False
    # FIXME TODO what about members?
    if empty_body:
        stub_file.write("    pass")
    stub_file.write("\n")


def generate_variable_stub(node: ast.Assign, stub_file: TextIO) -> None:
    for target in node.targets:
        if isinstance(target, ast.Name):
            if node.value:
                annotation = "Any"
                if isinstance(node.value, ast.Constant):
                    annotation = type(node.value.value).__name__
                stub_file.write(f"{target.id}: {annotation}\n")
            else:
                stub_file.write(f"{target.id}: Any\n")



class PyiTransformer(cst.CSTTransformer):
    # FIXME look for __all__ and, if defined, only export those names.
    # FIXME absent that, omit names starting with _ ?

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

