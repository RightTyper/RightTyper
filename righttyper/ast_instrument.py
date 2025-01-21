from typing import Self, cast
import ast


WRAPPER_NAME = "wrap_send"
WRAPPER_ASNAME = "rt___wrap_send"


class GeneratorSendTransformer(ast.NodeTransformer):
    def after_from_future(self, node: ast.Module) -> int:
        return next(reversed([
            i+1
            for i, n in enumerate(node.body)
            if isinstance(n, ast.ImportFrom)
            if n.module == '__future__'
        ]), 0)


    def visit_Module(self: Self, node: ast.Module) -> ast.Module:
        node = cast(ast.Module, self.generic_visit(node))

        if any(
            isinstance(n, ast.Call) and
            isinstance(n.func, ast.Name) and
            n.func.id == WRAPPER_ASNAME
            for n in ast.walk(node)
        ):
            new_import = ast.ImportFrom(
                module="righttyper.righttyper",
                names=[
                    ast.alias(name=WRAPPER_NAME, asname=WRAPPER_ASNAME)
                ],
                level=0
            )

            ast.fix_missing_locations(new_import)

            index = self.after_from_future(node)
            node.body[index:index] = [new_import]

        return node


    def visit_Attribute(self: Self, node: ast.Attribute) -> ast.Attribute|ast.Call:
        node = cast(ast.Attribute, self.generic_visit(node))

        if type(node.ctx) is ast.Load and node.attr in ("send", "asend"):
            new_node = ast.Call(
                func=ast.Name(id=WRAPPER_ASNAME, ctx=ast.Load()),
                args=[node],
                keywords=[]
            )

            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        return node


def instrument(m: ast.Module) -> ast.Module:
    return GeneratorSendTransformer().visit(m)
