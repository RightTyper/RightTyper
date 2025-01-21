from typing import Self, cast
import ast


SEND_HANDLER = "send_handler"
SEND_WRAPPER = "rt___wrap_send"
ASEND_HANDLER = "asend_handler"
ASEND_WRAPPER = "rt___wrap_asend"


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

        for wrapper, handler in ((SEND_WRAPPER, SEND_HANDLER), (ASEND_WRAPPER, ASEND_HANDLER)):
            if any(
                isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and
                n.func.id == wrapper
                for n in ast.walk(node)
            ):
                new_import = ast.ImportFrom(
                    module="righttyper.righttyper",
                    names=[
                        ast.alias(name=handler, asname=wrapper)
                    ],
                    level=0
                )

                index = self.after_from_future(node)

                for n in ast.walk(new_import):
                    if "lineno" in n._attributes:
                        n.lineno = n.end_lineno = 0         # type: ignore
                        n.col_offset = n.end_col_offset = 0 # type: ignore

                node.body[index:index] = [new_import]

        return node


    def visit_Call(self: Self, node: ast.Call) -> ast.Call:
        node = cast(ast.Call, self.generic_visit(node))

        if isinstance(node.func, ast.Attribute) and node.func.attr in ("send", "asend"):
            is_sync = (node.func.attr == "send")
            new_node = ast.Call(
                func=ast.Name(id=SEND_WRAPPER if is_sync else ASEND_WRAPPER, ctx=ast.Load()),
                args=[
                    node.func.value,
                    *node.args
                ],
                keywords=node.keywords
            )

            for n in ast.walk(new_node):
                if "lineno" in n._attributes:
                    n.lineno = node.lineno          # type: ignore
                    n.end_lineno = node.end_lineno  # type: ignore
                    n.col_offset = node.col_offset  # type: ignore
                    n.end_col_offset = node.end_col_offset # type: ignore

            return new_node

        return node


def instrument(m: ast.Module) -> ast.Module:
    return GeneratorSendTransformer().visit(m)
