from typing import cast
import ast


RANDOM_DICT_NAME = 'RandomDict'
RANDOM_DICT_ASNAME = 'rt___RandomDict'


class DictTransformer(ast.NodeTransformer):
    def after_from_future(self, node: ast.Module) -> int:
        return next(reversed([
            i+1
            for i, n in enumerate(node.body)
            if isinstance(n, ast.ImportFrom)
            if n.module == '__future__'
        ]), 0)


    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = cast(ast.Module, self.generic_visit(node))

        # Add import if needed
        if any(
            isinstance(n, ast.Call) and
            isinstance(n.func, ast.Name) and
            n.func.id == RANDOM_DICT_ASNAME
            for n in ast.walk(node)
        ):
            new_import = ast.ImportFrom(
                module="righttyper.random_dict",
                names=[
                    ast.alias(name=RANDOM_DICT_NAME, asname=RANDOM_DICT_ASNAME)
                ],
                level=0,
            )

            ast.fix_missing_locations(new_import)

            index = self.after_from_future(node)
            node.body[index:index] = [new_import]

        return node


    def visit_Dict(self, node: ast.Dict) -> ast.Call:
        # Dictionary literals {}
        node = cast(ast.Dict, self.generic_visit(node))

        new_node = ast.Call(
            func=ast.Name(id=RANDOM_DICT_ASNAME, ctx=ast.Load()),
            args=[node],
            keywords=[],
        )

        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node


    def visit_DictComp(self, node: ast.DictComp) -> ast.Call:
        # Dictionary comprehensions
        node = cast(ast.DictComp, self.generic_visit(node))

        new_node = ast.Call(
            func=ast.Name(id=RANDOM_DICT_ASNAME, ctx=ast.Load()),
            args=[node],
            keywords=[],
        )
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node


    def visit_Call(self, node: ast.Call) -> ast.Call:
        # Calls to dict()
        node = cast(ast.Call, self.generic_visit(node))

        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            node.func.id = RANDOM_DICT_ASNAME

        return node
