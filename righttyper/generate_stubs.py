import ast
from typing import TextIO

from righttyper.righttyper_types import FunctionName


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
