import logging
import os
import pathlib
from collections import defaultdict
from typing import Any

import libcst as cst

from righttyper.generate_stubs import PyiTransformer
from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    Filename,
    FuncInfo,
    FuncAnnotation,
    FunctionName,
    Typename,
    TypenameSet,
)
from righttyper.righttyper_utils import (
    debug_print,
    make_type_signature,
    skip_this_file,
    union_typeset_str,
)
from righttyper.unified_transformer import UnifiedTransformer
from righttyper.righttyper_runtime import source_to_module_fqn

logger = logging.getLogger("righttyper")


def correct_indentation_issues(file_contents: str) -> str:
    """Return a string corresponding to the file contents, but with indentation issues fixed if needed."""
    original_lines = file_contents.splitlines(
        keepends=True
    )  # Preserve line endings

    indent_stack: list[int] = []
    corrected_lines = []

    for line_number, line in enumerate(original_lines, start=1):
        stripped_line = line.lstrip()

        if not stripped_line or stripped_line.startswith("#"):
            # Preserve empty lines and comments
            corrected_lines.append(line)
            continue

        leading_whitespace = line[: len(line) - len(stripped_line)]

        # Convert tabs to spaces (assuming 4 spaces per tab)
        corrected_leading_whitespace = leading_whitespace.replace(
            "\t", " " * 4
        )

        # Check for mixed tabs and spaces and correct them
        if " " in leading_whitespace and "\t" in leading_whitespace:
            # print(f"Line {line_number}: Mixed spaces and tabs detected. Correcting to spaces.")
            leading_whitespace = corrected_leading_whitespace

        indent_level = len(corrected_leading_whitespace)

        # Check if the indentation level matches the current stack
        if indent_stack and indent_level < indent_stack[-1]:
            # If a dedent is found, adjust the stack accordingly
            while indent_stack and indent_level < indent_stack[-1]:
                indent_stack.pop()

        # Push the new indentation level if a new block starts
        if stripped_line.endswith(":"):  # Detect start of a block
            indent_stack.append(indent_level)

        # Reconstruct the corrected line and append to corrected lines list
        corrected_line = corrected_leading_whitespace + stripped_line
        corrected_lines.append(corrected_line)

    # Join corrected lines into a single string
    corrected_content = "".join(corrected_lines)

    # Return the corrected content if changes were made, otherwise return original content
    if corrected_content != file_contents:
        return corrected_content
    else:
        return file_contents


def process_file(
    filename: Filename,
    output_files: bool,
    generate_stubs: bool,
    type_annotations: dict[FuncInfo, FuncAnnotation],
    overwrite: bool,
    module_names: list[str],
    ignore_annotations: bool = False,
    srcdir: str = "",
) -> None:
    debug_print(f"process_file: {filename}")
    try:
        with open(filename, "r") as file:
            source = file.read()
    except FileNotFoundError:
        return

    if output_files and overwrite:
        with open(filename + ".bak", "w") as file:
            file.write(source)


    try:
        cst_tree = cst.parse_module(source)
    except cst._exceptions.ParserSyntaxError:  # type: ignore
        try:
            # Initial parse failed; fix any indentation issues and try again
            source = correct_indentation_issues(source)
            cst_tree = cst.parse_module(source)
        except cst._exceptions.ParserSyntaxError:  # type: ignore
            print(f"Failed to parse source for {filename}.")
            return

    transformer = UnifiedTransformer(
        filename, type_annotations, ignore_annotations,
        module_name=source_to_module_fqn(pathlib.Path(filename)),
        module_names=module_names
    )

    try:
        transformed = cst_tree.visit(transformer)
    except TypeError as e:
        # This happens when "Mock" is passed around.
        # Print a warning and bail.
        print(f"Failed to transform {filename}. ({e})")
        return

    if output_files:
        with open(
            filename + ("" if overwrite else ".typed"),
            "w",
        ) as file:
            file.write(transformed.code)


    if generate_stubs:
        stub_file = pathlib.Path(filename).with_suffix(".pyi")

        stubs = transformed.visit(PyiTransformer())

        if stub_file.exists():
            stub_file.with_suffix(stub_file.suffix + ".bak").write_text(stub_file.read_text())

        stub_file.write_text(stubs.code)


# Convert the collected data into the expected format for type_annotations
def collect_data(
    file_name: str,
    visited_funcs: set[FuncInfo],
    visited_funcs_arguments: dict[FuncInfo, list[ArgInfo]],
    visited_funcs_retval: dict[FuncInfo, TypenameSet],
    namespace: dict[str, Any] = globals(),
) -> dict[FuncInfo, FuncAnnotation]:
    type_annotations: dict[FuncInfo, FuncAnnotation] = {}
    for t in visited_funcs:
        args = visited_funcs_arguments[t]
        arg_annotations = [
            (
                ArgumentName(arginfo.arg_name),
                union_typeset_str(
                    file_name,
                    arginfo.type_name_set,
                    namespace,
                ),
            )
            for arginfo in args
        ]
        if t in visited_funcs_retval:
            retval = union_typeset_str(
                file_name,
                visited_funcs_retval[t],
                namespace,
            )
        else:
            retval = Typename("None")
        type_annotations[t] = FuncAnnotation(
            arg_annotations,
            retval,
        )
        # print(f"{type_annotations[t]} {t}")
    return type_annotations
