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
    TypeInfoSet,
)
from righttyper.righttyper_utils import (
    debug_print,
    skip_this_file,
    source_to_module_fqn
)
from righttyper.unified_transformer import UnifiedTransformer

logger = logging.getLogger("righttyper")

SignatureChanges = tuple[Filename, list[tuple[FunctionName, str, str]]]


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
) -> SignatureChanges:
    debug_print(f"process_file: {filename}")
    try:
        with open(filename, "r") as file:
            source = file.read()
    except FileNotFoundError:
        return filename, []

    try:
        cst_tree = cst.parse_module(source)
    except cst._exceptions.ParserSyntaxError:  # type: ignore
        try:
            # Initial parse failed; fix any indentation issues and try again
            source = correct_indentation_issues(source)
            cst_tree = cst.parse_module(source)
        except cst._exceptions.ParserSyntaxError:  # type: ignore
            print(f"Failed to parse source for {filename}.")
            raise

    transformer = UnifiedTransformer(
        filename, type_annotations, ignore_annotations,
        module_name=source_to_module_fqn(pathlib.Path(filename)),
        module_names=module_names
    )

    try:
        transformed = cst_tree.visit(transformer)
    except TypeError:
        # This happens when "Mock" is passed around.
        print(f"Failed to transform {filename}.")
        raise

    changes = transformer.get_signature_changes()

    if output_files and changes:
        if overwrite:
            with open(filename + ".bak", "w") as file:
                file.write(source)

            with open(filename, "w") as file:
                file.write(transformed.code)

        else:
            with open(filename + ".typed", "w") as file:
                file.write(transformed.code)

    if generate_stubs:
        stub_file = pathlib.Path(filename).with_suffix(".pyi")

        stubs = transformed.visit(PyiTransformer())

        if stub_file.exists():
            stub_file.with_suffix(stub_file.suffix + ".bak").write_text(stub_file.read_text())

        stub_file.write_text(stubs.code)

    return filename, changes
