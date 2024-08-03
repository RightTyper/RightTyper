from collections import defaultdict
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Set,
    Tuple,
)

import libcst as cst
import logging
import os
import pathlib

from righttyper.annotate_function_transformer import (
    AnnotateFunctionTransformer,
)
from righttyper.construct_import_transformer import (
    ConstructImportTransformer,
)
from righttyper.generate_stubs import (
    generate_stub,
)
from righttyper.insert_typing_import_transformer import (
    InsertTypingImportTransformer,
)
from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    Filename,
    FuncInfo,
    FunctionName,
    ImportInfo,
    Typename,
    TypenameSet,
)
from righttyper.righttyper_utils import (
    adjusted_file_name,
    adjusted_type_name,
    debug_print,
    make_type_signature,
    skip_this_file,
    union_typeset_str,
)

logger = logging.getLogger("righttyper")


def preface_with_typing_import(
    source_code: str,
) -> str:
    tree = cst.parse_module(source_code)
    transformer = InsertTypingImportTransformer()
    new_tree = tree.visit(transformer)
    return new_tree.code


def process_file(
    filename: Filename,
    type_annotations: Dict[
        FuncInfo,
        Tuple[
            List[Tuple[ArgumentName, Typename]],
            Typename,
        ],
    ],
    imports: Set[ImportInfo],
    overwrite: bool,
    not_annotated: Dict[FuncInfo, Set[str]],
    ignore_annotations: bool = False,
    srcdir: str = "",
) -> None:
    debug_print(f"process_file: {filename}")
    # print(f"process {filename}: {imports}")
    try:
        with open(filename, "r") as file:
            source = file.read()
    except FileNotFoundError:
        return

    # Make a backup
    if overwrite:
        with open(filename + ".bak", "w") as file:
            file.write(source)

    # First, update all type annotations so they are relative to the
    # source directory (--srcdir)
    for fi in type_annotations:
        adj_fname = adjusted_file_name(fi.file_name)
        args, retval_type = type_annotations[fi]
        new_arglist = []
        for arg in args:
            new_arg_type = adjusted_type_name(adj_fname, arg[1])
            new_arglist.append((arg[0], new_arg_type))
        new_retval_type = adjusted_type_name(adj_fname, retval_type)
        type_annotations[fi] = (
            new_arglist,
            new_retval_type,
        )

    # Now, rewrite all function definitions with annotations.
    cst_tree = cst.parse_module(source)
    transformer = AnnotateFunctionTransformer(
        filename, type_annotations, not_annotated
    )
    modified_tree = cst_tree.visit(transformer)

    # No changes, just skip the rest
    if modified_tree.code == source:
        return

    # Add an import statement if needed.
    # FIXME: this messes with from __future__ imports, which need to be the first import
    transformed = preface_with_typing_import(modified_tree.code)

    # If there are needed imports for class defs, process these
    needed_imports = set(imp for imp in imports if imp.function_fname == filename)
    if needed_imports:
        tree = cst.parse_module(transformed)
        import_transformer = ConstructImportTransformer(
            imports=needed_imports,
            root_path=srcdir,
        )
        try:
            transformed_tree = tree.visit(import_transformer)
            transformed = transformed_tree.code
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(e)

    with open(
        filename + ("" if overwrite else ".typed"),
        "w",
    ) as file:
        file.write(transformed)


# Convert the collected data into the expected format for type_annotations
def collect_data(
    file_name: str,
    visited_funcs: Set[FuncInfo],
    visited_funcs_arguments: Dict[FuncInfo, List[ArgInfo]],
    visited_funcs_retval: Dict[FuncInfo, TypenameSet],
    namespace: Dict[str, Any] = globals(),
) -> Dict[
    FuncInfo,
    Tuple[
        List[Tuple[ArgumentName, Typename]],
        Typename,
    ],
]:
    type_annotations: Dict[
        FuncInfo,
        Tuple[
            List[Tuple[ArgumentName, Typename]],
            Typename,
        ],
    ] = {}
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
        type_annotations[t] = (
            arg_annotations,
            retval,
        )
        # print(f"{type_annotations[t]} {t}")
    return type_annotations


def output_stub_files(
    namespace: Dict[str, Any],
    root_path: str,
    imports: Set[ImportInfo],
    visited_funcs: Set[FuncInfo],
    script_dir: str,
    include_all: bool,
    include_files_regex: str,
    visited_funcs_arguments: Dict[FuncInfo, List[ArgInfo]],
    visited_funcs_retval: Dict[FuncInfo, TypenameSet],
    not_annotated: Dict[FuncInfo, Set[str]],
    arg_types: Dict[
        Tuple[Filename, FunctionName, ArgumentName],
        ArgumentType,
    ],
    existing_annotations: Dict[FuncInfo, Dict[str, str]],
) -> None:
    # Print all type signatures
    output_str: Dict[str, str] = defaultdict(str)  # map file name to str
    already_imported: Set[Tuple[str, str]] = set()
    skip_functions: Dict[str, List[FunctionName]] = defaultdict(list)

    import site
    import sysconfig

    # Paths to the main and user libraries, with an os separator added
    # Used to generate import statements later.
    purelib = sysconfig.get_paths()["purelib"]
    platstdlib = sysconfig.get_paths()["platstdlib"]
    userlib = site.getusersitepackages()

    # Precompute imports mapping
    imports_map = {}
    for imp in imports:
        if (
            imp.class_fname.startswith(purelib)
            or imp.class_fname.startswith(userlib)
            or imp.class_fname.startswith(platstdlib)
        ):
            if imp.class_fname.startswith(
                purelib
            ) or imp.class_fname.startswith(userlib):
                class_src_file = os.path.dirname(imp.class_fname)
                class_src_file = class_src_file.removeprefix(purelib + os.sep)
                class_src_file = class_src_file.removeprefix(userlib + os.sep)
            else:
                class_src_file = imp.class_fname.removeprefix(platstdlib)
                class_src_file = os.path.basename(class_src_file)
                class_src_file, _ = os.path.splitext(class_src_file)

            class_src_file = class_src_file.replace(os.sep, ".")
            imports_map[(imp.function_fname, imp.class_name)] = class_src_file
            continue
        if imp.class_fname == "":
            # Note: not sure why this is happening.
            continue
        try:
            normalized_path = os.path.relpath(imp.class_fname, start=root_path)
        except ValueError:
            logger.warning(f"ValueError: {imp.class_fname=} {root_path=}")
            continue
        module_path, _ = os.path.splitext(normalized_path)
        if module_path.startswith(".."):  # SOMETHING HAS GONE OF THE TRACKS
            module_path = os.path.abspath(module_path)
        module_path = module_path.replace(os.sep, ".")
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]  # remove '.__init__'
        assert not module_path.startswith(
            ".."
        )  # SOMETHING HAS GONE OF THE TRACKS

        imports_map[(imp.function_fname, imp.class_name)] = module_path

    for t in visited_funcs:
        if skip_this_file(
            t.file_name,
            script_dir,
            include_all,
            include_files_regex,
        ):
            continue
        try:
            stub_fname = str(pathlib.Path(t.file_name).with_suffix(".pyi"))

            # Add relevant imports
            for (
                function_file_path,
                class_name,
            ), module_path in imports_map.items():
                if (
                    stub_fname,
                    class_name,
                ) not in already_imported:
                    if not class_name.startswith("_"):
                        # We refuse to try to import names starting with underscores, at least for now.
                        output_str[
                            stub_fname
                        ] += f"from {module_path} import {class_name}\n"
                    already_imported.add((stub_fname, class_name))

            s = make_type_signature(
                file_name=t.file_name,
                func_name=t.func_name,
                args=visited_funcs_arguments[t],
                retval=visited_funcs_retval[t],
                namespace=namespace,
                not_annotated=not_annotated,
                arg_types=arg_types,
                existing_annotations=existing_annotations,
            )
            skip_functions[stub_fname].append(t.func_name)
            output_str[stub_fname] += f"{s} ...\n"

        except KeyError:
            # Something weird happened
            logger.warning(f"KeyError: {t=}")

    for (
        stub_fname,
        stub_content,
    ) in output_str.items():
        with open(stub_fname, "w") as f:
            f.write(stub_content)

        input_file = str(pathlib.Path(stub_fname).with_suffix(".py"))
        output_file = stub_fname
        exclude_functions = skip_functions[stub_fname]
        generate_stub(
            input_file,
            output_file,
            exclude_functions,
        )
