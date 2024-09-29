import logging
import os
import re
from functools import cache
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Set,
    Tuple,
)

from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    Filename,
    FuncInfo,
    FunctionName,
    Typename,
    TypenameSet,
)

TOOL_ID: int = 3
TOOL_NAME: Final[str] = "righttyper"
_SAMPLING_INTERVAL = 0.01
_DEBUG_PRINT: bool = False

logger = logging.getLogger("righttyper")


def reset_sampling_interval() -> None:
    global _SAMPLING_INTERVAL
    _SAMPLING_INTERVAL = 1.0


def get_sampling_interval() -> float:
    return _SAMPLING_INTERVAL


def update_sampling_interval(
    instrumentation_overhead, target_overhead
) -> None:
    global _SAMPLING_INTERVAL
    if instrumentation_overhead < target_overhead:
        _SAMPLING_INTERVAL *= 0.9
    else:
        _SAMPLING_INTERVAL *= 1.2
    print(f"{_SAMPLING_INTERVAL=}")
    ## FIXME _SAMPLING_INTERVAL *= 1.5


def debug_print(args: Any, *varargs: Any, **kwargs: Any) -> None:
    if _DEBUG_PRINT:
        print(__file__, args, *varargs, **kwargs)


def debug_print_set_level(level: bool) -> None:
    _DEBUG_PRINT = level


@cache
def adjusted_file_name(file_name: str, start: str = "") -> str:
    return os.path.relpath(
        os.path.splitext(file_name)[0],
        start=start,
    ).replace(os.sep, ".")


def adjusted_type_name(fname: str, typename: str) -> Typename:
    # Adjust the type based on the context of the file_name
    new_typename = typename
    if typename.startswith(fname + "."):
        new_typename = typename[len(fname) + 1 :]
    return Typename(new_typename)


def unannotated(f: object, ignore_annotations: bool = False) -> Set[str]:
    """
    Returns a set of the unannotated arguments and, if
    unannotated, the return value (called "return"), for the
    given function.
    """
    import inspect

    assert inspect.isfunction(f)

    sig = inspect.signature(f)
    unannotated_args = set()

    for name, param in sig.parameters.items():
        if ignore_annotations or param.annotation is param.empty:
            unannotated_args.add(name)
    if ignore_annotations or sig.return_annotation == inspect.Signature.empty:
        unannotated_args.add("return")

    return unannotated_args


def union_typeset_str(
    file_name: str,
    typeset: TypenameSet,
    namespace: Dict[str, Any] = globals(),
    threshold_frequency: float = 0.25,
) -> Typename:
    adjusted_typeset = typeset  # trim_and_test(typeset, threshold_frequency)
    retval = None
    if not typeset:
        # Never observed any return types. Since we always sample at least once,
        # this means we did not return any values.
        retval = Typename("None")
    elif len(typeset) == 1:
        retval = adjusted_type_name(
            adjusted_file_name(file_name),
            list(typeset)[0].typename,
        )
    elif super := find_most_specific_common_superclass_by_name(
        list(
            adjusted_type_name(
                adjusted_file_name(file_name),
                t.typename,
            )
            for t in adjusted_typeset
        ),
        namespace,
    ):
        retval = super
    else:
        if adjusted_typeset:
            typenames = sorted(
                adjusted_type_name(
                    adjusted_file_name(file_name),
                    t.typename,
                )
                for t in adjusted_typeset
            )
            if len(typenames) == 1:
                return typenames[0]
            # Make a variant of typenames without Nones
            not_none_typenames = sorted(
                t.typename for t in adjusted_typeset if t.typename != "None"
            )
            # If None is one of the typenames, make the union an optional over the remaining types.
            if len(not_none_typenames) < len(typenames):
                if len(not_none_typenames) > 1:
                    return Typename(
                        "Optional[" + "Union[" + ", ".join(not_none_typenames) + "]" + "]"
                    )
                else:
                    return Typename(
                        "Optional[" + ", ".join(not_none_typenames) + "]"
                    )
                    
            if len(typenames) > 1:
                # Just Union everything.
                return Typename("Union[" + ", ".join(typenames) + "]")
            else:
                return typenames[0]
    if not retval:
        # Worst-case
        return Typename("Any")
    return retval


def find_most_specific_common_superclass_by_name(
    type_names: List[str],
    namespace: Dict[str, Any] = globals(),
) -> Optional[Typename]:
    if not type_names:
        return None

    try:
        classes = [namespace[name] for name in type_names]
    except KeyError:
        # Fail gracefully
        return None
    common_superclasses = set.intersection(
        *(set(cls.mro()) for cls in classes)
    )
    common_superclasses.discard(object)
    if not common_superclasses:
        return None

    return Typename(
        max(
            common_superclasses,
            key=lambda cls: cls.__mro__.index(object),
        ).__name__
    )


def make_type_signature(
    file_name: str,
    func_name: str,
    args: List[ArgInfo],
    retval: TypenameSet,
    namespace: Dict[str, Any],
    not_annotated: Dict[FuncInfo, Set[str]],
    arg_types: Dict[
        Tuple[FuncInfo, ArgumentName],
        ArgumentType,
    ],
    existing_annotations: Dict[FuncInfo, Dict[str, str]],
) -> str:
    # print(f"make_type_signature {file_name} {func_name} {args} {retval}")
    t = FuncInfo(
        Filename(file_name),
        FunctionName(func_name),
    )
    s = f"def {func_name}("
    for index, arginfo in enumerate(args):
        argname = arginfo.arg_name
        arg_type = arg_types[
            FuncInfo(Filename(file_name), FunctionName(func_name)),
            argname,
        ]
        if arg_type == ArgumentType.vararg:
            arg_prefix = "*"
        elif arg_type == ArgumentType.kwarg:
            arg_prefix = "**"
        else:
            assert arg_type == ArgumentType.positional
            arg_prefix = ""
        if argname not in existing_annotations[t]:  # not_annotated[t]:
            argtype_fullname_set = arginfo.type_name_set
            argtype_fullname = Typename(
                union_typeset_str(
                    file_name,
                    argtype_fullname_set,
                    namespace,
                )
            )

            s += f"{arg_prefix}{argname}: {argtype_fullname}"
        else:
            if argname in existing_annotations[t]:
                s += f"{arg_prefix}{argname}: {existing_annotations[t][argname]}"
            else:
                s += f"{arg_prefix}{argname}"

        if index < len(args) - 1:
            s += ", "
    s += ")"
    if "return" in existing_annotations[t]:
        retval_name = Typename(existing_annotations[t]["return"])
    else:
        # if "return" in not_annotated[t]:
        retval_name = union_typeset_str(file_name, retval, namespace)
    s += f" -> {retval_name}:"
    return s


@cache
def skip_this_file(
    filename: str,
    script_dir: str,
    include_all: bool,
    include_files_regex: str,
) -> bool:
    debug_print(
        f"checking skip_this_file: {script_dir=}, {filename=}, {include_files_regex=}"
    )
    if include_all:
        should_skip = False
    else:
        should_skip = (
            filename.startswith("<")
            or filename.startswith("/Library")
            or filename.startswith("/opt/homebrew/")
            or "/site-packages/" in filename
            or "righttyper.py" in filename
            or script_dir not in os.path.abspath(filename)
        )
    if include_files_regex:
        should_skip = should_skip or not re.search(
            include_files_regex, filename
        )
    return should_skip
