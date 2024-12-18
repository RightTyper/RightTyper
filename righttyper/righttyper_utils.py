import logging
import os
import re
from functools import cache
from typing import Any, Final, cast

from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    Filename,
    FuncInfo,
    FunctionName,
    Typename,
    TypeInfoSet,
    TYPE_OBJ_TYPES
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


def union_typeset_str(typeinfoset: TypeInfoSet) -> Typename:
    if not typeinfoset:
        return Typename("None") # Never observed any types.

    if len(typeinfoset) == 1:
        return Typename(str(next(iter(typeinfoset))))

    if super := find_most_specific_common_superclass_by_name(typeinfoset):
        return super

    typeset = {Typename(str(t)) for t in typeinfoset}

    if Typename("None") in typeset:
        # "None" at the end is considered to be more readable
        return Typename(
            "|".join([*(t for t in sorted(typeset) if t != "None"), "None"])
        )

    return Typename("|".join(sorted(typeset)))


def find_most_specific_common_superclass_by_name(typeinfoset: TypeInfoSet) -> Typename|None:
    if any(t.type_obj is None for t in typeinfoset):
        return None

    common_superclasses = set.intersection(
        *(set(cast(TYPE_OBJ_TYPES, t.type_obj).__mro__) for t in typeinfoset)
    )

    common_superclasses.discard(object) # not specific enough to be useful

    if not common_superclasses:
        return None

    return Typename(
        max(
            common_superclasses,
            key=lambda cls: cls.__mro__.index(object),
        ).__name__
    )


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
