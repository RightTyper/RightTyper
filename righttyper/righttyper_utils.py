import logging
import os
import re
from functools import cache
from typing import Any, Final

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


def union_typeset_str(
    typeset: TypenameSet,
    namespace: dict[str, Any] = globals(),
    threshold_frequency: float = 0.25,
) -> Typename:
    adjusted_typeset = typeset  # trim_and_test(typeset, threshold_frequency)
    retval = None

    if not typeset:
        # Never observed any return types. Since we always sample at least once,
        # this means we did not return any values.
        return Typename("None")

    if len(typeset) == 1:
        return next(iter(typeset))

    if super := find_most_specific_common_superclass_by_name(
        list(adjusted_typeset),
        namespace
    ):
        return super

    if adjusted_typeset:
        typenames = sorted(adjusted_typeset)
        if len(typenames) == 1:
            return typenames[0]

        if "None" in typenames:
            # "None" at the end is considered to be more readable
            return Typename(
                "|".join([t for t in typenames if t != "None"]
                         + ["None"])
            )

        return Typename(
            "|".join(typenames)
        )

    return Typename("Any") # worst case


def find_most_specific_common_superclass_by_name(
    type_names: list[str],
    namespace: dict[str, Any] = globals(),
) -> Typename|None:
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
