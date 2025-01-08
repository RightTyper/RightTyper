import logging
import re
import os
import sys

from functools import cache
from typing import Any, Final, cast, Iterator
import itertools
from pathlib import Path

from righttyper.righttyper_types import (
    Typename,
    TypeInfo,
    TypeInfoSet,
    TYPE_OBJ_TYPES
)

TOOL_ID: int = 3
TOOL_NAME: Final[str] = "righttyper"
_SAMPLING_INTERVAL = 0.01
_DEBUG_PRINT: bool = False

logger = logging.getLogger("righttyper")


def glob_translate_to_regex(r):
    if sys.version_info < (3, 13):
        # glob.translate not available until 3.13; use wcmatch's implementation
        from wcmatch import glob
        return glob.translate(r)[0][0]
    else:
        import glob
        return glob.translate(r)


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


def union_typeset(typeinfoset: TypeInfoSet) -> TypeInfo:
    if not typeinfoset:
        return TypeInfo.from_type(type(None)) # Never observed any types.

    if len(typeinfoset) == 1:
        return next(iter(typeinfoset))

    if super := find_most_specific_common_superclass_by_name(typeinfoset):
        return super

    # merge similar generics
    if any(t.args for t in typeinfoset):
        typeinfoset = TypeInfoSet({*typeinfoset})   # avoid modifying

        def group_key(t):
            return t.module, t.name, all(isinstance(arg, TypeInfo) for arg in t.args), len(t.args)
        group: Iterator[TypeInfo]|TypeInfoSet
        for (mod, name, all_info, nargs), group in itertools.groupby(
            sorted(typeinfoset, key=group_key),
            group_key
        ):
            if all_info:
                group = set(group)
                first = next(iter(group))
                typeinfoset -= group
                typeinfoset.add(first.replace(args=tuple(
                        union_typeset(TypeInfoSet({
                            cast(TypeInfo, member.args[i]) for member in group
                        }))
                        for i in range(nargs)
                    )
                ))

    # TODO merge jaxtyping annotations by shape

    return TypeInfo.from_set(typeinfoset)


def union_typeset_str(typeinfoset: TypeInfoSet) -> Typename:
    return Typename(str(union_typeset(typeinfoset)))


def find_most_specific_common_superclass_by_name(typeinfoset: TypeInfoSet) -> TypeInfo|None:
    if any(t.type_obj is None for t in typeinfoset):
        return None

    common_superclasses = set.intersection(
        *(set(cast(TYPE_OBJ_TYPES, t.type_obj).__mro__) for t in typeinfoset)
    )

    common_superclasses.discard(object) # not specific enough to be useful

    if not common_superclasses:
        return None

    specific = max(
            common_superclasses,
            key=lambda cls: cls.__mro__.index(object),
    )

    module = specific.__module__ if specific.__module__ != '__main__' else get_main_module_fqn()
    return TypeInfo(module, specific.__qualname__, type_obj=specific)


@cache
def skip_this_file(
    filename: str,
    script_dir: str,
    include_all: bool,
    include_files_pattern: str,
) -> bool:
    debug_print(
        f"checking skip_this_file: {script_dir=}, {filename=}, {include_files_pattern=}"
    )
    if include_all:
        should_skip = False
    else:
        should_skip = (
            filename.startswith("<")
            or filename.startswith("/Library")
            or filename.startswith("/opt/homebrew/")
            or os.sep + "site-packages" + os.sep in filename
            or "righttyper.py" in filename
            or script_dir not in os.path.abspath(filename)
        )
    if include_files_pattern:
        should_skip = should_skip or not re.search(
            include_files_pattern, filename
        )
    return should_skip


def _source_relative_to_pkg(file: Path) -> Path|None:
    """Returns a Python source file's path relative to its package"""
    if not file.is_absolute():
        file = file.resolve()

    parents = list(file.parents)

    for d in sys.path:
        path = Path(d)
        if not path.is_absolute():
            path = path.resolve()

        for p in parents:
            if p == path:
                return file.relative_to(p)

    return None


def source_to_module_fqn(file: Path) -> str|None:
    """Returns a source file's fully qualified package name, if possible."""
    if not (path := _source_relative_to_pkg(file)):
        return None

    path = path.parent if path.name == '__init__.py' else path.parent / path.stem
    return '.'.join(path.parts)


@cache
def get_main_module_fqn() -> str:
    main = sys.modules['__main__']
    if hasattr(main, "__file__") and main.__file__:
        if fqn := source_to_module_fqn(Path(main.__file__)):
            return fqn

    return "__main__"
