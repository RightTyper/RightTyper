import logging
import re
import os
import sys

from functools import cache
from typing import Any, Final
from pathlib import Path


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


def _get_righttyper_path() -> str:
    import importlib.util
    spec = importlib.util.find_spec(__package__)
    assert spec is not None and spec.origin is not None
    return str(Path(spec.origin).parent)

RIGHTTYPER_PATH = _get_righttyper_path()


def _get_python_libs() -> tuple[str, ...]:
    import sysconfig

    return tuple(
        set(
            sysconfig.get_path(p)
            for p in ('stdlib', 'platstdlib', 'purelib', 'platlib')
        )
    )

PYTHON_LIBS = _get_python_libs()


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
            # FIXME how about packages installed with 'pip install -e' (editable)?
            or any(filename.startswith(p) for p in PYTHON_LIBS)
            or filename.startswith(RIGHTTYPER_PATH)
            or script_dir not in os.path.abspath(filename)
        )
    if include_files_pattern:
        should_skip = should_skip or not re.search(
            include_files_pattern, filename
        )
    return should_skip


def _source_relative_to_pkg(file: Path) -> Path|None:
    """Returns a Python source file's path relative to its package"""
    try:
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
    except:
        # file.resolve() may throw in case of symlink loops;
        # Also, torch._dynamo seems to throw Unsupported (see issue 93)
        pass

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
