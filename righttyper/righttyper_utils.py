import logging
import re
import os
import sys

from functools import cache
from typing import Any
from pathlib import Path
from types import CodeType

from righttyper.logger import logger
from righttyper.options import options


_SAMPLING_INTERVAL = 0.01


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

detected_test_files: set[str] = set()
detected_test_modules: set[str] = set()

def set_test_files_and_modules(files: set[str], modules: set[str]) -> None:
    detected_test_files.update(files)
    detected_test_modules.update(modules)

    # Clear caches, as these functions' results may now change
    is_test_module.cache_clear()
    skip_this_file.cache_clear()
    should_skip_function.cache_clear()


@cache
def is_test_module(m: str) -> bool:
    return bool(
        m in detected_test_modules
        or (
            (opt_test_modules := options.test_modules_re)
            and opt_test_modules.match(m)
        )
    )


@cache
def should_skip_function(code: CodeType) -> bool:
    if skip_this_file(code.co_filename):
        return True

    if (
        (include_functions := options.include_functions_re)
        and not include_functions.search(code.co_name)
    ):
        logger.debug(f"skipping function {code.co_name}")
        return True

    if not (code.co_flags & 0x2):
        import dis

        assert dis.COMPILER_FLAG_NAMES[0x2] == "NEWLOCALS"
        return True
    return False


@cache
def skip_this_file(filename: str) -> bool:
    #logger.debug(f"checking skip_this_file {filename=}")
    if options.include_all:
        should_skip = False
    else:
        should_skip = (
            filename.startswith("<")
            or (options.exclude_test_files and filename in detected_test_files)
            # FIXME how about packages installed with 'pip install -e' (editable)?
            or any(filename.startswith(p) for p in PYTHON_LIBS)
            or filename.startswith(RIGHTTYPER_PATH)
            or options.script_dir not in os.path.abspath(filename)
        )

    if not should_skip and (include_files := options.include_files_re):
        should_skip = not include_files.search(filename)
        if should_skip:
            logger.debug(f"skipping file {filename}")

    return should_skip


# TODO compare to https://mypy.readthedocs.io/en/stable/running_mypy.html#mapping-file-paths-to-modules
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
