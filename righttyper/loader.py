import ast
from importlib.abc import MetaPathFinder, Loader
from importlib import machinery
from importlib.resources.abc import TraversableResources
import functools
from pathlib import Path
import sys
import sysconfig
import types
import typing

from righttyper.ast_instrument import instrument

class RightTyperLoader(Loader):
    def __init__(self, orig_loader: Loader, filename: Path):
        self.orig_loader = orig_loader  # original loader we're wrapping
        self.filename = filename

    # for compability with loaders supporting resources, used e.g. by sklearn
    def get_resource_reader(self, fullname: str) -> TraversableResources|None:
        if hasattr(self.orig_loader, 'get_resource_reader'):
            return self.orig_loader.get_resource_reader(fullname)
        return None

    def create_module(self, spec):
        return self.orig_loader.create_module(spec)

    def get_code(self, name=None):   # expected by pyrun
        tree = ast.parse(self.filename.read_bytes())
        tree = instrument(tree)
        return compile(tree, str(self.filename), "exec")

    def exec_module(self, module):
        exec(self.get_code(), module.__dict__)


class RightTyperMetaPathFinder(MetaPathFinder):
    def __init__(self: typing.Self) -> None:
        self._pylib_paths = (
            Path(sysconfig.get_path("stdlib")).resolve(),
            Path(sysconfig.get_path("purelib")).resolve(),
        )


    @functools.cache
    def _in_python_libs(self: typing.Self, filename: Path) -> bool:
        if any(filename.is_relative_to(p) for p in self._pylib_paths):
            return True

        return False


    def find_spec(
        self: typing.Self,
        fullname: str,
        path: typing.Sequence[str]|None,
        target: types.ModuleType|None = None
    ) -> machinery.ModuleSpec | None:
        # Enlist the help of other loaders to get a spec
        for f in sys.meta_path:
            if (
                isinstance(f, self.__class__) or
                not hasattr(f, "find_spec") or
                (spec := f.find_spec(fullname, path, target)) is None or
                spec.loader is None
            ):
                # this loader can't help us, try another
                continue

            if (
                spec.origin and
                spec.origin != 'built-in' and
                not isinstance(spec.loader, machinery.ExtensionFileLoader) and
                (filename := Path(spec.origin)).exists() and
                not self._in_python_libs(filename)
            ):
                # For those that look like we can load, insert our loader
                spec.loader = RightTyperLoader(spec.loader, filename)

            return spec

        return None


class ImportManager:
    """A context manager that enables instrumentation while active."""

    def __init__(self) -> None:
        self.mpf = RightTyperMetaPathFinder()


    def __enter__(self) -> typing.Self:
        sys.meta_path.insert(0, self.mpf)
        return self


    def __exit__(self, *args) -> None:
        i = 0
        while i < len(sys.meta_path):
            if sys.meta_path[i] is self.mpf:
                sys.meta_path.pop(i)
                break
            i += 1
