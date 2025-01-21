import ast
from importlib.abc import MetaPathFinder, Loader, ExecutionLoader, PathEntryFinder
from importlib import machinery
from importlib.resources.abc import TraversableResources
import importlib.util
import functools
from pathlib import Path
import sys
import sysconfig
import types
import typing

from righttyper.ast_instrument import instrument

class RightTyperLoader(ExecutionLoader):
    def __init__(self, fullname: str, path: Path, orig_loader: Loader|None=None):
        self.fullname = fullname
        self.path = path.resolve()
        self.orig_loader = orig_loader

    # for compability with loaders supporting resources, used e.g. by sklearn
    def get_resource_reader(self, fullname: str) -> TraversableResources|None:
        if self.orig_loader and hasattr(self.orig_loader, 'get_resource_reader'):
            return self.orig_loader.get_resource_reader(fullname)
        return None

    def get_filename(self, fullname: str) -> str:
        return str(self.path)

    def get_source(self, fullname: str) -> str:
        return str(self.path.read_bytes(), "utf-8")

    def get_code(self, fullname: str) -> types.CodeType:
        tree = ast.parse(self.get_source(fullname))
        tree = instrument(tree)
        return compile(tree, str(self.path), "exec")

    def create_module(self, spec):
        if self.orig_loader:
            return self.orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        exec(self.get_code(''), module.__dict__)


class RightTyperMetaPathFinder(MetaPathFinder):
    def __init__(self: typing.Self) -> None:
        self._pylib_paths = {
            Path(sysconfig.get_path(p)).resolve()
            for p in ('stdlib', 'platstdlib', 'purelib', 'platlib')
        }


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
                filename.suffix == '.py' and
                not self._in_python_libs(filename)
            ):
                # For those that look like we can load, insert our loader
                spec.loader = RightTyperLoader(fullname, filename, spec.loader)

            return spec

        return None


class RightTyperPathEntryFinder(PathEntryFinder):
    def __init__(self, filename):
        self.filename = filename

    def find_spec(self, fullname: str, target: types.ModuleType|None=None) -> machinery.ModuleSpec|None:
        # we want to intercept runpy's loading of some script (".py") as __main__
        if fullname == "__main__":
            return importlib.util.spec_from_loader(
                fullname,
                RightTyperLoader(fullname, Path(self.filename))
            )

        return None


def path_entry_hook(filename: str) -> PathEntryFinder:
    # we want to intercept runpy's loading of some script (".py") as __main__
    if filename.endswith(".py"):
        return RightTyperPathEntryFinder(filename)

    raise ImportError()


class ImportManager:
    """A context manager that enables instrumentation while active."""

    def __init__(self) -> None:
        self.mpf = RightTyperMetaPathFinder()


    def __enter__(self) -> typing.Self:
        sys.meta_path.insert(0, self.mpf)
        sys.path_hooks.insert(0, path_entry_hook)
        return self


    def __exit__(self, *args) -> None:
        sys.meta_path.remove(self.mpf)
        sys.path_hooks.remove(path_entry_hook)
